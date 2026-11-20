import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_TEXT = "/help, /memory, /session, /reset, /skill, /exit"
WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)
HELP_DETAILS = "\n".join(
    [
        "Commands:",
        "/help    Show this help message.",
        "/memory  Show the agent's distilled working memory.",
        "/session Show the path to the saved session file.",
        "/reset   Clear the current session history and memory.",
        "/skill   Run a prompt with a skill: /skill <name> <prompt>. Bare /skill lists skills.",
        "/exit    Exit the agent.",
    ]
)
MAX_TOOL_OUTPUT = 16000
MAX_HISTORY = 12000
RECENT_TOOL_OUTPUT = 3000
OUTPUT_CHUNK_SIZE = 2800
IGNORED_PATH_NAMES = {".git", ".mini-coding-agent", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}

##############################
#### Six Agent Components ####
##############################
# 1) Live Repo Context -> WorkspaceContext
# 2) Prompt Shape And Cache Reuse -> build_prefix, memory_text, prompt
# 3) Structured Tools, Validation, And Permissions -> build_tools, run_tool, validate_tool, approve, parse, path, tool_*
# 4) Context Reduction And Output Management -> clip, history_text
# 5) Transcripts, Memory, And Resumption -> SessionStore, record, note_tool, ask, reset
# 6) Delegation And Bounded Subagents -> tool_delegate


def now():
    return datetime.now(timezone.utc).isoformat()


# Supporting helper for component 4 (context reduction and output management).
def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def retain_tool_output(text):
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n[discarded {len(text) - MAX_TOOL_OUTPUT} characters beyond the retention limit; unavailable for retrieval]"


def safe_terminal_text(text):
    return "".join(
        char if char in "\n\t" or not unicodedata.category(char).startswith("C")
        else f"\\u{ord(char):04x}"
        for char in text
    )


def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


SKILL_NAME_RE = re.compile(r"^[a-z0-9-]+$")
SKILL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def parse_frontmatter(text):
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter")
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        raise ValueError("unterminated frontmatter")
    meta = {}
    for line in lines[1:end]:
        if not line.strip() or line[0] in " \t" or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, "".join(lines[end + 1:])


def load_skill(path, root):
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"unreadable: {exc}") from exc
    meta, body = parse_frontmatter(text)
    name = meta.get("name", "")
    description = meta.get("description", "")
    if not name:
        raise ValueError("missing name")
    if not description:
        raise ValueError("missing description")
    if name[0] in "|>" or description[0] in "|>":
        raise ValueError("block scalar not supported")
    if not SKILL_NAME_RE.match(name):
        raise ValueError("invalid name")
    return {
        "name": name,
        "description": description,
        "path": path.resolve(),
        "rel": path.relative_to(root).as_posix(),
        "body": body,
    }


def discover_skills(root):
    root = Path(root)
    skills_dir = root / "skills"
    if not skills_dir.is_dir():
        return [], []
    skills, warnings, seen = [], [], set()
    for folder in sorted((item for item in skills_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        path = folder / "SKILL.md"
        if not path.is_file():
            continue
        label = f"warning: skipped skills/{folder.name}/SKILL.md: "
        try:
            skill = load_skill(path, root)
        except ValueError as exc:
            warnings.append(label + str(exc))
            continue
        if skill["name"] in seen:
            warnings.append(label + f"duplicate skill name '{skill['name']}'")
            continue
        seen.add(skill["name"])
        skills.append(skill)
    return skills, warnings


def skill_cap_for_model(model_name, override=None):
    if override is not None:
        return override
    size = None
    if model_name:
        name = str(model_name)
        for candidate in (name.rsplit(":", 1)[-1], name):
            match = SKILL_SIZE_RE.search(candidate)
            if match:
                size = float(match.group(1))
                break
    if size is None or size > 13:
        return 24000
    return 4000 if size <= 4 else 12000


def truncate_skill(body, cap):
    if len(body) <= cap:
        return body
    return body[:cap] + f"\n[skill truncated at {cap} chars]"


##############################
#### 1) Live Repo Context ####
##############################
class WorkspaceContext:
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        docs = {}
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main").removeprefix("origin/"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return "\n".join([
            "Workspace:",
            f"- cwd: {self.cwd}",
            f"- repo_root: {self.repo_root}",
            f"- branch: {self.branch}",
            f"- default_branch: {self.default_branch}",
            "- status:",
            self.status,
            "- recent_commits:",
            commits,
            "- project_docs:",
            docs,
        ])


##############################
#### 5) Session Memory #######
##############################
class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session):
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Ollama request timed out after {self.timeout} seconds: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON: {exc}") from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")


class MiniAgent:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        skills=None,
        skill_max_chars=None,
        model_name=None,
        show_command_output=False,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.show_command_output = show_command_output
        self.skill_cap = skill_cap_for_model(model_name, skill_max_chars)
        discovered = skills is None
        if discovered:
            skills, warnings = discover_skills(self.root)
            for warning in warnings:
                print(warning, file=sys.stderr)
        self.skills = skills
        self.active_skills = {}
        self.current_request = ""
        self._skill_read = None
        resumed = session is not None
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": {"task": "", "files": [], "notes": []},
        }
        used = self.session.get("skills_used")
        self.session["skills_used"] = used if isinstance(used, list) else []
        for item in self.session["history"]:
            if item["role"] == "tool" and "result_id" not in item:
                item["result_id"] = uuid.uuid4().hex
        if resumed and discovered:
            known = {skill["name"] for skill in self.skills}
            warned = set()
            for entry in self.session["skills_used"]:
                missing = entry.get("name") if isinstance(entry, dict) else None
                if missing and missing not in known and missing not in warned:
                    warned.add(missing)
                    print(f"warning: session used skill '{missing}' which is no longer on disk", file=sys.stderr)
        self.tools = self.build_tools()
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    ###############################################
    #### 3) Structured Tools And Permissions ######
    ###############################################
    def build_tools(self):
        tools = {
            "list_files": {
                "schema": {"path": "str='.'"},
                "risky": False,
                "description": "List files in the workspace.",
                "run": self.tool_list_files,
            },
            "read_file": {
                "schema": {"path": "str", "start": "int=1", "end": "int=200"},
                "risky": False,
                "description": "Read a UTF-8 file by line range.",
                "run": self.tool_read_file,
            },
            "read_tool_output": {
                "schema": {"id": "str", "start": "int", "end": "int"},
                "risky": False,
                "description": "Retrieve a retained tool result by ID and zero-based, end-exclusive character range (max 2800 characters).",
                "run": self.tool_read_tool_output,
            },
            "search": {
                "schema": {"pattern": "str", "path": "str='.'"},
                "risky": False,
                "description": "Search the workspace with rg or a simple fallback.",
                "run": self.tool_search,
            },
            "run_shell": {
                "schema": {"command": "str", "timeout": "int=20"},
                "risky": True,
                "description": "Run a shell command in the repo root.",
                "run": self.tool_run_shell,
            },
            "write_file": {
                "schema": {"path": "str", "content": "str"},
                "risky": True,
                "description": "Write a text file.",
                "run": self.tool_write_file,
            },
            "patch_file": {
                "schema": {"path": "str", "old_text": "str", "new_text": "str"},
                "risky": True,
                "description": "Replace one exact text block in a file.",
                "run": self.tool_patch_file,
            },
        }
        if self.depth < self.max_depth:
            tools["delegate"] = {
                "schema": {"task": "str", "max_steps": "int=3"},
                "risky": False,
                "description": "Ask a bounded read-only child agent to investigate.",
                "run": self.tool_delegate,
            }
        return tools

    ############################################
    #### 2) Prompt Shape And Cache Reuse #######
    ############################################
    def build_prefix(self):
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                "<final>Done.</final>",
            ]
        )
        rules = "\n".join([
            "- Use tools instead of guessing about the workspace.",
            "- Return exactly one <tool>...</tool> or one <final>...</final>.",
            "- Tool calls must look like:",
            '  <tool>{"name":"tool_name","args":{...}}</tool>',
            "- For write_file and patch_file with multi-line text, prefer XML style:",
            '  <tool name="write_file" path="file.py"><content>...</content></tool>',
            "- Final answers must look like:",
            "  <final>your answer</final>",
            "- Never invent tool results.",
            "- Keep answers concise and concrete.",
            "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
            "- Before writing tests for existing code, read the implementation first.",
            "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
            "- New files should be complete and runnable, including obvious imports.",
            "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
            "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}.",
        ])
        sections = [
            "You are Mini-Coding-Agent, a small local coding agent running through Ollama.",
            "Rules:\n" + rules,
            "Tools:\n" + tool_text,
            "Valid response examples:\n" + examples,
        ]
        if self.skills:
            skill_lines = "\n".join(f"- {s['name']}: {s['description']} ({s['rel']})" for s in self.skills)
            sections.append(
                "Available skills:\n" + skill_lines + "\n"
                "Skill rules:\n"
                "- When the task matches a skill's description, read its SKILL.md with read_file first, then follow it.\n"
                "- Skill instructions apply only to the current request; read the skill again if needed later."
            )
        sections.append(self.workspace.text())
        return "\n\n".join(sections)

    def memory_text(self):
        memory = self.session["memory"]
        notes = "\n".join(f"- {note}" for note in memory["notes"]) or "- none"
        lines = [
            "Memory:",
            f"- task: {memory['task'] or '-'}",
            f"- files: {', '.join(memory['files']) or '-'}",
            "- notes:",
            notes,
        ]
        used = []
        for entry in self.session["skills_used"]:
            name = entry.get("name") if isinstance(entry, dict) else None
            if name and name not in used:
                used.append(name)
        if used or self.active_skills:
            lines.append(f"- skills_used: {', '.join(used) or '-'}")
            lines.append(f"- skills_active: {', '.join(self.active_skills) or '-'}")
        return "\n".join(lines)

    #####################################################
    #### 4) Context Reduction And Output Management #####
    #####################################################
    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        entries = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        skipped_reads = set()
        for index in range(len(history) - 1, -1, -1):
            item = history[index]
            if item["role"] == "tool" and item["name"] in ("write_file", "patch_file"):
                seen_reads.discard(str(item["args"].get("path", "")))
            if item["role"] == "tool" and item["name"] == "read_file" and index < recent_start:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    skipped_reads.add(index)
                seen_reads.add(path)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if index in skipped_reads:
                continue

            if item["role"] == "tool":
                limit = RECENT_TOOL_OUTPUT if recent else 180
                args = clip(json.dumps(item["args"], sort_keys=True), 1000)
                result_id = item.get("result_id", "")
                content = item["content"]
                entry = f"[tool:{item['name']} id:{result_id}] {args}\n{clip(content, limit)}"
                if "original_length" in item and len(content) > limit:
                    discarded = item["original_length"] - item["retained_length"]
                    entry += f"\n[discarded {discarded} characters beyond the retention limit; unavailable for retrieval]"
                if len(content) > limit:
                    entry += f"\n[Use read_tool_output with id {result_id} to retrieve retained content.]"
                entries.append(entry)
            else:
                limit = 900 if recent else 220
                entries.append(f"[{item['role']}] {clip(item['content'], limit)}")

        marker = "[... earlier transcript entries omitted]"
        selected = []
        size = len(marker) + 1
        for entry in reversed(entries):
            extra = len(entry) + (1 if selected else 0)
            if size + extra > MAX_HISTORY:
                break
            selected.append(entry)
            size += extra
        selected.reverse()
        if len(selected) < len(entries):
            selected.insert(0, marker)
        return "\n".join(selected)

    ########################################################
    #### 2) Prompt Shape And Cache Reuse (Continued) #######
    ########################################################
    def prompt(self, user_message):
        sections = [self.prefix, self.memory_text()]
        if self.active_skills:
            sections.append("The delegate tool is unavailable while skills are active; perform the work using the other tools.")
            sections.append(
                "Skill note: the skills below are already loaded. Follow them now; "
                "do not read their SKILL.md again."
            )
            bodies = "\n\n".join(f"### {name}\n{body}" for name, body in self.active_skills.items())
            sections.append("Active skills:\n" + bodies)
        sections.append("Transcript:\n" + self.history_text())
        sections.append("Current user request:\n" + user_message)
        return "\n\n".join(sections)

    ###############################################
    #### 5) Session Memory (Continued) ###########
    ###############################################
    def record(self, item):
        if item["role"] == "tool" and "result_id" not in item:
            item["result_id"] = uuid.uuid4().hex
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name, args, result):
        memory = self.session["memory"]
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and path:
            self.remember(memory["files"], str(path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory["notes"], note, 5)

    def ask(self, user_message, forced_skill=None):
        self.current_request = user_message
        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        try:
            if forced_skill is not None:
                self.activate_skill(forced_skill, forced_skill["body"])
            return self._ask_loop(user_message)
        finally:
            self.active_skills = {}

    def activate_skill(self, skill, body):
        capped = truncate_skill(body, self.skill_cap)
        self.active_skills[skill["name"]] = capped
        request = clip(self.current_request, 300)
        for entry in self.session["skills_used"]:
            if isinstance(entry, dict) and entry.get("name") == skill["name"] and entry.get("request") == request:
                break
        else:
            self.session["skills_used"].append({"name": skill["name"], "request": request, "at": now()})
        return capped

    def skill_for_path(self, resolved):
        for skill in self.skills:
            if skill["path"] == resolved:
                return skill
        return None

    def _ask_loop(self, user_message):
        memory = self.session["memory"]
        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        failure_reason = None
        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            try:
                raw = self.model_client.complete(self.prompt(user_message), self.max_new_tokens)
            except RuntimeError as exc:
                failure_reason = f"model request failed: {exc}"
                break
            kind, payload = self.parse(raw)

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                result = self.run_tool(name, args)
                if self._skill_read is not None:
                    skill = self._skill_read
                    self._skill_read = None
                    result = f"[skill loaded: {skill['name']} - {skill['description']}]"
                event = {
                    "role": "tool",
                    "name": name,
                    "args": args,
                    "content": result,
                    "created_at": now(),
                }
                if self._result_original_length is not None and self._result_original_length > MAX_TOOL_OUTPUT:
                    event["retained_length"] = MAX_TOOL_OUTPUT
                    event["original_length"] = self._result_original_length
                self.record(event)
                self.note_tool(name, args, result)
                continue

            if kind == "retry":
                self.record({"role": "assistant", "content": payload, "created_at": now()})
                continue

            final = (payload or raw).strip()
            self.record({"role": "assistant", "content": final, "created_at": now()})
            self.remember(memory["notes"], clip(final, 220), 5)
            return final

        if failure_reason is None:
            if attempts >= max_attempts and tool_steps < self.max_steps:
                failure_reason = "too many malformed model responses"
            else:
                failure_reason = "tool-call step limit reached"
        final_prompt = (
            self.prompt(user_message)
            + f"\n\nFinal-answer turn: {failure_reason}. No tools are available. "
            "Return a <final> answer based on available results, or report what could not be completed."
        )
        try:
            raw = self.model_client.complete(final_prompt, self.max_new_tokens)
        except RuntimeError as exc:
            final = f"Final-answer turn failed after {failure_reason}: {exc}"
        else:
            kind, payload = self.parse(raw)
            if kind == "final":
                final = payload.strip()
            elif kind == "tool":
                final = f"Final-answer turn failed after {failure_reason}: model requested a tool; no tool was executed."
            else:
                final = f"Final-answer turn failed after {failure_reason}: model returned a malformed or empty answer."
        self.record({"role": "assistant", "content": final, "created_at": now()})
        self.remember(memory["notes"], clip(final, 220), 5)
        return final

    #############################################################
    #### 3) Structured Tools, Validation, And Permissions #######
    #############################################################
    def run_tool(self, name, args):
        self._skill_read = None
        self._result_original_length = None

        def finish(result):
            self._result_original_length = len(result)
            return retain_tool_output(result)

        tool = self.tools.get(name)
        if tool is None:
            return finish(f"error: unknown tool '{name}'")
        if name == "delegate" and self.active_skills:
            return finish("error: delegate is unavailable while skills are active")
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            return finish(message)
        if self.repeated_tool_call(name, args):
            return finish(f"error: repeated identical tool call for {name}; choose a different tool or return a final answer")
        if tool["risky"] and not self.approve(name, args):
            return finish(f"error: approval denied for {name}")
        try:
            result = tool["run"](args)
        except Exception as exc:
            return finish(f"error: tool {name} failed: {exc}")
        if self._skill_read is not None:
            return result
        return finish(result)

    def repeated_tool_call(self, name, args):
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    def tool_example(self, name):
        examples = {
            "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            "read_tool_output": '<tool>{"name":"read_tool_output","args":{"id":"RESULT_ID","start":0,"end":1000}}</tool>',
            "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
            "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
        }
        return examples.get(name, "")

    def validate_tool(self, name, args):
        args = args or {}

        if name == "list_files":
            path = self.path(args.get("path", "."))
            if not path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "read_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            if self.skill_for_path(path) is not None:
                return  # skill files are read whole; line ranges are ignored
            start = int(args.get("start", 1))
            end = int(args.get("end", 200))
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "read_tool_output":
            result_id = args.get("id")
            if not isinstance(result_id, str) or not result_id:
                raise ValueError("id must be a non-empty string")
            start, end = args.get("start"), args.get("end")
            if type(start) is not int or type(end) is not int or start < 0 or end <= start:
                raise ValueError("invalid character range")
            if end - start > OUTPUT_CHUNK_SIZE:
                raise ValueError(f"range exceeds {OUTPUT_CHUNK_SIZE} characters")
            return

        if name == "search":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            self.path(args.get("path", "."))
            return

        if name == "run_shell":
            command = str(args.get("command", "")).strip()
            if not command:
                raise ValueError("command must not be empty")
            timeout = int(args.get("timeout", 20))
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "write_file":
            path = self.path(args["path"])
            if path.exists() and path.is_dir():
                raise ValueError("path is a directory")
            if "content" not in args:
                raise ValueError("missing content")
            return

        if name == "patch_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            old_text = str(args.get("old_text", ""))
            if not old_text:
                raise ValueError("old_text must not be empty")
            if "new_text" not in args:
                raise ValueError("missing new_text")
            text = path.read_text(encoding="utf-8")
            count = text.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once, found {count}")
            return

        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            task = str(args.get("task", "")).strip()
            if not task:
                raise ValueError("task must not be empty")
            return

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        raw = str(raw)
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = MiniAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", MiniAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", MiniAgent.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", MiniAgent.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", MiniAgent.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = MiniAgent.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", MiniAgent.retry_notice()
        if "<final>" in raw:
            final = MiniAgent.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", MiniAgent.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", MiniAgent.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = MiniAgent.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = MiniAgent.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"] = {"task": "", "files": [], "notes": []}
        self.session["skills_used"] = []
        self.active_skills = {}
        self.skills, warnings = discover_skills(self.root)
        for warning in warnings:
            print(warning, file=sys.stderr)
        self.prefix = self.build_prefix()
        self.session_store.save(self.session)

    def parse_skill_command(self, text):
        parts = text.split(None, 2)
        if len(parts) == 1:
            return "list", None
        skill = next((s for s in self.skills if s["name"] == parts[1]), None)
        if skill is None:
            return "unknown", parts[1]
        if len(parts) < 3:
            return "usage", None
        return "run", (skill, parts[2].strip())

    def path_is_within_root(self, resolved):
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not self.path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

    def tool_list_files(self, args):
        path = self.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(self.root)}")
        return "\n".join(lines) or "(empty)"

    def tool_read_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        skill = self.skill_for_path(path)
        if skill is not None:
            _, body = parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
            capped = self.activate_skill(skill, body)
            self._skill_read = skill
            return capped
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
        return f"# {path.relative_to(self.root)}\n{body}"

    def tool_read_tool_output(self, args):
        for item in self.session["history"]:
            if item["role"] == "tool" and item.get("result_id") == args["id"]:
                content = item["content"][:item.get("retained_length", len(item["content"]))]
                start, end = args["start"], args["end"]
                if end > len(content):
                    raise ValueError(f"range exceeds retained content length {len(content)}; unavailable or discarded")
                header = f"result {args['id']} characters {start}-{end} of {len(content)}"
                if "original_length" in item:
                    header += f" (discarded {item['original_length'] - item['retained_length']} characters)"
                return f"{header}:\n{content[start:end]}"
        raise ValueError("tool result ID not found in current session")

    def tool_search(self, args):
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = self.path(args.get("path", "."))

        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches = []
        files = [path] if path.is_file() else [
            item for item in path.rglob("*")
            if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(self.root).parts)
        ]
        for file_path in files:
            for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(self.root)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def tool_run_shell(self, args):
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            status = f"exit_code: {result.returncode}"
            stdout, stderr = result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            status = f"error: command timed out after {timeout} seconds"
            stdout, stderr = exc.stdout, exc.stderr
        except OSError as exc:
            status = f"error: command could not start: {exc}"
            stdout, stderr = None, None

        def output_text(value):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            return (value or "").strip() or "(empty)"

        stdout, stderr = output_text(stdout), output_text(stderr)
        output = f"{status}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        if len(output) > MAX_TOOL_OUTPUT:
            output = f"{status}\nstderr:\n{stderr}\nstdout:\n{stdout}"
        if self.show_command_output:
            print(safe_terminal_text(f"[command: {command}]\n{retain_tool_output(output)}\n[end command output]"))
        return output

    def tool_write_file(self, args):
        path = self.path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(self.root)} ({len(content)} chars)"

    def tool_patch_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
        return f"patched {path.relative_to(self.root)}"

    ###################################################
    #### 6) Delegation And Bounded Subagents ##########
    ###################################################
    def tool_delegate(self, args):
        if self.depth >= self.max_depth:
            raise ValueError("delegate depth exceeded")
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
            skills=self.skills,
            skill_max_chars=self.skill_cap,
        )
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)


def build_welcome(agent, model, host):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center("MINI CODING AGENT"),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".mini-coding-agent" / "sessions")
    model = OllamaModelClient(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
    )
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return MiniAgent.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            model_name=args.model,
            skill_max_chars=args.skill_max_chars,
            show_command_output=args.show_command_output,
        )
    return MiniAgent(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        model_name=args.model,
        skill_max_chars=args.skill_max_chars,
        show_command_output=args.show_command_output,
    )


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model name.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--show-command-output", action="store_true", help="Display executed command output; may reveal sensitive data.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    parser.add_argument(
        "--skill-max-chars",
        type=positive_int,
        default=None,
        help="Max characters of a skill body given to the model (default depends on model size: 4000/12000/24000).",
    )
    return parser


def is_skill_command(text):
    return text == "/skill" or text.startswith("/skill ")


def run_skill_command(agent, text):
    kind, value = agent.parse_skill_command(text)
    if kind == "list":
        print("\n".join(f"{s['name']}: {s['description']}" for s in agent.skills) or "no skills found")
    elif kind == "usage":
        print("usage: /skill <name> <prompt>")
    elif kind == "unknown":
        available = ", ".join(s["name"] for s in agent.skills) or "none"
        print(f"unknown skill '{value}'. available: {available}")
    else:
        skill, prompt = value
        print()
        print(agent.ask(prompt, forced_skill=skill))


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)

    print(build_welcome(agent, model=args.model, host=args.host))

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                if is_skill_command(prompt):
                    run_skill_command(agent, prompt)
                else:
                    print(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        try:
            user_input = input("\nmini-coding-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            if is_skill_command(user_input):
                run_skill_command(agent, user_input)
            else:
                print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
