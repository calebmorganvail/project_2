import json
import subprocess
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

from mini_coding_agent import (
    FakeModelClient,
    MiniAgent,
    OllamaModelClient,
    SessionStore,
    WorkspaceContext,
    build_welcome,
    build_arg_parser,
    discover_skills,
    main,
    skill_cap_for_model,
    skill_similarity,
    skill_tokens,
    truncate_skill,
)


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Read the file successfully.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "<final>Recovered after retry.</final>",
        ],
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Recovered after malformed tool output.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("valid <tool> call" in item for item in notices)


def test_agent_accepts_xml_write_file_tool(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "",
            "<final>Recovered after several retries.</final>",
        ],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    assert agent.ask("Start a session") == "First pass."

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"inspect README","max_steps":2}}</tool>',
            "<final>Child result.</final>",
            "<final>Parent incorporated the child result.</final>",
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_patch_file_replaces_exact_match(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "sample.txt",
            "old_text": "world",
            "new_text": "agent",
        },
    )

    assert result == "patched sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: 'path'")
    assert 'example: <tool name="write_file"' in result
    mock_input.assert_not_called()


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".mini-coding-agent").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".mini-coding-agent" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_path_rejects_parent_escape(tmp_path):
    agent = build_agent(tmp_path, [])

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("../outside.txt")


def test_path_rejects_symlink_escape(tmp_path):
    agent = build_agent(tmp_path, [])
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("outside-link/secret.txt")


def test_path_accepts_case_variant_on_case_insensitive_filesystems(tmp_path):
    project_root = tmp_path / "Proj"
    project_root.mkdir()
    agent = build_agent(project_root, [])
    variant = project_root.parent / project_root.name.lower() / "README.md"

    if not variant.exists():
        pytest.skip("case-sensitive filesystem")

    resolved = agent.path(str(variant))

    assert resolved.samefile(project_root / "README.md")


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    result = agent.run_tool("list_files", {})

    assert result == "error: repeated identical tool call for list_files; choose a different tool or return a final answer"


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "mini" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b", host="http://127.0.0.1:11434")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    assert len({len(line) for line in lines}) == 1
    assert "..." in welcome
    assert "O   O" in welcome
    assert "MINI-CODING-AGENT" not in welcome
    assert "MINI CODING AGENT" in welcome
    assert "// READY" not in welcome
    assert "SLASH" not in welcome
    assert "READY      " not in welcome
    assert "commands: Commands:" not in welcome


def test_prompt_top_level_sections_stay_flush_left_with_multiline_content(tmp_path):
    workspace = WorkspaceContext(
        cwd=str(tmp_path),
        repo_root=str(tmp_path),
        branch="fix/prompt-indentation",
        default_branch="main",
        status=" M mini_coding_agent.py\n?? tests/test_prompt.py",
        recent_commits=["abc123 first commit", "def456 second commit"],
        project_docs={"README.md": "line1\nline2"},
    )
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    agent = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    agent.session["memory"] = {
        "task": "verify prompt formatting",
        "files": ["mini_coding_agent.py"],
        "notes": ["saw inconsistent indentation", "need regression coverage"],
    }
    agent.record({"role": "user", "content": "inspect prompt()", "created_at": "1"})
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "mini_coding_agent.py"},
            "content": "    def prompt(self, user_message):\n        ...",
            "created_at": "2",
        }
    )

    prompt = agent.prompt("is this issue legit?")
    lines = prompt.splitlines()

    for label in ["Rules:", "Tools:", "Valid response examples:", "Workspace:", "Memory:", "Transcript:", "Current user request:"]:
        assert label in lines
        assert f"            {label}" not in prompt


def _make_filler(i):
    return {"role": "tool", "name": "list_files", "args": {}, "content": "", "created_at": str(i)}


def test_history_text_deduplicates_reads_but_not_after_write(tmp_path):
    """read_file deduplication must not skip a read that follows a write.

    Realistic prior-turn history (non-recent window):
        user: "update config"
        assistant: <tool>read_file config</tool>
        tool:   config v1 (content: setting=true)
        assistant: <tool>write_file config</tool>
        tool:   wrote
        assistant: <tool>read_file config</tool>
        tool:   config v2 (content: setting=false)   <- MUST NOT be skipped

    Without fix: seen_reads={"config"} after first read; write does NOT clear it;
                 second read is wrongly skipped (LLM sees stale content).
    With fix: write clears seen_reads, second read is correctly shown.
    """
    agent = build_agent(tmp_path, [])

    # Simulate a prior turn with read->write->read on the same file
    # history_length=13, recent_start=7 (indices 0-6 non-recent, 7-12 recent)
    agent.record({"role": "user", "content": "update config", "created_at": "0"})        # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=true\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"write_file","args":{"path":"config.txt","content":"setting=false\n"}}</tool>', "created_at": "3"})
    agent.record({"role": "tool", "name": "write_file", "args": {"path": "config.txt", "content": "setting=false\n"}, "content": "wrote config.txt", "created_at": "4"})  # index 4, non-recent
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "5"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=false\n", "created_at": "6"})  # index 6, non-recent, ADDED (write cleared dedup)
    # recent entries
    for i in range(7, 13):
        agent.record(_make_filler(i))

    history = agent.history_text()

    # Both read contents appear exactly once (check full line to avoid JSON false positives)
    assert "# config.txt\n   1: setting=true\n" in history
    assert "# config.txt\n   1: setting=false\n" in history
    # Also verify duplicate read (setting=true, same path) does NOT appear twice
    assert history.count("setting=true") == 1


def test_history_text_deduplicates_unchanged_repeated_reads(tmp_path):
    """read_file deduplication should still skip repeated reads with no write in between."""
    agent = build_agent(tmp_path, [])

    # Realistic: two identical reads with no write between them
    # history_length=10, recent_start=4 (indices 0-3 non-recent, 4-9 recent)
    agent.record({"role": "user", "content": "check logs", "created_at": "0"})  # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "log.txt"}, "content": "# log.txt\n   1: stable\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "3"})  # index 3, non-recent, SKIPPED (dup)
    for i in range(4, 10):
        agent.record(_make_filler(i))  # indices 4-9, recent

    history = agent.history_text()

    # Only first read should appear; duplicates must be skipped
    assert history.count("stable") == 1


def test_history_deduplication_keeps_newest_non_recent_read(tmp_path):
    agent = build_agent(tmp_path, [])
    for text in ("old content", "new content"):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"},
                      "content": text, "created_at": "1"})
    for i in range(6):
        agent.record(_make_filler(i))

    history = agent.history_text()
    assert "new content" in history
    assert "old content" not in history


def test_ollama_client_posts_expected_payload():
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OllamaModelClient(
        model="qwen3.5:4b",
        host="http://127.0.0.1:11434",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 30
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["prompt"] == "hello"
    assert captured["body"]["stream"] is False
    assert captured["body"]["raw"] is False
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["num_predict"] == 42


def test_ollama_client_timeout_is_reported_as_model_error():
    client = OllamaModelClient(model="local", host="http://localhost:11434", temperature=0.2, top_p=0.9, timeout=1)
    with patch("mini_coding_agent.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(RuntimeError, match="timed out"):
            client.complete("hello", 42)


############################
#### Skills: discovery #####
############################
def write_skill(root, folder, name=None, description="does a thing", body="Body text.\n", raw=None):
    path = root / "skills" / folder / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text(f"---\nname: {name or folder}\ndescription: {description}\n---\n{body}", encoding="utf-8")
    return path


def test_discover_skill_reads_name_description_and_relative_path(tmp_path):  # REQ-1, REQ-5
    write_skill(tmp_path, "alpha", description="Alpha skill")

    skills, warnings = discover_skills(tmp_path)

    assert warnings == []
    assert [(s["name"], s["description"], s["rel"]) for s in skills] == [
        ("alpha", "Alpha skill", "skills/alpha/SKILL.md")
    ]
    assert skills[0]["body"] == "Body text.\n"


def test_discover_ignores_loose_and_nested_files(tmp_path):  # REQ-2
    write_skill(tmp_path, "a")
    (tmp_path / "skills" / "loose.md").write_text("---\nname: loose\ndescription: x\n---\n", encoding="utf-8")
    write_skill(tmp_path, "a/nested", name="nested")
    (tmp_path / "skills" / "empty-folder").mkdir()

    skills, warnings = discover_skills(tmp_path)

    assert [s["name"] for s in skills] == ["a"]
    assert warnings == []


def test_no_skills_dir_or_empty_dir_is_silent(tmp_path, capsys):  # REQ-3
    assert discover_skills(tmp_path) == ([], [])
    (tmp_path / "skills").mkdir()
    assert discover_skills(tmp_path) == ([], [])
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("---\ndescription: d\n---\nbody", "missing name"),
        ("---\nname: bad\n---\nbody", "missing description"),
        ("---\nname: bad\ndescription:\n---\nbody", "missing description"),
        ("---\nname:\ndescription: d\n---\nbody", "missing name"),
        ("---\nname: bad\ndescription: |\n---\nbody", "block scalar"),
        ("---\nname: bad\ndescription: >\n---\nbody", "block scalar"),
        ("# no frontmatter\n", "missing frontmatter"),
        ("---\nname: bad\ndescription: d\n", "unterminated frontmatter"),
        ("---\nname: Bad_Name\ndescription: d\n---\nbody", "invalid name"),
    ],
)
def test_invalid_skills_are_skipped_with_warning(tmp_path, raw, reason):  # REQ-6, REQ-6a
    write_skill(tmp_path, "bad", raw=raw)
    write_skill(tmp_path, "good")

    skills, warnings = discover_skills(tmp_path)

    assert [s["name"] for s in skills] == ["good"]
    assert len(warnings) == 1
    assert warnings[0].startswith("warning: skipped skills/bad/SKILL.md: ")
    assert reason in warnings[0]


def test_quotes_are_kept_literally(tmp_path):  # REQ-5a
    write_skill(tmp_path, "q", raw='---\nname: q\ndescription: "quoted"\n---\nbody')

    skills, _ = discover_skills(tmp_path)

    assert skills[0]["description"] == '"quoted"'


def test_noise_lines_and_unknown_keys_are_ignored(tmp_path):
    write_skill(tmp_path, "n", raw="---\nname: n\nno colon here\n  indented: x\nlicense: MIT\ndescription: d\n---\nbody")

    skills, warnings = discover_skills(tmp_path)

    assert warnings == []
    assert skills[0]["description"] == "d"


def test_duplicate_names_keep_first_folder(tmp_path):  # REQ-7
    write_skill(tmp_path, "a-first", name="same")
    write_skill(tmp_path, "b-second", name="same")

    skills, warnings = discover_skills(tmp_path)

    assert [s["rel"] for s in skills] == ["skills/a-first/SKILL.md"]
    assert warnings == ["warning: skipped skills/b-second/SKILL.md: duplicate skill name 'same'"]


def test_invalid_utf8_skill_is_skipped(tmp_path):  # REQ-8
    path = write_skill(tmp_path, "bin")
    path.write_bytes(b"---\nname: bin\ndescription: \xff\xfe\n---\n")
    write_skill(tmp_path, "ok")

    skills, warnings = discover_skills(tmp_path)

    assert [s["name"] for s in skills] == ["ok"]
    assert len(warnings) == 1 and "skills/bin/SKILL.md: unreadable" in warnings[0]


def test_repo_train_and_report_skill_parses():
    skills, warnings = discover_skills(Path(__file__).resolve().parent.parent)

    assert warnings == []
    assert "train-and-report" in [s["name"] for s in skills]


############################
#### Skills: size cap ######
############################
@pytest.mark.parametrize(
    "model, cap",
    [
        ("qwen3.5:4b", 4000),
        ("llama3.1:8b", 12000),
        ("x:13b", 12000),
        ("x:70b", 24000),
        ("mistral", 24000),
        (None, 24000),
    ],
)
def test_skill_cap_tiers(model, cap):  # REQ-31
    assert skill_cap_for_model(model) == cap


def test_skill_cap_override_wins():  # REQ-31
    assert skill_cap_for_model("qwen3.5:4b", 99) == 99


def test_truncate_skill_adds_notice_only_over_cap():  # REQ-16
    assert truncate_skill("x" * 10, 10) == "x" * 10
    assert truncate_skill("x" * 11, 10) == "x" * 10 + "\n[skill truncated at 10 chars]"


############################
#### Skills: prompt ########
############################
def test_prefix_lists_skills_without_bodies(tmp_path):  # REQ-9, 10, 11
    write_skill(tmp_path, "alpha", description="Alpha skill", body="SECRET-BODY\n")
    agent = build_agent(tmp_path, [])

    assert "Available skills:\n- alpha: Alpha skill (skills/alpha/SKILL.md)" in agent.prefix
    assert "read its SKILL.md with read_file first" in agent.prefix
    assert "SECRET-BODY" not in agent.prefix
    assert "SECRET-BODY" not in agent.prompt("hi")


def test_zero_skill_output_is_unchanged(tmp_path):  # REQ-12, 29
    agent = build_agent(tmp_path, [])

    assert "Available skills" not in agent.prefix
    assert agent.memory_text() == "Memory:\n- task: -\n- files: -\n- notes:\n- none"
    assert agent.prompt("hi") == "\n\n".join(
        [agent.prefix, agent.memory_text(), "Transcript:\n- empty", "Current user request:\nhi"]
    )


def test_active_skills_section_is_flush_left_and_prefix_stable(tmp_path):  # REQ-35
    write_skill(tmp_path, "alpha")
    agent = build_agent(tmp_path, [])
    prefix_before = agent.prefix

    agent.active_skills["alpha"] = "BODY"
    prompt = agent.prompt("hi")

    assert "\n\nActive skills:\n### alpha\nBODY\n\nTranscript:" in prompt
    assert agent.prefix == prefix_before


def test_prompt_variants_for_inactive_and_active_skills(tmp_path):
    write_skill(tmp_path, "alpha")
    agent = build_agent(tmp_path, [])

    inactive = agent.prompt("hi")
    assert "Skill note:" not in inactive
    assert "read its SKILL.md with read_file first" in inactive

    agent.active_skills["alpha"] = "BODY"
    active = agent.prompt("hi")
    note = "Skill note: the skills below are already loaded. Follow them now; do not read their SKILL.md again."
    assert f"\n\n{note}\n\nActive skills:\n### alpha\nBODY" in active
    assert active.index(note) < active.index("Active skills:") < active.index("Transcript:")


def test_forced_skill_prompt_has_note_every_step(tmp_path):
    write_skill(tmp_path, "a", body="FORCED-BODY\n")
    client, _ = run_main(
        tmp_path,
        ['<tool>{"name":"list_files","args":{"path":"."}}</tool>', "<final>done</final>"],
        inputs=["/skill a do it"],
    )

    assert len(client.prompts) == 2
    assert all("Skill note:" in p for p in client.prompts)


def test_skill_note_absent_after_request_ends(tmp_path):
    write_skill(tmp_path, "a")
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    agent.ask("x", forced_skill=agent.skills[0])

    assert "Skill note:" not in agent.prompt("next")


def test_memory_text_reports_skill_usage(tmp_path):  # REQ-23
    write_skill(tmp_path, "alpha")
    agent = build_agent(tmp_path, [])
    agent.session["skills_used"].append({"name": "alpha", "request": "r", "at": "t"})

    assert "- skills_used: alpha\n- skills_active: -" in agent.memory_text()
    agent.active_skills["alpha"] = "BODY"
    assert "- skills_active: alpha" in agent.memory_text()


############################
#### Skills: model use #####
############################
def read_call(path, **extra):
    return json.dumps({"name": "read_file", "args": {"path": path, **extra}}).join(["<tool>", "</tool>"])


def test_model_read_of_skill_activates_it(tmp_path):  # REQ-13, 27
    write_skill(tmp_path, "alpha", description="Alpha skill", body="STEP-ONE\n")
    agent = build_agent(tmp_path, [read_call("skills/alpha/SKILL.md"), "<final>done</final>"])

    agent.ask("do alpha work")

    second_prompt = agent.model_client.prompts[1]
    assert "\n\nActive skills:\n### alpha\nSTEP-ONE" in second_prompt
    assert len(agent.session["skills_used"]) == 1
    assert agent.session["skills_used"][0]["name"] == "alpha"
    assert agent.session["skills_used"][0]["request"] == "do alpha work"
    tool_item = [i for i in agent.session["history"] if i["role"] == "tool"][0]
    assert tool_item["content"] == "[skill loaded: alpha - Alpha skill]"
    assert "STEP-ONE" not in json.dumps(agent.session)
    assert "STEP-ONE" not in "\n".join(agent.session["memory"]["notes"])


def test_two_skills_active_together(tmp_path):  # REQ-14
    write_skill(tmp_path, "a", body="BODY-A\n")
    write_skill(tmp_path, "b", body="BODY-B\n")
    agent = build_agent(
        tmp_path, [read_call("skills/a/SKILL.md"), read_call("skills/b/SKILL.md"), "<final>ok</final>"]
    )

    agent.ask("both")

    last = agent.model_client.prompts[2]
    assert "### a\nBODY-A" in last and "### b\nBODY-B" in last


def test_reading_other_skill_files_does_not_activate(tmp_path):  # REQ-15
    write_skill(tmp_path, "a")
    (tmp_path / "skills" / "a" / "notes.md").write_text("extra notes\n", encoding="utf-8")
    agent = build_agent(tmp_path, [read_call("skills/a/notes.md"), "<final>ok</final>"])

    agent.ask("read notes")

    assert "extra notes" in agent.session["history"][1]["content"]
    assert "Active skills:" not in agent.model_client.prompts[1]
    assert agent.session["skills_used"] == []


def test_skill_read_ignores_range_has_no_line_numbers_and_path_forms(tmp_path):
    write_skill(tmp_path, "a", body="line1\nline2\nline3\n")
    agent = build_agent(tmp_path, [])

    for path in ("skills/a/SKILL.md", "./skills/a/SKILL.md", "skills/a/../a/SKILL.md"):
        out = agent.run_tool("read_file", {"path": path, "start": 2, "end": 2})
        assert out == "line1\nline2\nline3\n"
        assert agent.active_skills["a"] == out
        agent.active_skills = {}


@pytest.mark.parametrize("extra", [{"start": 0}, {"start": 0, "end": 0}, {"start": 5, "end": 2}])
def test_skill_read_accepts_bad_ranges_but_normal_files_do_not(tmp_path, extra):
    write_skill(tmp_path, "a", body="line1\n")
    (tmp_path / "skills" / "a" / "notes.md").write_text("n\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    assert agent.run_tool("read_file", {"path": "skills/a/SKILL.md", **extra}) == "line1\n"
    assert "invalid line range" in agent.run_tool("read_file", {"path": "skills/a/notes.md", **extra})


def test_large_skill_read_uses_skill_cap_not_tool_clip(tmp_path):  # REQ-16
    write_skill(tmp_path, "big", body="x" * 30000)
    agent = build_agent(tmp_path, [])

    out = agent.run_tool("read_file", {"path": "skills/big/SKILL.md"})

    assert out == "x" * 24000 + "\n[skill truncated at 24000 chars]"
    small = build_agent(tmp_path, [], model_name="qwen3.5:4b")
    assert len(small.run_tool("read_file", {"path": "skills/big/SKILL.md"})) == 4000 + len(
        "\n[skill truncated at 4000 chars]"
    )


############################
#### Skills: forced ########
############################
def make_main_agent_patch(tmp_path, outputs):
    client = FakeModelClient(outputs)
    return client, patch("mini_coding_agent.OllamaModelClient", lambda **kw: client)


def run_main(tmp_path, outputs, inputs=None, argv=()):
    client, patcher = make_main_agent_patch(tmp_path, outputs)
    feed = iter(inputs or [])

    def fake_input(_prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    with patcher, patch("builtins.input", fake_input):
        code = main(["--cwd", str(tmp_path), *argv])
    return client, code


def test_forced_skill_is_in_every_step_but_not_history(tmp_path, capsys):  # REQ-17, 27
    write_skill(tmp_path, "a", body="FORCED-BODY\n")
    client, _ = run_main(
        tmp_path,
        ['<tool>{"name":"list_files","args":{"path":"."}}</tool>', "<final>done</final>"],
        inputs=["/skill a do it"],
    )

    assert all("Active skills:\n### a\nFORCED-BODY" in p for p in client.prompts)
    assert len(client.prompts) == 2
    session_file = next((tmp_path / ".mini-coding-agent" / "sessions").glob("*.json"))
    saved = json.loads(session_file.read_text(encoding="utf-8"))
    assert "FORCED-BODY" not in json.dumps(saved)
    assert saved["memory"]["task"] == "do it"
    assert saved["history"][0]["content"] == "do it"
    assert saved["skills_used"][0]["name"] == "a"


def test_skill_command_without_prompt_prints_usage(tmp_path, capsys):  # REQ-18
    write_skill(tmp_path, "a")
    client, _ = run_main(tmp_path, [], inputs=["/skill a"])

    assert "usage: /skill <name> <prompt>" in capsys.readouterr().out
    assert client.prompts == []


def test_skill_command_unknown_name_lists_available(tmp_path, capsys):  # REQ-19
    write_skill(tmp_path, "a")
    write_skill(tmp_path, "b")
    client, _ = run_main(tmp_path, [], inputs=["/skill nope do it"])

    assert "unknown skill 'nope'. available: a, b" in capsys.readouterr().out
    assert client.prompts == []


def test_bare_skill_lists_and_help_mentions_skill(tmp_path, capsys):  # REQ-21
    write_skill(tmp_path, "a", description="Alpha skill")
    client, _ = run_main(tmp_path, [], inputs=["/skill", "/help"])

    out = capsys.readouterr().out
    assert "a: Alpha skill" in out
    assert "/skill <name> <prompt>" in out
    assert client.prompts == []


def test_bare_skill_with_no_skills(tmp_path, capsys):
    run_main(tmp_path, [], inputs=["/skill"])

    assert "no skills found" in capsys.readouterr().out


def test_one_shot_skill_prompt(tmp_path, capsys):  # REQ-21
    write_skill(tmp_path, "a", body="ONE-SHOT\n")
    client, code = run_main(tmp_path, ["<final>ok</final>"], argv=["/skill", "a", "do", "it"])

    assert code == 0
    assert "Active skills:\n### a\nONE-SHOT" in client.prompts[0]
    assert "ok" in capsys.readouterr().out


def test_skills_prefix_is_not_captured_by_skill_command(tmp_path):  # REQ-20
    write_skill(tmp_path, "a")
    client, _ = run_main(tmp_path, ["<final>ok</final>"], inputs=["/skills something"])

    assert "Current user request:\n/skills something" in client.prompts[0]
    assert "Active skills:" not in client.prompts[0]


############################
#### Skills: lifecycle #####
############################
class BoomClient:
    def complete(self, prompt, max_new_tokens):
        raise RuntimeError("boom")


@pytest.mark.parametrize("case", ["final", "step_limit", "malformed", "error"])
def test_skills_inactive_after_request_ends(tmp_path, case):  # REQ-32, 33, 34
    write_skill(tmp_path, "a", body="BODY-A\n")
    read = read_call("skills/a/SKILL.md")
    outputs = {
        "final": [read, "<final>ok</final>"],
        "step_limit": [read, '<tool>{"name":"list_files","args":{"path":"."}}</tool>'],
        "malformed": [read] + ["garbage <tool>"] * 20,
        "error": [read],
    }[case]
    agent = build_agent(tmp_path, outputs, max_steps=2)

    answer = agent.ask("go")
    if case == "error":
        assert "Final-answer turn failed" in answer

    assert agent.active_skills == {}
    agent.model_client = FakeModelClient(["<final>next</final>"])
    agent.ask("next request")
    assert "Active skills:" not in agent.model_client.prompts[0]
    assert "BODY-A" not in agent.model_client.prompts[0]


def test_model_client_error_during_forced_skill_clears_state(tmp_path):
    write_skill(tmp_path, "a")
    agent = build_agent(tmp_path, [])
    agent.model_client = BoomClient()

    assert "Final-answer turn failed" in agent.ask("x", forced_skill=agent.skills[0])

    assert agent.active_skills == {}


def test_memory_shows_active_during_request_and_used_after(tmp_path):  # REQ-23
    write_skill(tmp_path, "a")
    agent = build_agent(tmp_path, [read_call("skills/a/SKILL.md"), "<final>ok</final>"])

    agent.ask("go")

    assert "- skills_active: a" in agent.model_client.prompts[1]
    assert "- skills_used: a\n- skills_active: -" in agent.memory_text()


def test_exit_eof_and_interrupt_leave_nothing_active(tmp_path):
    write_skill(tmp_path, "a")
    for inputs in (["/exit"], ["/quit"], []):
        client, code = run_main(tmp_path, [], inputs=inputs)
        assert code == 0

    client, patcher = make_main_agent_patch(tmp_path, [])
    with patcher, patch("builtins.input", side_effect=KeyboardInterrupt):
        assert main(["--cwd", str(tmp_path)]) == 0


############################
#### Skills: session #######
############################
def test_skills_used_round_trips_but_active_does_not(tmp_path):  # REQ-22
    write_skill(tmp_path, "a")
    agent = build_agent(tmp_path, [read_call("skills/a/SKILL.md"), "<final>ok</final>"])
    agent.ask("go")

    resumed = MiniAgent.from_session(
        FakeModelClient([]), agent.workspace, agent.session_store, agent.session["id"]
    )

    assert resumed.session["skills_used"][0]["name"] == "a"
    assert resumed.active_skills == {}


@pytest.mark.parametrize("value", ["absent", None])
def test_old_sessions_without_skills_used_load(tmp_path, value):  # REQ-25
    write_skill(tmp_path, "a")
    agent = build_agent(tmp_path, [])
    data = json.loads(agent.session_path.read_text(encoding="utf-8"))
    data.pop("skills_used")
    if value is None:
        data["skills_used"] = None
    agent.session_path.write_text(json.dumps(data), encoding="utf-8")

    resumed = MiniAgent.from_session(
        FakeModelClient([read_call("skills/a/SKILL.md"), "<final>ok</final>"]),
        agent.workspace,
        agent.session_store,
        agent.session["id"],
    )

    assert resumed.session["skills_used"] == []
    assert "skills_used" not in resumed.memory_text()
    resumed.ask("go")
    resumed.reset()
    assert resumed.session["skills_used"] == []


def test_resume_with_deleted_skill_warns_once(tmp_path, capsys):  # REQ-26
    agent = build_agent(tmp_path, [])
    agent.session["skills_used"] = [
        {"name": "gone", "request": "a", "at": "t"},
        {"name": "gone", "request": "b", "at": "t"},
        "malformed",
    ]
    agent.session_store.save(agent.session)
    capsys.readouterr()

    MiniAgent.from_session(FakeModelClient([]), agent.workspace, agent.session_store, agent.session["id"])

    err = capsys.readouterr().err
    assert err.count("warning: session used skill 'gone' which is no longer on disk") == 1


def test_reset_clears_usage_and_rescans(tmp_path):  # REQ-4, 24
    write_skill(tmp_path, "a")
    agent = build_agent(tmp_path, [read_call("skills/a/SKILL.md"), "<final>ok</final>"])
    agent.ask("go")
    write_skill(tmp_path, "b", description="Bee skill")
    (tmp_path / "skills" / "a" / "SKILL.md").unlink()

    agent.reset()

    assert agent.session["skills_used"] == []
    assert agent.active_skills == {}
    assert "- b: Bee skill" in agent.prefix
    assert "- a:" not in agent.prefix


def test_startup_warnings_go_to_stderr(tmp_path, capsys):
    write_skill(tmp_path, "bad", raw="no frontmatter")

    build_agent(tmp_path, [])

    captured = capsys.readouterr()
    assert captured.err == "warning: skipped skills/bad/SKILL.md: missing frontmatter\n"


############################
#### Skills: delegation ####
############################
@pytest.mark.parametrize("forced", [True, False])
def test_delegate_blocked_until_active_skill_request_ends(tmp_path, forced):
    write_skill(tmp_path, "a", body="Follow this skill.\n")
    delegate = '<tool>{"name":"delegate","args":{"task":"investigate"}}</tool>'
    outputs = [delegate, "<final>skill request done</final>"]
    if not forced:
        outputs.insert(0, read_call("skills/a/SKILL.md"))
    outputs.extend([delegate, "<final>child result</final>", "<final>next request done</final>"])
    agent = build_agent(tmp_path, outputs)

    assert agent.ask("skill request", forced_skill=agent.skills[0] if forced else None) == "skill request done"
    events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert events[-1]["content"] == "error: delegate is unavailable while skills are active"
    assert "delegate tool is unavailable while skills are active" in agent.model_client.prompts[-1]
    assert len(list((tmp_path / ".mini-coding-agent" / "sessions").glob("*.json"))) == 1

    assert agent.ask("next request") == "next request done"
    assert agent.session["history"][-2]["content"] == "delegate_result:\nchild result"


def test_active_skill_keeps_write_and_patch_approval(tmp_path):
    write_skill(tmp_path, "a")
    agent = build_agent(tmp_path, [], approval_policy="ask")
    agent.activate_skill(agent.skills[0], agent.skills[0]["body"])

    with patch("builtins.input", side_effect=["n", "y", "y"]) as approve:
        assert agent.run_tool("write_file", {"path": "out.txt", "content": "first"}) == "error: approval denied for write_file"
        assert agent.run_tool("write_file", {"path": "out.txt", "content": "first"}).startswith("wrote ")
        assert agent.run_tool("patch_file", {"path": "out.txt", "old_text": "first", "new_text": "second"}).startswith("patched ")
    assert approve.call_count == 3
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "second"


def test_child_inherits_skills_without_touching_parent_usage(tmp_path, capsys):
    write_skill(tmp_path, "a", body="CHILD-BODY\n")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"look","max_steps":2}}</tool>',
            read_call("skills/a/SKILL.md"),
            "<final>child done</final>",
            "<final>parent done</final>",
        ],
    )
    capsys.readouterr()

    agent.ask("delegate it")

    assert "Available skills:\n- a:" in agent.model_client.prompts[1]
    assert agent.session["skills_used"] == []
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("size", [15999, 16000, 16001])
def test_tool_results_are_retained_with_explicit_limit(tmp_path, size):
    agent = build_agent(tmp_path, [])
    agent.tools["list_files"]["run"] = lambda args: "x" * size
    result = agent.run_tool("list_files", {"path": "."})
    assert result[:16000] == "x" * min(size, 16000)
    if size > 16000:
        assert "discarded" in result
    else:
        assert result == "x" * size


def test_retrieve_result_ranges_and_old_session(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "a" * 8000, "created_at": "1"})
    event = agent.session["history"][0]
    output_id = event["result_id"]
    result = agent.run_tool("read_tool_output", {"id": output_id, "start": 3000, "end": 5000})
    assert "a" * 2000 in result
    assert "error:" not in result
    assert "error:" in agent.run_tool("read_tool_output", {"id": output_id, "start": 1, "end": 9000})
    assert "error:" in agent.run_tool("read_tool_output", {"id": output_id, "start": -1, "end": 1})
    resumed = MiniAgent.from_session(FakeModelClient([]), agent.workspace, agent.session_store, agent.session["id"])
    assert "a" * 20 in resumed.run_tool("read_tool_output", {"id": output_id, "start": 0, "end": 20})
    resumed.reset()
    resumed.record({"role": "tool", "name": "list_files", "args": {}, "content": "new", "created_at": "2"})
    assert "error:" in resumed.run_tool("read_tool_output", {"id": output_id, "start": 0, "end": 1})


def test_history_preserves_recent_grid_and_bounded_latest_entries(tmp_path):
    agent = build_agent(tmp_path, [])
    for i in range(60):
        agent.record({"role": "user", "content": "old" + str(i) + "x" * 800, "created_at": "1"})
    content = "x" * 1302 + "\nGreedy policy:\n> > > v\n"
    agent.record({"role": "tool", "name": "run_shell", "args": {"command": "train"}, "content": content, "created_at": "2"})
    history = agent.history_text()
    assert "Greedy policy:\n> > > v" in history
    assert len(history) <= 12000
    assert "old0" not in history
    assert "omitted" in history


@pytest.mark.parametrize("size", [2999, 3000, 3001])
def test_recent_result_prompt_boundary_and_retrieval_hint(tmp_path, size):
    agent = build_agent(tmp_path, [])
    content = "x" * (size - 3) + "END"
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": content, "created_at": "1"})
    history = agent.history_text()
    if size <= 3000:
        assert content in history
    else:
        assert content not in history
        assert "truncated" in history and "read_tool_output" in history
        event = agent.session["history"][0]
        result = agent.run_tool("read_tool_output", {"id": event["result_id"], "start": 2990, "end": 3001})
        assert result.endswith(content[2990:3001])


def test_retention_metadata_and_discarded_range(tmp_path):
    agent = build_agent(
        tmp_path, ['<tool>{"name":"list_files","args":{"path":"."}}</tool>', "<final>done</final>"]
    )
    agent.tools["list_files"]["run"] = lambda args: "q" * 16001
    assert agent.ask("work") == "done"
    event = next(item for item in agent.session["history"] if item["role"] == "tool")
    assert event["retained_length"] == 16000 and event["original_length"] == 16001
    assert "discarded 1 characters" in event["content"]
    assert "discarded 1 characters" in agent.prompt("continue")
    assert "discarded 1 characters" in agent.run_tool(
        "read_tool_output", {"id": event["result_id"], "start": 0, "end": 10}
    )
    assert "unavailable or discarded" in agent.run_tool(
        "read_tool_output", {"id": event["result_id"], "start": 15999, "end": 16001}
    )


def test_large_tool_error_is_bounded_and_recorded(tmp_path):
    agent = build_agent(tmp_path, ['<tool>{"name":"list_files","args":{"path":"."}}</tool>', "<final>done</final>"])

    def fail(_args):
        raise ValueError("e" * 17000)

    agent.tools["list_files"]["run"] = fail
    assert agent.ask("work") == "done"
    event = next(item for item in agent.session["history"] if item["role"] == "tool")
    assert event["retained_length"] == 16000
    assert event["original_length"] > 16000
    assert "discarded" in event["content"]


def test_old_session_results_gain_stable_ids_and_are_session_scoped(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "old result", "created_at": "1"})
    agent.session["history"][0].pop("result_id")
    agent.session_store.save(agent.session)
    resumed = MiniAgent.from_session(FakeModelClient([]), agent.workspace, agent.session_store, agent.session["id"])
    result_id = resumed.session["history"][0]["result_id"]
    assert "old result" in resumed.run_tool("read_tool_output", {"id": result_id, "start": 0, "end": 10})
    again = MiniAgent.from_session(FakeModelClient([]), agent.workspace, agent.session_store, agent.session["id"])
    assert again.session["history"][0]["result_id"] == result_id
    other = build_agent(tmp_path, [])
    assert "not found" in other.run_tool("read_tool_output", {"id": result_id, "start": 0, "end": 1})


def test_retrieval_uses_unicode_character_offsets(tmp_path):
    agent = build_agent(tmp_path, [])
    content = "A\N{GRINNING FACE}\N{LATIN SMALL LETTER E WITH ACUTE}\nZ"
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": content, "created_at": "1"})
    result_id = agent.session["history"][0]["result_id"]
    assert agent.run_tool("read_tool_output", {"id": result_id, "start": 1, "end": 4}).endswith(content[1:4])


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 0), (3, 2), (0, 2801), ("0", 1), (False, 1)])
def test_retrieval_rejects_invalid_ranges(tmp_path, start, end):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "content", "created_at": "1"})
    result_id = agent.session["history"][0]["result_id"]
    assert agent.run_tool("read_tool_output", {"id": result_id, "start": start, "end": end}).startswith("error:")


def test_retrieval_chunk_fits_next_prompt_and_skill_body_is_not_saved(tmp_path):
    write_skill(tmp_path, "a", body="PRIVATE-SKILL-BODY\n")
    agent = build_agent(tmp_path, [read_call("skills/a/SKILL.md"), "<final>ok</final>"])
    assert agent.ask("work") == "ok"
    event = next(item for item in agent.session["history"] if item["role"] == "tool")
    result = agent.run_tool("read_tool_output", {"id": event["result_id"], "start": 0, "end": len(event["content"])})
    assert "skill loaded" in result and "PRIVATE-SKILL-BODY" not in result
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "q" * 8000, "created_at": "2"})
    result_id = agent.session["history"][-1]["result_id"]
    chunk = agent.run_tool("read_tool_output", {"id": result_id, "start": 2000, "end": 4800})
    assert len(chunk) < 3000
    agent.record({"role": "tool", "name": "read_tool_output", "args": {}, "content": chunk, "created_at": "3"})
    assert chunk in agent.prompt("continue")


@pytest.mark.parametrize("forced", [False, True])
def test_final_turn_after_tool_limit_cannot_execute_tools(tmp_path, forced):
    write_skill(tmp_path, "a", body="ACTIVE\n")
    outputs = [
        '<tool>{"name":"list_files","args":{"path":"missing"}}</tool>',
        '<tool>{"name":"write_file","args":{"path":"danger.txt","content":"bad"}}</tool>',
    ]
    agent = build_agent(tmp_path, outputs, max_steps=1)
    answer = agent.ask("work", forced_skill=agent.skills[0] if forced else None)
    assert "final" in answer.lower() and "tool" in answer.lower()
    assert not (tmp_path / "danger.txt").exists()
    assert len(agent.model_client.prompts) == 2
    assert ("Active skills:" in agent.model_client.prompts[1]) == forced
    assert agent.active_skills == {}


def test_final_turn_after_malformed_response_limit(tmp_path):
    agent = build_agent(tmp_path, ["<tool>bad</tool>"] * 5 + ["<final>partial</final>"], max_steps=1)
    assert agent.ask("work") == "partial"
    assert len(agent.model_client.prompts) == 6


@pytest.mark.parametrize("final_response", ["", "<tool>bad</tool>", "<final></final>"])
def test_final_turn_reports_invalid_response_without_retry(tmp_path, final_response):
    agent = build_agent(
        tmp_path, ['<tool>{"name":"list_files","args":{"path":"."}}</tool>', final_response], max_steps=1
    )
    assert "Final-answer turn failed" in agent.ask("work")
    assert len(agent.model_client.prompts) == 2
    assert sum(item["role"] == "tool" for item in agent.session["history"]) == 1


def test_final_turn_reports_model_unavailable(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.model_client = BoomClient()
    answer = agent.ask("work")
    assert "Final-answer turn failed" in answer and "boom" in answer


def test_final_turn_after_model_error(tmp_path):
    class RecoveringClient:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, max_new_tokens):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection failed")
            return "<final>Could not complete the task.</final>"

    agent = build_agent(tmp_path, [])
    agent.model_client = RecoveringClient()
    assert agent.ask("work") == "Could not complete the task."
    assert agent.model_client.calls == 2


def test_command_output_display_is_optional_and_respects_approval(tmp_path, capsys):
    agent = build_agent(tmp_path, [], approval_policy="ask", show_command_output=True)
    result = subprocess.CompletedProcess(args="command", returncode=3, stdout="hello\n", stderr="oops\n")
    with patch("builtins.input", side_effect=["n", "y"]), patch("mini_coding_agent.subprocess.run", return_value=result) as run:
        assert "approval denied" in agent.run_tool("run_shell", {"command": "command"})
        run.assert_not_called()
        assert capsys.readouterr().out == ""
        agent.run_tool("run_shell", {"command": "command"})
    output = capsys.readouterr().out
    assert "command" in output and "exit_code: 3" in output and "hello" in output and "oops" in output
    agent.show_command_output = False
    with patch("builtins.input", return_value="y"), patch("mini_coding_agent.subprocess.run", return_value=result):
        agent.run_tool("run_shell", {"command": "command"})
    assert capsys.readouterr().out == ""


def test_command_timeout_displays_available_output(tmp_path, capsys):
    agent = build_agent(tmp_path, [], show_command_output=True)
    timeout = subprocess.TimeoutExpired(cmd="command", timeout=1, output=b"partial")
    with patch("mini_coding_agent.subprocess.run", side_effect=timeout):
        result = agent.run_tool("run_shell", {"command": "command", "timeout": 1})
    assert "timed out" in result and "partial" in result
    assert "timed out" in capsys.readouterr().out


def test_command_start_error_displays_failure_without_exit_code(tmp_path, capsys):
    agent = build_agent(tmp_path, [], show_command_output=True)
    with patch("mini_coding_agent.subprocess.run", side_effect=OSError("cannot start")):
        result = agent.run_tool("run_shell", {"command": "command"})
    assert "cannot start" in result and "exit_code:" not in result
    assert "cannot start" in capsys.readouterr().out


def test_command_output_display_sanitizes_controls_and_marks_truncation(tmp_path, capsys):
    agent = build_agent(tmp_path, [], show_command_output=True)
    result = subprocess.CompletedProcess(args="command", returncode=0, stdout="\x1b[2J" + "x" * 17000, stderr="bad")
    with patch("mini_coding_agent.subprocess.run", return_value=result):
        stored = agent.run_tool("run_shell", {"command": "command"})
    printed = capsys.readouterr().out
    assert "\x1b" not in printed and "\\u001b" in printed
    assert "stderr:\nbad" in printed and "discarded" in printed and "discarded" in stored


def test_one_shot_cli_displays_command_output_when_enabled(tmp_path, capsys):
    command = f'"{sys.executable}" -c "print(123456)"'
    call = "<tool>" + json.dumps({"name": "run_shell", "args": {"command": command}}) + "</tool>"
    _, code = run_main(
        tmp_path, [call, "<final>finished</final>"],
        inputs=["y"], argv=["--show-command-output", "run", "command"],
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "stdout:\n123456" in output and "exit_code: 0" in output
    assert output.index("stdout:\n123456") < output.index("finished")


############################
#### Skills: CLI ###########
############################
def test_skill_max_chars_flag():  # REQ-30, 31
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--skill-max-chars", "0"])
    assert build_arg_parser().parse_args([]).skill_max_chars is None
    assert build_arg_parser().parse_args(["--skill-max-chars", "7"]).skill_max_chars == 7


#####################################################
#### Skills: similarity check and skill log #########
#####################################################
def read_skill_log(root):
    path = root / ".mini-coding-agent" / "skills.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_skill_tokens_and_similarity_bounds():  # REQ-046
    assert skill_tokens("Solve the Lab4 tests") == ["solve", "lab", "4", "test"]
    assert skill_similarity("vacuum robot", "vacuum robot") == pytest.approx(1.0)
    assert skill_similarity("vacuum robot", "binary search") == 0.0
    assert skill_similarity("", "anything") == 0.0


def test_repo_skill_matches_lab4_prompts():  # REQ-046, REQ-047
    skills, _ = discover_skills(Path(__file__).resolve().parent.parent)
    skill = next(s for s in skills if s["name"] == "train-and-report")
    text = f"train and report {skill['description']}"
    assert skill_similarity("train the vacuum robot and report the results", text) >= 0.25
    assert skill_similarity("solve lab 4", text) >= 0.25
    assert skill_similarity("write a binary search function", text) < 0.25


def test_similar_prompt_activates_skill_and_logs_it(tmp_path):  # REQ-047, REQ-048
    write_skill(tmp_path, "vacuum", description="Train the vacuum robot and report results", body="STEP-ONE\n")
    write_skill(tmp_path, "review", description="Review a pull request diff")
    agent = build_agent(tmp_path, ["<final>report</final>"])

    assert agent.ask("train the vacuum robot") == "report"

    assert "\n\nActive skills:\n### vacuum\nSTEP-ONE" in agent.model_client.prompts[0]
    assert agent.active_skills == {}
    log = read_skill_log(tmp_path)
    assert [entry["event"] for entry in log] == ["match", "executed", "finished"]
    match = log[0]
    assert match["prompt"] == "train the vacuum robot"
    assert match["scores"]["vacuum"] >= match["threshold"] > match["scores"]["review"]
    assert match["selected"] == "vacuum" and match["trigger"] == "similarity"
    assert log[1]["skill"] == "vacuum" and log[1]["trigger"] == "similarity"
    assert log[2]["answer"] == "report"
    assert agent.session["skills_used"][0]["trigger"] == "similarity"
    assert "STEP-ONE" not in json.dumps(agent.session)


def test_unrelated_prompt_logs_match_without_selection(tmp_path):  # REQ-047, REQ-048
    write_skill(tmp_path, "vacuum", description="Train the vacuum robot", body="STEP-ONE\n")
    agent = build_agent(tmp_path, ["<final>ok</final>"])

    agent.ask("write a binary search function")

    assert "Active skills" not in agent.model_client.prompts[0]
    log = read_skill_log(tmp_path)
    assert [entry["event"] for entry in log] == ["match"]
    assert log[0]["selected"] is None and log[0]["trigger"] is None


def test_model_loaded_skill_is_logged_with_model_trigger(tmp_path):  # REQ-048
    write_skill(tmp_path, "vacuum", description="Train the vacuum robot", body="STEP-ONE\n")
    agent = build_agent(tmp_path, [read_call("skills/vacuum/SKILL.md"), "<final>ok</final>"])

    agent.ask("write a binary search function")

    executed = [entry for entry in read_skill_log(tmp_path) if entry["event"] == "executed"]
    assert len(executed) == 1
    assert executed[0]["skill"] == "vacuum" and executed[0]["trigger"] == "model"


def test_rereading_auto_loaded_skill_logs_execution_once(tmp_path):  # REQ-048
    write_skill(tmp_path, "vacuum", description="Train the vacuum robot", body="STEP-ONE\n")
    agent = build_agent(tmp_path, [read_call("skills/vacuum/SKILL.md"), "<final>ok</final>"])

    agent.ask("train the vacuum robot")

    events = [entry["event"] for entry in read_skill_log(tmp_path)]
    assert events == ["match", "executed", "finished"]


def test_threshold_none_disables_automatic_selection(tmp_path):  # REQ-047
    write_skill(tmp_path, "vacuum", description="Train the vacuum robot")
    agent = build_agent(tmp_path, ["<final>ok</final>"], skill_threshold=None)

    agent.ask("train the vacuum robot")

    assert "Active skills" not in agent.model_client.prompts[0]
    assert read_skill_log(tmp_path)[0]["selected"] is None


def test_forced_skill_is_logged_as_explicit(tmp_path, capsys):  # REQ-048, REQ-049
    write_skill(tmp_path, "a", body="FORCED-BODY\n")
    run_main(tmp_path, ["<final>ok</final>"], inputs=["/skill a write a binary search"])

    log = read_skill_log(tmp_path)
    assert log[0]["event"] == "match" and log[0]["selected"] == "a" and log[0]["trigger"] == "explicit"
    assert log[1]["event"] == "executed" and log[1]["trigger"] == "explicit"
    assert "[skills] executing skill 'a' (trigger=explicit)" in capsys.readouterr().err


def test_no_skills_writes_no_skill_log(tmp_path):  # REQ-050
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    agent.ask("train the vacuum robot")
    assert not (tmp_path / ".mini-coding-agent" / "skills.jsonl").exists()


def test_skill_log_echo_is_off_by_default(tmp_path, capsys):  # REQ-048
    write_skill(tmp_path, "vacuum", description="Train the vacuum robot")
    build_agent(tmp_path, ["<final>ok</final>"]).ask("train the vacuum robot")
    assert "[skills]" not in capsys.readouterr().err


def test_ollama_num_ctx_is_sent_and_configurable():  # REQ-051
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"response": "ok"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OllamaModelClient(model="m", host="http://h", temperature=0.2, top_p=0.9, timeout=5, num_ctx=16384)
    with patch("urllib.request.urlopen", fake_urlopen):
        client.complete("hi", 10)
    assert captured["body"]["options"]["num_ctx"] == 16384
    assert build_arg_parser().parse_args([]).num_ctx == 8192
    assert build_arg_parser().parse_args([]).skill_threshold == 0.25
