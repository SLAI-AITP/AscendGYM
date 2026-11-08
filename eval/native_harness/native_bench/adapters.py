"""Launch a native agent once. Never replace its model/tool loop."""
import shutil
import subprocess
import sys


KINDS = ("claude", "codex", "opencode", "custom")


def probe(kind, executable=None):
    executable = executable or kind
    path = shutil.which(executable)
    if path is None:
        return {"kind": kind, "available": False, "executable": executable}
    p = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15)
    return {"kind": kind, "available": p.returncode == 0, "executable": path,
            "version": (p.stdout or p.stderr).strip()}


def command(config, prompt, workspace, run_root):
    kind = config["kind"]
    exe = config.get("executable", kind)
    extra = config.get("extra_args", [])
    # No hidden run-duration, token, dollar or agent-turn termination flags.
    blocked = ("--max-budget-usd", "--max-turns", "--system-prompt", "--system-prompt-file",
               "--output-format", "--format", "--input-format", "--resume", "--continue",
               "--session-id", "--fork-session")
    if any(arg.split("=", 1)[0] in blocked for arg in extra):
        raise ValueError("budget/system replacement flags are not supported in native mode")
    if kind == "claude":
        argv = [exe, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages"]
        # Permissions remain explicit configuration, not an automatic bypass.
        if config.get("allowed_tools"):
            argv += ["--allowedTools", ",".join(config["allowed_tools"])]
        if config.get("permission_mode"):
            argv += ["--permission-mode", config["permission_mode"]]
    elif kind == "codex":
        argv = [exe, "exec", "--json", "--skip-git-repo-check", "--sandbox",
                config.get("sandbox", "workspace-write")]
    elif kind == "opencode":
        argv = [exe, "run", "--format", "json"]
    elif kind == "custom":
        replacements = {"{python}": sys.executable, "{workspace}": str(workspace),
                        "{run_root}": str(run_root), "{prompt_file}": str(run_root / "prompt.txt")}
        argv = []
        for arg in config["command"]:
            for old, new in replacements.items():
                arg = arg.replace(old, new)
            argv.append(arg)
        return argv, prompt
    else:
        raise ValueError("unknown native adapter: " + kind)
    if config.get("model"):
        argv += ["--model", config["model"]]
    argv += extra
    if kind == "codex":
        argv.append("-")
    return argv, prompt
