"""Build a small, allowlisted source handoff. Never package credentials or run logs."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def build(output):
    root = Path(__file__).resolve().parents[1]
    target = Path(output).resolve()
    if target.exists() or target.with_suffix(target.suffix + ".sha256").exists():
        raise ValueError("package output must be new")
    files = {}
    for p in sorted((root / "native_bench").rglob("*")):
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".txt"):
            files["code/" + p.relative_to(root).as_posix()] = p.read_bytes()
    keep = ["core/__init__.py", "core/contract.py", "core/metrics.py", "core/profiling.py",
            "core/tests/test_live_bridge.py",
            "core/references/metrics_master.md", "core/examples/fixture_evaluator.py",
            "core/examples/sandbox_add_rms_norm.py", "core/examples/toy/spec.md",
            "core/examples/toy/source/op_kernel/kernel.py",
            "core/examples/toy/source/op_speed_test/test_kernel.py"]
    keep.extend(["tools/prepare_native_claude_real.py", "tools/prepare_live_add_rms_norm.py", "tools/audit_native_real.py"])
    for name in keep:
        files["code/" + name] = (root / name).read_bytes()
    files["code/native_bench/examples/CONFIGURATION.md"] = (root / "native_bench/examples/CONFIGURATION.md").read_bytes()
    docs = {
        "HANDOFF.md": "README.md",
        "README.md": "docs/01_usage_architecture/README.md",
        "VALIDATION.md": "docs/02_validation/VALIDATION.md",
        "TOKEN_ACCOUNTING_STATUS.md": "docs/03_metrics/TOKEN_ACCOUNTING_STATUS.md",
    }
    for source, destination in docs.items():
        path = root / "native_bench" / source
        if not path.exists():
            path = root.parent / destination  # Rebuild from an extracted handoff's code/.
        content = path.read_text(encoding="utf-8")
        if source == "README.md":
            content = content.replace("](VALIDATION.md)", "](../02_validation/VALIDATION.md)")
            content = content.replace("](TOKEN_ACCOUNTING_STATUS.md)", "](../03_metrics/TOKEN_ACCOUNTING_STATUS.md)")
            content = content.replace("见 `VALIDATION.md`", "见 [验证记录](../02_validation/VALIDATION.md)")
        elif source == "VALIDATION.md":
            content = content.replace("](TOKEN_ACCOUNTING_STATUS.md)", "](../03_metrics/TOKEN_ACCOUNTING_STATUS.md)")
        files[destination] = content.encode("utf-8")
    files["docs/03_metrics/metrics_master.md"] = files["code/core/references/metrics_master.md"]
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files["MANIFEST.sha256.json"] = (json.dumps(checksums, ensure_ascii=False, indent=2) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            z.writestr(name, data)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(digest + "  " + target.name + "\n")
    return {"path": str(target), "sha256": digest, "files": len(files), "bytes": target.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
