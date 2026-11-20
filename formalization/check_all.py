"""Rebuild each proof and check all local declaration dependencies."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
REV = "905b95818eb32af7874a58b427f50c1711a5e96c"
VERSION = "4.32.2"
MODULES = ["Basic", "Radial", "MixedNorm", "Reflection", "Retention",
           "RetentionLipschitz", "Kernel", "Readout", "Grid", "Main",
           "ReadoutIdentity", "Extensions", "Precision", "Saturation",
           "Envelope", "Sharpness", "AllGroupsSharpness"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    mathlib = Path(os.environ.get("NSI_MATHLIB", ROOT / ".lake/packages/mathlib"))
    lean = os.environ.get("NSI_LEAN") or shutil.which("lean")
    if not lean or not (mathlib / ".lake/build/lib/lean/Mathlib.olean").is_file():
        raise SystemExit("Install the pinned Lean version and run lake exe cache get first.")
    version = subprocess.check_output([lean, "--version"], cwd=ROOT, text=True).strip()
    if f"version {VERSION}," not in version:
        raise SystemExit(f"Expected Lean {VERSION}; received {version}")
    commit = subprocess.check_output(
        ["git", "-c", "safe.directory=" + mathlib.resolve().as_posix(),
         "-C", str(mathlib), "rev-parse", "HEAD"], text=True).strip()
    if commit != REV:
        raise SystemExit(f"Expected mathlib {REV}; received {commit}")
    files = [Path("NSI") / (name + ".lean") for name in MODULES]
    files += [Path("NSI.lean"), Path("Audit.lean")]
    actual = set()
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name != ".lake"]
        actual.update((Path(directory) / name).relative_to(ROOT)
                      for name in names if name.endswith(".lean"))
    if actual != set(files):
        raise SystemExit("The proof file list has changed. Update and audit the build list.")
    # Reject proof escape hatches even if they occur in a source comment.
    forbidden = r"\b(sorry|admit|axiom|unsafe|implemented_by|native_decide|sorryAx)\b"
    for relative in files:
        if relative.name != "Audit.lean" and re.search(forbidden, (ROOT / relative).read_text(encoding="utf-8-sig")):
            raise SystemExit(f"Forbidden proof token in {relative}")
    (ROOT / ".lake").mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="proof-check-", dir=ROOT / ".lake"))
    lib = build / "lib"
    (lib / "NSI").mkdir(parents=True)
    out = ROOT / "verification"
    out.mkdir(exist_ok=True)
    paths = [lib, mathlib / ".lake/build/lib/lean"]
    for packages in (ROOT / ".lake/packages", mathlib / ".lake/packages"):
        if packages.is_dir():
            paths += sorted(p / ".lake/build/lib/lean" for p in packages.iterdir()
                            if (p / ".lake/build/lib/lean").is_dir())
    env = os.environ.copy()
    env["LEAN_PATH"] = os.pathsep.join(map(str, paths))
    report = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "lean_version": version, "mathlib_commit": commit,
              "fresh_local_build": True, "warnings_as_errors": True,
              "allowed_axioms": ["propext", "Classical.choice", "Quot.sound"],
              "modules": [], "declarations": [], "success": False}
    for relative in files:
        started = time.monotonic()
        result = subprocess.run([lean, "-DwarningAsError=true", "-R", ".", "-o",
                                 str(lib / relative.with_suffix(".olean")), str(relative)],
                                cwd=ROOT, env=env, text=True, encoding="utf-8",
                                errors="replace", capture_output=True)
        log = result.stdout + result.stderr
        (out / (relative.stem + ".log")).write_text(log, encoding="utf-8")
        report["modules"].append({"source": relative.as_posix(), "sha256": sha(ROOT / relative),
                                  "exit_code": result.returncode,
                                  "seconds": round(time.monotonic() - started, 2)})
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{relative.as_posix()}: exit={result.returncode}", flush=True)
        if result.returncode or "sorryAx" in log or "warning:" in log:
            print(log, flush=True)
            raise SystemExit(1)
        if relative.name == "Audit.lean":
            for line in log.splitlines():
                if line.startswith("NSI_AUDIT "):
                    report["declarations"].append(json.loads(line[len("NSI_AUDIT "):]))
    if not report["declarations"]:
        raise SystemExit("Missing declaration audit results.")
    report["declarations"].sort(key=lambda row: row["name"])
    report["success"] = True
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(files)} Lean files; {len(report['declarations'])} declarations audited.", flush=True)


if __name__ == "__main__":
    main()
