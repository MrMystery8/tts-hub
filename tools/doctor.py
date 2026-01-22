from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUB_ROOT))

from hub.model_registry import get_model_specs
from hub.paths import resolve_model_runtime_paths


def _run_handshake(python: Path, worker_script: Path, cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.Popen(
            [str(python), "-u", str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
        )
    except Exception as e:
        return False, f"spawn failed: {e}"

    assert proc.stdin is not None
    assert proc.stdout is not None

    try:
        hello = proc.stdout.readline().strip()
        if not hello:
            err = (proc.stderr.read() if proc.stderr else "")[:300]
            return False, f"no handshake; stderr={err!r}"
        try:
            hello_obj = json.loads(hello)
        except json.JSONDecodeError:
            return False, f"invalid handshake: {hello!r}"
        if not hello_obj.get("ok"):
            return False, f"handshake not ok: {hello_obj}"

        proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
        proc.stdin.flush()
        resp = proc.stdout.readline().strip()
        if not resp:
            return False, "no shutdown response"
        resp_obj = json.loads(resp)
        if not resp_obj.get("ok"):
            return False, f"shutdown not ok: {resp_obj}"
        return True, hello_obj.get("msg", "ok")
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def main() -> int:
    hub_root = HUB_ROOT
    ok = True

    ffmpeg = shutil.which("ffmpeg")
    print(f"[ffmpeg] {'OK' if ffmpeg else 'MISSING'} {ffmpeg or ''}".rstrip())
    if not ffmpeg:
        ok = False

    print("")
    for spec in get_model_specs():
        runtime = resolve_model_runtime_paths(hub_root, spec.id)
        worker_script = hub_root / "workers" / spec.worker_entry

        issues: list[str] = []
        if not runtime.repo_root.exists():
            issues.append(f"missing repo: {runtime.repo_root}")
        if not runtime.python.exists():
            issues.append(f"missing python: {runtime.python}")
        if not worker_script.exists():
            issues.append(f"missing worker: {worker_script}")

        if issues:
            ok = False
            print(f"[{spec.id}] FAIL")
            for i in issues:
                print(f"  - {i}")
            continue

        passed, msg = _run_handshake(runtime.python, worker_script, runtime.repo_root)
        if not passed:
            ok = False
            print(f"[{spec.id}] FAIL: {msg}")
        else:
            print(f"[{spec.id}] OK: {runtime.python.name} @ {runtime.repo_root.name} ({msg})")

    if not ok:
        print("\nOne or more checks failed.")
        print("Common fixes:")
        print("- Install ffmpeg (macOS): `brew install ffmpeg`")
        print("- Create each repo’s venv under `<repo>/.venv/`")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
