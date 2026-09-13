"""One offline regression command: python -m scripts.run_checks.

Each suite runs in its own process with a temporary database and no network.
Full output is retained under output/checks; only failures are printed in full.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ("Agent 2 workflows", ["-m", "tests.test_agent2"]),
    (
        "Agent 3 workflows",
        [
            "-m",
            "pytest",
            "-q",
            "tests/test_agent3.py",
            "tests/test_agent3_react_tools.py",
        ],
    ),
    ("Database and interview PDFs", ["-m", "tests.test_agent2_database"]),
    ("Cosine matcher", ["-m", "tests.test_cosine_matcher"]),
    ("Parser fixtures", ["-m", "tests.test_parser_fixtures"]),
    ("Product and Streamlit", ["-m", "unittest", "tests.test_product_flow", "-v"]),
    ("Evaluation labels", ["-m", "evaluation.calibrate_cosine_threshold", "--check"]),
    ("Document extraction cache", ["-m", "tests.test_agent2_document_extractor"]),
]


def main():
    log_dir = ROOT / "output/checks"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    with tempfile.TemporaryDirectory() as temporary:
        env = {**os.environ, "AGENT2_DATABASE_PATH": str(Path(temporary) / "checks.sqlite3"),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1"}
        # Refuse accidental outbound connections while leaving local framework sockets available.
        guard = Path(temporary) / "sitecustomize.py"
        guard.write_text("import socket\n_original=socket.socket.connect\n"
            "def connect(self, address):\n"
            "    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', 'localhost', '::1'):\n"
            "        raise RuntimeError('Outbound networking disabled by offline test runner')\n"
            "    return _original(self, address)\n"
            "socket.socket.connect=connect\n", encoding="utf-8")
        env["PYTHONPATH"] = os.pathsep.join([temporary, str(ROOT), env.get("PYTHONPATH", "")])
        for index, (name, args) in enumerate(SUITES, 1):
            result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            log = result.stdout + result.stderr
            (log_dir / f"{index:02d}.log").write_text(log, encoding="utf-8")
            print(f"{'PASS' if result.returncode == 0 else 'FAIL'}: {name}", flush=True)
            if result.returncode:
                failed.append(name)
                print(log[-10000:])
    print(f"{len(SUITES)-len(failed)}/{len(SUITES)} suites passed. Logs: {log_dir}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
