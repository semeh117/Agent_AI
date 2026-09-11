"""Verify a built wheel outside the checkout, using installed dependencies.

Run after building: python -m scripts.check_package output/wheels/<wheel>.whl
No provider calls, personal data, or delivery are used by this smoke test.
"""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve(strict=True)
    checkout = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="next-chapter-wheel-") as directory:
        temporary = Path(directory).resolve()
        target = temporary / "installed"
        workspace = temporary / "workspace"
        workspace.mkdir()
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                        "--no-index", "--target", str(target), str(wheel)], check=True)
        smoke = r'''
import sys
from pathlib import Path
target, checkout, workspace = map(Path, sys.argv[1:])
# Drop the editable checkout path added by site initialization.
sys.path[:] = [str(target)] + [p for p in sys.path if Path(p).resolve() not in (checkout, checkout / "src")]
import socket
def refuse_network(*args, **kwargs):
    raise AssertionError("Wheel smoke test attempted network access")
socket.socket.connect = refuse_network
import importlib
import pkgutil
from importlib.resources import files
import next_chapter
assert Path(next_chapter.__file__).resolve().is_relative_to(target)
modules = list(pkgutil.walk_packages(next_chapter.__path__, next_chapter.__name__ + "."))
for module in modules:
    importlib.import_module(module.name)
from next_chapter.paths import PROJECT_ROOT, CACHE_DIR, RUNTIME_DIR
assert PROJECT_ROOT == workspace
assert CACHE_DIR == workspace / "cache"
assert RUNTIME_DIR == workspace / "runtime"
assert ".workspace-hero" in files("next_chapter.ui").joinpath("assets/workspace.css").read_text()
from streamlit.testing.v1 import AppTest
app = AppTest.from_string("from next_chapter.ui.app import main\nmain()", default_timeout=30).run()
assert not app.exception, [item.message for item in app.exception]
def button(label):
    return next(item for item in app.button if item.label == label)
button("Explore a sample workspace").click().run()
assert app.session_state["page"] == "Job matches"
assert button("Find my matches").disabled
button("Start with my own CV").click().run()
assert app.session_state["page"] == "Your profile"
assert "profile" not in app.session_state
assert not app.exception, [item.message for item in app.exception]
assert not app.error, [item.value for item in app.error]
print(f"PASS: {len(modules)} production modules import from the wheel outside the checkout.")
print("PASS: packaged CSS, workspace paths, and sample navigation with network blocked.")
'''
        subprocess.run([sys.executable, "-I", "-c", smoke, str(target),
                        str(checkout), str(workspace)], cwd=workspace, check=True,
                       env={**os.environ, "NEXT_CHAPTER_HOME": str(workspace),
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "AGENT2_DATABASE_PATH": str(workspace / "runtime/check.sqlite3")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
