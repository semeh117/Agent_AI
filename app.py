"""Start Next Chapter locally or on Streamlit Community Cloud."""

from pathlib import Path
import sys


# Community Cloud runs this file directly and does not install the repository's
# src-layout package. Make the package importable from a fresh checkout while
# retaining normal editable-install behavior for local development.
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from next_chapter.ui.app import main


if __name__ == "__main__":
    main()
