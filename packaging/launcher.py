"""Entry point for the packaged CandleCast app (PyInstaller)."""
import multiprocessing
import os
import sys
import traceback


def _crash_log(exc_text: str) -> None:
    base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    folder = os.path.join(base, "CandleCast")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "crash.log"), "a", encoding="utf-8") as fh:
        fh.write(exc_text + "\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        from candlecast.app import main

        main(sys.argv[1:])
    except Exception:  # a windowed exe has no console: leave a log the user can send in an issue
        _crash_log(traceback.format_exc())
        raise
