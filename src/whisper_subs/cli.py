"""Command-line entry point. Milestone 1 fills this in."""

import sys

from whisper_subs import __version__


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-V", "--version"):
        print(f"whisper-subs {__version__}")
        return 0
    print("whisper-subs: not implemented yet (see docs/DEVELOPMENT.md, milestone 1)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
