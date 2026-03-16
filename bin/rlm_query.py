import sys

from clawmory_rlm.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "query")
    raise SystemExit(main())
