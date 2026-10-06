"""Command-line interface for the benchmark and its archived results."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tudtf", description=__doc__)
    parser.add_argument("command", choices=["run", "analyse", "verify", "compare"])
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv[:1])
    rest = argv[1:]
    if args.command == "run":
        from .study import main as command
    elif args.command == "analyse":
        from .analysis import main as command
    elif args.command == "compare":
        from .compare import main as command
    else:
        from .verify import main as command
    command(rest)


if __name__ == "__main__":
    main()
