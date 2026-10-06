"""Command-line interface for the benchmark and its archived results."""
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(prog="tudtf", description=__doc__)
    parser.add_argument("command", choices=["run", "analyse", "verify"])
    args, rest = parser.parse_known_args(argv)
    if args.command == "run":
        from .study import main as command
    elif args.command == "analyse":
        from .analysis import main as command
    else:
        from .verify import main as command
    command(rest)

if __name__ == "__main__":
    main()
