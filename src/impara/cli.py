import argparse

from . import __version__, add, multiply, average


def _fmt(x):
    """Show 35.0 as 35, but keep 2.5 as 2.5."""
    return str(int(x)) if x == int(x) else str(x)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="impara", description="A simple calculator")
    parser.add_argument("--version", action="version", version=f"impara {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("add", help="add two numbers")
    p.add_argument("a", type=float)
    p.add_argument("b", type=float)
    p.set_defaults(func=lambda x: add(x.a, x.b))

    p = sub.add_parser("multiply", help="multiply two numbers")
    p.add_argument("a", type=float)
    p.add_argument("b", type=float)
    p.set_defaults(func=lambda x: multiply(x.a, x.b))

    p = sub.add_parser("average", help="average of one or more numbers")
    p.add_argument("numbers", type=float, nargs="+")
    p.set_defaults(func=lambda x: average(x.numbers))

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    print(_fmt(args.func(args)))
    return 0