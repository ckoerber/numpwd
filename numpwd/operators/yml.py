"""Decompose operator from expression using yml files."""
from logging import getLogger, basicConfig, DEBUG, INFO

from yaml import safe_load
from json import dumps

from numpwd.operators.expression import decompose_operator
from numpwd.operators.h5 import write

LOGGER = getLogger("numpwd")


def read_yml(filename: str):
    """Read yml file and call decompose_operator."""
    LOGGER.info("Reading `%s`", filename)
    with open(filename, "r") as inp:
        data = safe_load(inp)
    LOGGER.info("Data\n:%s", dumps(data))
    op = decompose_operator(**data)

    return op


def main():
    """CLI for yaml read."""
    from argparse import ArgumentParser

    level_config = {"debug": DEBUG, "info": INFO}

    parser = ArgumentParser(description="Run numpwd for operator over yml file.")
    parser.add_argument("input", help="Name of the yml input file", type=str)
    parser.add_argument("-o", "--output", help="Name of the yml input file", type=str)

    parser.add_argument(
        "-l",
        "--log",
        help="Provide logging level.",
        choices=["debug", "info"],
        default="info",
    )
    args = parser.parse_args()
    level = level_config.get(args.log)
    basicConfig(level=level)

    op = read_yml(args.input)

    if args.output:
        write(op, args.output)


if __name__ == "__main__":
    main()
