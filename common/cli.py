"""Shared command-line argument helpers."""

import argparse
from typing import Optional


def add_boolean_argument(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    default: bool,
    help_text: str,
    dest: Optional[str] = None,
) -> None:
    """Add Python 3.8-compatible positive and negative boolean options."""
    if not option.startswith("--"):
        raise ValueError("boolean option must start with '--'")

    argument_dest = dest or option[2:].replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        option,
        dest=argument_dest,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        "--no-" + option[2:],
        dest=argument_dest,
        action="store_false",
        help="Disable: " + help_text,
    )
    parser.set_defaults(**{argument_dest: bool(default)})
