import sys

from labgroupassigner.errors import SolverError
from labgroupassigner.preprocess import load_and_prepare
from labgroupassigner.model import build_and_solve
from labgroupassigner.report import print_report


def main():
    if len(sys.argv) == 2:
        try:
            data = load_and_prepare(sys.argv[1])
            result = build_and_solve(data)
            print_report(data, result["assignments"])
        except SolverError as exc:
            sys.exit(str(exc))
    else:
        sys.exit(
            "Usage: labgroupassigner <roster.csv>"
        )
