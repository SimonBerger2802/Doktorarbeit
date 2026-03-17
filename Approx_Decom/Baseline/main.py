import argparse
import sys

from ubplanner import UBPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="UB-ANC Planner",
        description="UB-ANC Planner Python"
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Set input file containing area information."
    )

    parser.add_argument(
        "-r", "--resolution",
        type=float,
        default=10.0,
        help="Set resolution of the decomposition in meters."
    )

    parser.add_argument(
        "-l", "--limit",
        type=float,
        default=1000000000.0,
        help="Set optimizer time limit in seconds."
    )

    parser.add_argument(
        "-g", "--gap",
        type=float,
        default=0.01,
        help="Set gap to the optimal solution."
    )

    parser.add_argument(
        "-a", "--lambda_weight",
        dest="lambda_weight",
        type=float,
        default=1.0,
        help="Set distance factor in cost function."
    )

    parser.add_argument(
        "-m", "--gamma",
        type=float,
        default=1.0,
        help="Set turn factor in cost function."
    )

    parser.add_argument(
        "-k", "--kappa",
        type=float,
        default=1000000000.0,
        help="Set maximum capacity for each drone."
    )

    parser.add_argument(
        "-p", "--precision",
        type=float,
        default=1.0,
        help="Set precision for capacity calculation."
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    planner = UBPlanner()
    planner.set_file(args.file)
    planner.set_resolution(args.resolution)
    planner.set_limit(args.limit)
    planner.set_gap(args.gap)
    planner.set_lambda(args.lambda_weight)
    planner.set_gamma(args.gamma)
    planner.set_kappa(args.kappa)
    planner.set_precision(args.precision)

    try:
        planner.start_planner()
        planner.visualize_tours()
    except Exception as exc:
        print(f"Planner failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())