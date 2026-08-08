from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from clustering import clustering
from pade import pade, save_results
from stabilization import (
    StabilizationFail,
    StabilizationSuccess,
    stabilization,
    verify_data,
)


def run_pade_sections(stable_zone: np.ndarray, output_dir: Path, min_size: int, max_size: int) -> pd.DataFrame:
    result_df = pd.DataFrame()

    for start in range(len(stable_zone) - min_size + 1):
        for size in range(min_size, max_size + 1):
            end = start + size
            if end > len(stable_zone):
                continue

            sliced_data = stable_zone[start:end]
            curr_result = pade(sliced_data)
            if curr_result is None:
                continue

            save_results(
                sliced_data,
                curr_result,
                output_dir / f"pade_output_from_{start + 1}_to_{end}.dat",
            )
            result_df = pd.concat(
                [result_df, curr_result.loc[curr_result["imag"] < 0, ["real", "imag", "alpha", "theta", "imag_err"]]],
                ignore_index=True,
            )

    return result_df


def run_auto_rvp_sequential(
    input_file: Path,
    threshold: float = 1.3,
    minimum_stable_zone_points: int = 10,
    interpolation_percentage: float = 0.4,
    stabilization_output_size: int = 25,
    maximum_derivative: float = 1.0,
    skip_stabilization: bool = False,
    stabilization_smooth_only: bool = False,
    min_pade_input_size: int = 8,
    max_pade_input_size: int = 35,
) -> Path:
    data = np.genfromtxt(input_file)
    error = verify_data(
        data,
        threshold,
        stabilization_output_size,
        minimum_stable_zone_points,
        maximum_derivative,
        interpolation_percentage,
    )
    if error is not None:
        raise ValueError(error)

    if skip_stabilization and stabilization_smooth_only:
        raise ValueError("skip_stabilization and stabilization_smooth_only cannot both be True")
    if min_pade_input_size < 8:
        raise ValueError("min_pade_input_size must be at least 8")
    if max_pade_input_size < min_pade_input_size:
        raise ValueError("max_pade_input_size must be >= min_pade_input_size")

    if skip_stabilization:
        stabilization_result = StabilizationSuccess(data, data)
    else:
        stabilization_result = stabilization(
            data,
            threshold,
            stabilization_output_size,
            stabilization_output_size,
            interpolation_percentage,
            minimum_stable_zone_points,
            stabilization_smooth_only,
        )

    if isinstance(stabilization_result, StabilizationFail):
        raise ValueError(str(stabilization_result))

    stable_zone = stabilization_result.get_results()

    output_dir = input_file.parent / f"results_{input_file.stem}"
    output_dir.mkdir(exist_ok=True)

    np.savetxt(output_dir / "stabilization_output.dat", stable_zone, fmt="%.15f")
    np.savetxt(output_dir / input_file.name, data, fmt="%.15f")

    effective_max_size = min(max_pade_input_size, len(stable_zone))
    result_df = run_pade_sections(stable_zone, output_dir, min_pade_input_size, effective_max_size)
    result_df.to_csv(output_dir / "clustering_input.csv", index=False)
    if result_df.empty:
        raise ValueError("No negative-imaginary Pade results were produced for clustering")

    clustering_results = clustering(result_df)
    if clustering_results is None:
        raise ValueError("Failed to find a cluster")

    clustering_results.save_results(str(output_dir))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic-rvp sequentially on one or more txt inputs.")
    parser.add_argument("inputs", nargs="+", help="Input .txt files such as A2_root1.txt")
    parser.add_argument("--threshold", type=float, default=1.3)
    parser.add_argument("--minimum-stable-zone-points", type=int, default=10)
    parser.add_argument("--interpolation-percentage", type=float, default=0.4)
    parser.add_argument("--stabilization-output-size", type=int, default=25)
    parser.add_argument("--maximum-derivative", type=float, default=1.0)
    parser.add_argument("--skip-stabilization", action="store_true")
    parser.add_argument("--stabilization-smooth-only", action="store_true")
    parser.add_argument("--min-pade-input-size", type=int, default=8)
    parser.add_argument("--max-pade-input-size", type=int, default=35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code = 0

    for raw_input in args.inputs:
        input_file = Path(raw_input)
        try:
            output_dir = run_auto_rvp_sequential(
                input_file=input_file,
                threshold=args.threshold,
                minimum_stable_zone_points=args.minimum_stable_zone_points,
                interpolation_percentage=args.interpolation_percentage,
                stabilization_output_size=args.stabilization_output_size,
                maximum_derivative=args.maximum_derivative,
                skip_stabilization=args.skip_stabilization,
                stabilization_smooth_only=args.stabilization_smooth_only,
                min_pade_input_size=args.min_pade_input_size,
                max_pade_input_size=args.max_pade_input_size,
            )
            print(f"{input_file.name}: wrote results to {output_dir}")
        except Exception as exc:
            exit_code = 1
            print(f"{input_file.name}: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
