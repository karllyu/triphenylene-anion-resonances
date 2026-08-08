from pathlib import Path

import pandas as pd


SOURCE_FILE = Path(__file__).with_name("tri_cc-pvdz^M1s3p_neutral_reference.csv")
OUTLIER_ALPHAS = {1.94, 1.95}


def load_source_dataframe() -> tuple[pd.DataFrame, dict[float, str]]:
    source_df = pd.read_csv(SOURCE_FILE, index_col=0)

    source_cols: dict[float, str] = {}
    for col in source_df.columns:
        try:
            col_float = float(col)
        except ValueError:
            continue
        source_cols[col_float] = col

    return source_df, source_cols


def write_root_file(output_path: Path, alphas: list[float], energies: list[float]) -> None:
    lines = [f"{alpha:.15f}\t{energy:.15f}" for alpha, energy in zip(alphas, energies)]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_root_txts() -> None:
    source_df, source_cols = load_source_dataframe()
    alpha_values = sorted(alpha for alpha in source_cols if alpha not in OUTLIER_ALPHAS)

    root_groups = {
        "A2": [f"{root}/A2" for root in range(1, 7)],
        "B2": [f"{root}/B2" for root in range(1, 7)],
    }

    for prefix, row_labels in root_groups.items():
        for root_index, row_label in enumerate(row_labels, start=1):
            energies = [float(source_df.loc[row_label, source_cols[alpha]]) for alpha in alpha_values]
            output_path = Path(__file__).with_name(f"{prefix}_root{root_index}.txt")
            write_root_file(output_path, alpha_values, energies)
            print(f"Wrote {output_path.name} ({len(alpha_values)} rows)")


if __name__ == "__main__":
    export_root_txts()