import pandas as pd
import os

# Path to the source file
source_file = "tri_cc-pvdz^M1s3p_neutral_reference.csv"

# Read the source file
source_df = pd.read_csv(source_file, index_col=0)

# Column names in source file - convert to strings and then to floats for matching
source_cols = {}
for col in source_df.columns:
    try:
        col_float = float(col)
        source_cols[col_float] = col
    except:
        pass

# Outlier alpha values to exclude
outlier_alphas = {1.94, 1.95}

source_alpha_values = sorted([a for a in source_cols.keys() if a not in outlier_alphas])
print(f"Source file alpha values (excluding outliers): {len(source_alpha_values)}")

# B2 roots file
roots_file = "B2_roots_1-6.csv"

# Define which B2 rows to use for each root
b2_rows = ["1/B2", "2/B2", "3/B2", "4/B2", "5/B2", "6/B2"]

# Create a new dataframe with all 6 roots
new_data = {
    "alpha": source_alpha_values
}

# Extract each root from the corresponding B2 row
for idx, b2_row in enumerate(b2_rows):
    root_col_name = f"root{idx+1}"
    root_values = []
    
    for alpha in source_alpha_values:
        col_str = source_cols[alpha]
        value = source_df.loc[b2_row, col_str]
        root_values.append(value)
    
    new_data[root_col_name] = root_values

# Create the new dataframe
new_df = pd.DataFrame(new_data)

# Save to the roots file
new_df.to_csv(roots_file, index=False)
print(f"Updated {roots_file}")
print(f"  Total rows: {len(new_df)}")
print(f"  Columns: {list(new_df.columns)}")
