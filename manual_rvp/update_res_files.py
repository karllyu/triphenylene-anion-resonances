import pandas as pd
import numpy as np
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

print(f"Source file alpha values: {sorted(source_cols.keys())}")

# Define the mapping: (res_file, source_row_label)
mappings = [
    ("A2_res1.csv", "1/A2"),
    ("A2_res2.csv", "2/A2"),
    ("A2_res3.csv", "3/A2"),
]

# Process each res file in root directory only
for res_file, source_row in mappings:
    if os.path.exists(res_file):
        # Read the current res file to get the alpha values
        res_df = pd.read_csv(res_file)
        
        # Extract the alpha values from the res file
        alpha_values = res_df['alpha'].tolist()
        
        # Extract the corresponding energy values from the source file
        # and track which rows are valid
        energy_values = []
        valid_rows = []
        
        for idx, alpha in enumerate(alpha_values):
            if pd.isna(alpha):
                continue
                
            # Find matching column in source file
            alpha_found = False
            for col_float, col_str in source_cols.items():
                if abs(col_float - float(alpha)) < 1e-6:
                    value = source_df.loc[source_row, col_str]
                    energy_values.append(value)
                    valid_rows.append(idx)
                    alpha_found = True
                    break
            
            if not alpha_found:
                print(f"Deleting row with alpha={alpha} from {res_file} (not in source file)")
        
        # Keep only valid rows
        res_df_updated = res_df.iloc[valid_rows].copy()
        res_df_updated['energy'] = energy_values
        
        # Save the updated file
        res_df_updated.to_csv(res_file, index=False)
        print(f"Updated {res_file} with values from {source_row}")
        print(f"  Kept {len(valid_rows)} rows out of {len(alpha_values)}")
    else:
        print(f"File {res_file} not found")

print("Done!")
