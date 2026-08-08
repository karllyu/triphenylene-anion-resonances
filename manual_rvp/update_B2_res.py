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

# Define the mapping for B2 res files: (res_file, source_row_label)
res_mappings = [
    ("B2_res1.csv", "1/B2"),
    ("B2_res2.csv", "2/B2"),
    ("B2_res3.csv", "3/B2"),
]

# Process each B2 res file
for res_file, source_row in res_mappings:
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
            
            # Skip if alpha is an outlier
            if alpha in outlier_alphas:
                print(f"Skipping outlier alpha={alpha} from {res_file}")
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
        
        # Find missing alpha values in the res file (excluding outliers)
        existing_alphas = set()
        for alpha in alpha_values:
            if not pd.isna(alpha) and alpha not in outlier_alphas:
                for col_float in source_cols.keys():
                    if abs(col_float - float(alpha)) < 1e-6:
                        existing_alphas.add(col_float)
                        break
        
        missing_alphas = [a for a in source_alpha_values if a not in existing_alphas]
        
        # Add rows for missing alpha values (excluding outliers)
        if missing_alphas:
            print(f"Adding {len(missing_alphas)} missing alpha values to {res_file}")
            new_rows_data = []
            for alpha in missing_alphas:
                col_str = source_cols[alpha]
                value = source_df.loc[source_row, col_str]
                new_rows_data.append({"alpha": alpha, "energy": value})
            
            # Add new rows to the dataframe
            new_rows_df = pd.DataFrame(new_rows_data)
            res_df_updated = pd.concat([res_df_updated, new_rows_df], ignore_index=True)
            
            # Sort by alpha
            res_df_updated = res_df_updated.sort_values('alpha').reset_index(drop=True)
        
        # Save the updated file
        res_df_updated.to_csv(res_file, index=False)
        print(f"Updated {res_file}")
        print(f"  Total rows: {len(res_df_updated)}")
    else:
        print(f"File {res_file} not found")

print("Done!")
