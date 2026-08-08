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

# Outlier alpha values to exclude
outlier_alphas = {1.94, 1.95}

source_alpha_values = sorted([a for a in source_cols.keys() if a not in outlier_alphas])
print(f"Source file alpha values (excluding outliers): {source_alpha_values}")

# Define the mapping: (roots_file, source_row_labels)
# source_row_labels is a list of (column_name, source_row) tuples
mappings = [
    ("A2_roots_1,2.csv", [("root1", "1/A2"), ("root2", "2/A2")]),
    ("A2_roots_2,3.csv", [("root2", "2/A2"), ("root3", "3/A2")]),
    ("A2_roots_2,3,4,5.csv", [("root2", "2/A2"), ("root3", "3/A2"), ("root4", "4/A2"), ("root5", "5/A2")]),
    ("A2_roots_3,4.csv", [("root3", "3/A2"), ("root4", "4/A2")]),
]

# Process each roots file
for roots_file, column_mappings in mappings:
    if os.path.exists(roots_file):
        # Read the current roots file to get the alpha values
        roots_df = pd.read_csv(roots_file)
        
        # Extract the alpha values from the roots file
        alpha_values = roots_df['alpha'].tolist()
        
        # Extract the corresponding root values from the source file
        # and track which rows are valid
        new_columns = {}
        valid_rows = []
        new_rows_data = []
        
        for col_name, source_row in column_mappings:
            new_columns[col_name] = []
        
        # Process existing alpha values
        for idx, alpha in enumerate(alpha_values):
            if pd.isna(alpha):
                continue
            
            # Skip if alpha is an outlier
            if alpha in outlier_alphas:
                print(f"Skipping outlier alpha={alpha} from {roots_file}")
                continue
                
            # Find matching column in source file
            alpha_found = False
            for col_float, col_str in source_cols.items():
                if abs(col_float - float(alpha)) < 1e-6:
                    # This alpha exists in source file and is not an outlier
                    # Extract all required columns for this alpha
                    for col_name, source_row in column_mappings:
                        value = source_df.loc[source_row, col_str]
                        new_columns[col_name].append(value)
                    
                    valid_rows.append(idx)
                    alpha_found = True
                    break
            
            if not alpha_found:
                print(f"Deleting row with alpha={alpha} from {roots_file} (not in source file)")
        
        # Keep only valid rows
        roots_df_updated = roots_df.iloc[valid_rows].copy()
        
        # Update all root columns with extracted values
        for col_name, values in new_columns.items():
            roots_df_updated[col_name] = values
        
        # Find missing alpha values in the roots file (excluding outliers)
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
            print(f"Adding {len(missing_alphas)} missing alpha values to {roots_file}")
            for alpha in missing_alphas:
                col_str = source_cols[alpha]
                row_data = {"alpha": alpha}
                for col_name, source_row in column_mappings:
                    value = source_df.loc[source_row, col_str]
                    row_data[col_name] = value
                new_rows_data.append(row_data)
            
            # Add new rows to the dataframe
            new_rows_df = pd.DataFrame(new_rows_data)
            roots_df_updated = pd.concat([roots_df_updated, new_rows_df], ignore_index=True)
            
            # Sort by alpha
            roots_df_updated = roots_df_updated.sort_values('alpha').reset_index(drop=True)
        
        # Save the updated file
        roots_df_updated.to_csv(roots_file, index=False)
        print(f"Updated {roots_file}")
        print(f"  Total rows: {len(roots_df_updated)}")
        print(f"  Columns: {list([col for col in roots_df_updated.columns if col != 'alpha'])}")
    else:
        print(f"File {roots_file} not found")

print("Done!")
