import pandas as pd
import matplotlib.pyplot as plt

# File paths to both Excel files
file_paths = [
    "data/raw/24-07-19_Engine mapping 2.xlsx",
    "data/raw/24-06-26_Engine mapping 1.xlsx"
]

# Excel column indices: K (10), M (12), O (14), Q (16), U (20)
col_indices = [10, 12, 14, 16, 20]

# List to collect rows from all sheets
combined_rows = []

# Iterate through each file and each sheet
for file_path in file_paths:
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        try:
            headers = df.iloc[0, col_indices].tolist()
            values = df.iloc[1, col_indices].tolist()
            row_dict = dict(zip(headers, values))
            combined_rows.append(row_dict)
        except IndexError:
            continue  # skip malformed sheets

# Create a combined DataFrame
combined_df = pd.DataFrame(combined_rows)

# Define target and features
target = "η elec (%)"
features = [col for col in combined_df.columns if col != target]

# Plot subplots: target vs each feature
fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 5))

# Ensure axes is iterable
if len(features) == 1:
    axes = [axes]

# Generate each subplot
for ax, feature in zip(axes, features):
    ax.scatter(combined_df[feature], combined_df[target])
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{target} vs {feature}")
    ax.grid(True)

plt.tight_layout()
plt.show()