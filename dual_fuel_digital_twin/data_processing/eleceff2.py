import pandas as pd
import matplotlib.pyplot as plt

# File paths to both Excel files
file_paths = [
    "data/raw/24-07-19_Engine mapping 2.xlsx",
    "data/raw/24-06-26_Engine mapping 1.xlsx"
]

# Column indices for B to S and Q
row26_indices = list(range(1, 19))  # Columns B to S (indices 1–18)
efficiency_col_index = 16           # Column Q for η elec (%)

# Containers
row26_data = []
efficiencies = []
labels = None

# Process each sheet in both files
for file_path in file_paths:
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        try:
            if labels is None:
                labels = df.iloc[3, row26_indices].tolist()  # Row 4 = labels

            values = df.iloc[25, row26_indices].tolist()     # Row 26 = feature values
            eff = df.iloc[1, efficiency_col_index]           # Q2 = η elec (%)

            row26_data.append(values)
            efficiencies.append(eff)
        except IndexError:
            continue

# Create DataFrame
row26_df = pd.DataFrame(row26_data, columns=labels)
eff_series = pd.Series(efficiencies, name="η elec (%)")

# Plot subplots: η elec (%) vs each B–S column (row 26)
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
axes = axes.flatten()

for i, feature in enumerate(row26_df.columns):
    axes[i].scatter(row26_df[feature], eff_series)
    axes[i].set_xlabel(feature, fontsize=8)
    axes[i].set_ylabel("η elec (%)", fontsize=8)
    axes[i].set_title(f"η elec (%) vs {feature}", fontsize=9)
    axes[i].tick_params(labelsize=8)
    axes[i].grid(True)

# Remove unused axes if any
for j in range(len(row26_df.columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()