import pandas as pd
import matplotlib.pyplot as plt

# File paths to both Excel files
file_paths = [
    "data/raw/24-07-19_Engine mapping 2.xlsx",
    "data/raw/24-06-26_Engine mapping 1.xlsx"
]

# Define column indices: B–S (1–18), and J (9) for power output
col_indices = list(range(1, 19))  # B to S
power_col_index = 9               # J

# Containers for data
all_feature_rows = []
all_power_outputs = []
feature_labels = None

# Read both Excel files and extract values
for file_path in file_paths:
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        try:
            # Row 4 (index 3) has labels — only store once
            if feature_labels is None:
                feature_labels = df.iloc[3, col_indices].tolist()

            # Row 26 (index 25): values
            feature_values = df.iloc[25, col_indices].tolist()
            power_value = df.iloc[25, power_col_index]

            all_feature_rows.append(feature_values)
            all_power_outputs.append(power_value)
        except IndexError:
            continue

# Create DataFrame
features_df = pd.DataFrame(all_feature_rows, columns=feature_labels)
power_series = pd.Series(all_power_outputs, name="Avg Power Output (J26)")

# Plot: All subplots in a smaller layout (fits in one figure)
fig, axes = plt.subplots(3, 6, figsize=(18, 9))  # Smaller layout
axes = axes.flatten()

for i, feature in enumerate(features_df.columns):
    axes[i].scatter(features_df[feature], power_series)
    axes[i].set_xlabel(feature, fontsize=8)
    axes[i].set_ylabel("Avg Power Output", fontsize=8)
    axes[i].set_title(f"Power vs {feature}", fontsize=9)
    axes[i].tick_params(labelsize=8)
    axes[i].grid(True)

# Remove any unused axes
for j in range(len(features_df.columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()