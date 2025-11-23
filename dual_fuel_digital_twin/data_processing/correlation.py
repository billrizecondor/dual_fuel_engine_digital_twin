import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataframe_correlation(df, title="Correlation Matrix"):
    """
    Berechnet und visualisiert die Korrelationsmatrix für einen DataFrame
    und gibt die Top-3-Korrelationen für jede Variable zurück.

    Parameter:
    - df: DataFrame mit numerischen Spalten
    - title: Titel der Heatmap

    Rückgabe:
    - top_corrs_df: DataFrame mit den Top-3-Korrelationen je Feature
    """

    # Nur numerische Spalten
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()

    # --- Heatmap-Plot ---
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        annot_kws={"size": 8},  # 🧠 kleinere Werte
        cbar_kws={"label": "Correlation Coefficient"}
    )

    plt.title(title, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

    # --- Top-3-Korrelationen ---
    top_corrs = []
    for col in corr_matrix.columns:
        top_related = corr_matrix[col].drop(labels=[col]).abs().sort_values(ascending=False).head(3)
        for related_col in top_related.index:
            top_corrs.append({
                "Feature": col,
                "Correlated With": related_col,
                "Correlation": corr_matrix.loc[col, related_col]
            })

    top_corrs_df = pd.DataFrame(top_corrs)

    print("\nTop 3 Correlations for Each Feature:\n")
    print(top_corrs_df.to_string(index=False))

    return top_corrs_df