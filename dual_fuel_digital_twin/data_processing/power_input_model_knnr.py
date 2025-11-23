import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def predict_efficiency_with_tuned_knnr(df, target_power_output):
    """
    Trains a KNN regressor with GridSearchCV to predict electrical efficiency
    based on power output, and compares it with the closest measured value.

    Parameters:
        df (DataFrame): Must contain 'power_output' and 'efficiency_electric'
        target_power_output (float): Desired power output in kW

    Returns:
        dict: Includes prediction, closest datapoint, difference, metrics, and model
    """

    # --- 1. Clean input ---
    df_clean = df[['power_output', 'efficiency_electric']].dropna()
    X = df_clean[['power_output']].values
    y = df_clean['efficiency_electric'].values

    # --- 2. Pipeline with scaler and KNN ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])

    # --- 3. Hyperparameter grid ---
    param_grid = {
        'knn__n_neighbors': [9,10],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # 1: Manhattan, 2: Euclidean
    }

    # --- 4. Grid Search CV ---
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # --- 5. Predict efficiency for input ---
    predicted_eff = best_model.predict(np.array([[target_power_output]]))[0]

    # --- 6. Find closest real datapoint ---
    df_clean['abs_diff'] = np.abs(df_clean['power_output'] - target_power_output)
    closest_row = df_clean.loc[df_clean['abs_diff'].idxmin()]
    real_eff = closest_row['efficiency_electric']
    real_power = closest_row['power_output']

    # --- 7. Train performance metrics ---
    y_pred_train = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))
    r2 = r2_score(y, y_pred_train)

    # --- 8. Return result ---
    return {
        "target_power_output": round(target_power_output, 2),
        "predicted_efficiency": round(predicted_eff, 2),
        "closest_measured_power": round(real_power, 2),
        "measured_efficiency": round(real_eff, 2),
        "difference": round(predicted_eff - real_eff, 2),
        "best_params": grid_search.best_params_,
        "cv_r2": round(grid_search.best_score_, 4),
        "train_rmse": round(rmse, 4),
        "train_r2": round(r2, 4),
        "model": best_model  # Needed for GUI plotting
    }