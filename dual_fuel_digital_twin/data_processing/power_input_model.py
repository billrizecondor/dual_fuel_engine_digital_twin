import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


def predict_efficiency_with_tuned_gb(df, target_power_output):
    """
    Sucht per GridSearchCV die besten Hyperparameter für GradientBoostingRegressor
    und sagt die elektrische Effizienz für einen Zielwert voraus.

    Parameter:
    - df: DataFrame mit 'power_output' und 'efficiency_electric'
    - target_power_output: float, gewünschte Ausgangsleistung

    Rückgabe:
    - dict mit Vorhersage, echtem Wert und Modellmetriken
    """

    # 1. Daten vorbereiten
    df_clean = df[['power_output', 'efficiency_electric']].dropna()
    X = df_clean[['power_output']].values
    y = df_clean['efficiency_electric'].values

    # 2. Hyperparameter-Grid
    param_grid = {
        'n_estimators': [100, 200, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3],
        'max_depth': [2, 3, 4],
        'min_samples_leaf': [1, 3, 5],
        'subsample': [0.8, 1.0, 1.5]
    }

    # 3. GridSearch mit CV
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_

    # 4. Vorhersage
    predicted_eff = best_model.predict(np.array([[target_power_output]]))[0]

    # 5. Nächstliegender Messwert im DataFrame
    df_clean['abs_diff'] = np.abs(df_clean['power_output'] - target_power_output)
    closest_row = df_clean.loc[df_clean['abs_diff'].idxmin()]
    real_eff = closest_row['efficiency_electric']
    real_power = closest_row['power_output']

    # 6. Performance auf Trainingsdaten
    y_pred_train = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))
    r2 = r2_score(y, y_pred_train)

    return {
        "target_power_output": round(target_power_output, 2),
        "predicted_efficiency": round(predicted_eff, 2),
        "closest_measured_power": round(real_power, 2),
        "measured_efficiency": round(real_eff, 2),
        "difference": round(predicted_eff - real_eff, 2),
        "best_params": grid_search.best_params_,
        "cv_r2": round(grid_search.best_score_, 4),
        "train_rmse": round(rmse, 4),
        "train_r2": round(r2, 4)
    }