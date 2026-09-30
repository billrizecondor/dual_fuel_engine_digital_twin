"""
Train the digital-twin models on the engine-mapping data and export them for the web demo.

Uses the same models as the Tkinter dashboard (dual_fuel_digital_twin/data_processing):
  * K-nearest neighbours (GridSearchCV over k, weights, p) for electrical efficiency vs. power
  * Linear regression for exhaust gas temperature vs. power
  * Energy balance for the diesel and methane mass flows at a given diesel energy share (DES)

Input:  dual_fuel_digital_twin/outputs/digital_twin_cleaned_24cols.csv
Output: docs/data/twin_model.json

Usage:
    pip install pandas numpy scikit-learn
    python analysis/build_demo_data.py
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dual_fuel_digital_twin" / "outputs" / "digital_twin_cleaned_24cols.csv"
OUT = ROOT / "docs" / "data" / "twin_model.json"

LHV_DIESEL = 42.7   # MJ/kg
LHV_CH4 = 50.03     # MJ/kg


def load() -> pd.DataFrame:
    cols = ["power_output", "efficiency_electric", "exhaust_temp", "diesel_mass_flow",
            "ch4_mass_flow_calc", "des_percent", "measured_ch4_percent", "sheet"]
    df = pd.read_csv(DATA)[cols].dropna()
    # Sheet names encode the test point, e.g. "P8C60_2" = 8 kW set-point at 60 % CH4 in the biogas.
    parsed = df["sheet"].str.extract(r"P(\d+)C(\d+)")
    df["setpoint_kw"] = parsed[0].astype(int)
    df["ch4_setpoint"] = parsed[1].astype(int)
    return df


def fit_efficiency_knn(df: pd.DataFrame):
    X, y = df[["power_output"]].values, df["efficiency_electric"].values
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())])
    grid = {"knn__n_neighbors": [9, 10], "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}
    search = GridSearchCV(pipe, grid, cv=5, scoring="r2", n_jobs=-1).fit(X, y)
    model = search.best_estimator_
    cv_pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return model, search, cv_pred


def fit_exhaust_lr(df: pd.DataFrame):
    X, y = df[["power_output"]].values, df["exhaust_temp"].values
    model = LinearRegression().fit(X, y)
    cv_pred = cross_val_predict(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    return model, cv_pred


def fuel_mass_flows(power_kw: float, efficiency: float, des: float) -> dict:
    """Energy balance: fuel energy in = power out / efficiency, split by diesel energy share."""
    q_total = power_kw / efficiency * 3.6                  # MJ/h
    return {
        "diesel_kg_h": des * q_total / LHV_DIESEL,
        "ch4_kg_h": (1 - des) * q_total / LHV_CH4,
    }


def metrics(y, pred) -> dict:
    return {"r2": round(float(r2_score(y, pred)), 3), "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 2)}


def main() -> None:
    df = load()
    knn, search, eff_cv = fit_efficiency_knn(df)
    lr, temp_cv = fit_exhaust_lr(df)

    params = search.best_params_
    eff_metrics = {"cv": metrics(df["efficiency_electric"], eff_cv),
                   "train": metrics(df["efficiency_electric"], knn.predict(df[["power_output"]].values))}
    temp_metrics = {"cv": metrics(df["exhaust_temp"], temp_cv),
                    "train": metrics(df["exhaust_temp"], lr.predict(df[["power_output"]].values))}

    example = 10.0
    eff = knn.predict([[example]])[0]
    flows = fuel_mass_flows(example, eff / 100, 0.15)
    print(f"{len(df)} operating points from {df['sheet'].nunique()} test runs")
    print(f"KNN efficiency: k={params['knn__n_neighbors']}, weights={params['knn__weights']}, p={params['knn__p']}, "
          f"CV R2={eff_metrics['cv']['r2']}, train R2={eff_metrics['train']['r2']}")
    print(f"Exhaust temperature: T = {lr.coef_[0]:.2f} x P + {lr.intercept_:.1f}, CV R2={temp_metrics['cv']['r2']}")
    print(f"At {example} kW, DES 15%: efficiency {eff:.2f}%, diesel {flows['diesel_kg_h']:.2f} kg/h, CH4 {flows['ch4_kg_h']:.2f} kg/h")

    rnd = lambda s, d=3: [round(float(v), d) for v in s]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "constants": {"lhv_diesel": LHV_DIESEL, "lhv_ch4": LHV_CH4},
        "knn": {"k": int(params["knn__n_neighbors"]), "weights": params["knn__weights"], "p": int(params["knn__p"]),
                "metrics": eff_metrics},
        "exhaust_lr": {"slope": float(lr.coef_[0]), "intercept": float(lr.intercept_), "metrics": temp_metrics},
        "n_runs": int(df["sheet"].nunique()),
        "points": {
            "power": rnd(df["power_output"]),
            "efficiency": rnd(df["efficiency_electric"]),
            "exhaust": rnd(df["exhaust_temp"], 1),
            "diesel": rnd(df["diesel_mass_flow"]),
            "ch4": rnd(df["ch4_mass_flow_calc"]),
            "des": rnd(df["des_percent"], 2),
            "ch4_setpoint": [int(v) for v in df["ch4_setpoint"]],
            "efficiency_cv_pred": rnd(eff_cv),
            "exhaust_cv_pred": rnd(temp_cv, 1),
        },
    }), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
