# Dual-Fuel Engine Digital Twin

A digital twin of an 18 kW, four-cylinder generator (Cat C2.2-based DE18E3) converted to run on **biogas with a diesel pilot**. Measured engine-mapping data is processed in Python, machine-learning models predict the engine's key outputs, and the results are paired with a 3D model on the GreenTwin web platform.

🌐 **Live demo:** https://billrizecondor.github.io/dual_fuel_engine_digital_twin/ · 📄 [Project report (PDF)](<Project Report GDocs Version (1).pdf>)

**Team:** Billriz Condor, Hafeez Bashir, Kien van Ho. PM3E/ME3+, IMT Atlantique Nantes, June 2025.

## Key results

| Model | Predicts | Performance |
|---|---|---|
| K-nearest neighbours (k = 10, uniform, Manhattan) | Electrical efficiency from power output | R² 0.929 (training), 0.911 (5-fold cross-validation) |
| Linear regression (T = 18.56 × P + 141.6) | Exhaust gas temperature from power output | R² 0.956 |

Both models are trained on **1,750 measured operating points** from **53 test runs** (0–14 kW, 40–70% methane in the biogas).

## Live demo

The [interactive demo](https://billrizecondor.github.io/dual_fuel_engine_digital_twin/) runs the twin in your browser:
- set a power output and diesel energy share, and watch the engine schematic update its sensor readings
- compare the twin's prediction with the measured average near that operating point
- operating maps for efficiency, fuel mass flows and exhaust temperature
- cross-validated predicted vs. measured plots for both models
- the model code, loaded directly from this repo

## What the project includes

- **Data pipeline**: extracts and cleans two engine-mapping campaigns (`data/raw/*.xlsx`) into one dataset (`outputs/digital_twin_cleaned_24cols.csv`), with methane content, CH₄ mass flow, diesel energy share and efficiencies calculated from the sensor readings.
- **Correlation analysis**: finds which measured inputs drive the engine's outputs.
- **Predictive models**: KNN for efficiency and linear regression for exhaust temperature, plus linear, KNN and SVR variants for power input.
- **Energy balance**: diesel and methane mass flows from power, efficiency and diesel energy share, using lower heating values of 42.7 MJ/kg (diesel) and 50.03 MJ/kg (CH₄).
- **Interactive GUI**: a Tkinter dashboard for entering operating conditions and reading the predictions.
- **3D model**: an engine CAD model built in Blender, deployed on the GreenTwin web platform via Microsoft Azure and FastAPI.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl

# Tkinter dashboard
cd dual_fuel_digital_twin
python main.py

# Train, cross-validate and export the models for the web demo
cd ..
python analysis/build_demo_data.py
```

## Project structure

```text
dual_fuel_digital_twin/
├── main.py                  Entry point (launches the Tkinter dashboard)
├── data/raw/                Engine-mapping measurements (Excel)
├── data_processing/         Extraction, correlation, models, GUI
├── outputs/                 Cleaned dataset
└── plots/                   Model and correlation plots
analysis/                    Model training and export for the web demo
docs/                        Interactive demo (GitHub Pages)
```

## Tech

Python · pandas · NumPy · scikit-learn · Matplotlib · Tkinter · Blender · Chart.js
