# Dual-Fuel Engine Digital Twin

Digitalization of a dual-fuel engine and creation of its digital twin. Measured engine-mapping data is processed in Python, and machine-learning models predict key operating outputs. The results are paired with a 3D model of the engine and integrated into the GreenTwin web platform.

**Team:** Billriz Condor, Hafeez Bashir, Kien van Ho. PM3E/ME3+, IMT Atlantique Nantes, June 2025.

📄 Full write-up: [Project Report (PDF)](<Project Report GDocs Version (1).pdf>)

## What it does

- **Data pipeline**: extracts and cleans two engine-mapping campaigns (`data/raw/*.xlsx`) into a single dataset (`outputs/digital_twin_cleaned_24cols.csv`) and calculates the mass flows.
- **Correlation analysis**: finds which measured inputs drive the engine's outputs.
- **Predictive models**:
  - Linear regression for **exhaust gas temperature**
  - K-nearest neighbours regression for **electrical efficiency** from power output
  - Linear, KNN and SVR variants for power input
- **Interactive GUI**: a Tkinter dashboard where you enter operating conditions and read the predicted parameters.
- **3D model**: an engine CAD model built in Blender and shown in the GreenTwin web application with live input forms and time-series graphs.

## Project structure

```text
dual_fuel_digital_twin/
├── main.py                  Entry point (launches the interactive GUI)
├── data/raw/                Engine mapping measurements (Excel)
├── data_processing/         Extraction, correlation, models, GUI
├── outputs/                 Cleaned dataset
└── plots/                   Model and correlation plots
```

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
cd dual_fuel_digital_twin
python main.py
```

## Tech

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Tkinter · Blender
