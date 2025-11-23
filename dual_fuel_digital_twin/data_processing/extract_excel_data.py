import pandas as pd
import numpy as np
import os

# Konstanten
PCI_diesel = 42.7
PCI_ch4 = 50.03
rho_ch4 = 0.016
Vm_ch4 = 0.0224
cp_water = 4.18  # kJ/kg·K → MJ/kg·K = 0.00418

def create_final_dataframe():
    file_paths = [
        "data/raw/24-07-19_Engine mapping 2.xlsx",
        "data/raw/24-06-26_Engine mapping 1.xlsx"
    ]

    def extract_all_sheets_to_dataframe(file_paths):
        all_data = []
        for file_path in file_paths:
            excel_file = pd.ExcelFile(file_path)
            for sheet in excel_file.sheet_names:
                try:
                    df = excel_file.parse(sheet, skiprows=2)
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df = df.loc[:, ~df.columns.duplicated()]
                    df = df.dropna(axis=1, how='all')
                    df = df.dropna(how='all')
                    df['Sheet'] = sheet
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ Fehler bei {sheet} in {file_path}: {e}")
        combined_df = pd.concat(all_data, ignore_index=True)
        first_col = combined_df.columns[0]
        combined_df = combined_df[~combined_df[first_col].astype(str).str.contains(
            "mittel|moyenne|average|ø", case=False, na=False)]
        return combined_df

    # Daten einlesen & vorbereiten
    combined_df = extract_all_sheets_to_dataframe(file_paths)
    combined_df.dropna(how='all', inplace=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.str.contains("zeit|time", case=False, na=False)]
    numeric_cols = combined_df.columns.difference(['Sheet'])
    combined_df[numeric_cols] = combined_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Berechnungen
    combined_df['ṁ CH₄ (kg/h)'] = combined_df['FT08(ln/min)'] * 0.001 * 60 * rho_ch4
    combined_df['Q_CH4'] = combined_df['ṁ CH₄ (kg/h)'] * PCI_ch4
    combined_df['% CH4 réel'] = 100 * combined_df['Q_CH4'] / (
        combined_df['Q_CH4'] + combined_df['Q CO2pd(ln/min)']
    )
    combined_df['ṁ CH4 (kg/h) Formel'] = (
        combined_df['FT08(ln/min)'] * 0.001 * 60 *
        (combined_df['% CH4 réel'] / 100) *
        (rho_ch4 / Vm_ch4)
    )
    combined_df['DES (%)'] = 100 * (
        (combined_df['FT05(kg/h)'] * PCI_diesel) /
        ((combined_df['FT05(kg/h)'] * PCI_diesel) + (combined_df['ṁ CH4 (kg/h) Formel'] * PCI_ch4))
    )
    combined_df['η elec (%)'] = 100 * (
        combined_df['JT11(kW)'] /
        (((PCI_diesel / 3.6) * combined_df['FT05(kg/h)']) +
         ((PCI_ch4 / 3.6) * combined_df['ṁ CH4 (kg/h) Formel']))
    )
    combined_df['η therm (%)'] = 100 * (
        ((combined_df['FT07(l/min)'] * 0.001 * 60) *
         (combined_df['TE02(°C)'] - combined_df['TE03(°C)']) *
         (cp_water / 3.6)) /
        (((PCI_diesel / 3.6) * combined_df['FT05(kg/h)']) +
         ((PCI_ch4 / 3.6) * combined_df['ṁ CH4 (kg/h) Formel']))
    )

    # 24 Spalten erzeugen
    mess_spalten = combined_df.columns.difference([
        '% CH4 réel', 'ṁ CH4 (kg/h) Formel', 'DES (%)', 'η elec (%)', 'η therm (%)', 'Q_CH4', 'Sheet'
    ])[:18].tolist()

    final_cols = mess_spalten + [
        '% CH4 réel', 'ṁ CH4 (kg/h) Formel', 'DES (%)', 'η elec (%)', 'η therm (%)', 'Sheet'
    ]
    final_df = combined_df[final_cols].copy()

    # Spaltennamen umbenennen
    rename_columns = {
        '% vanne gaz': "gas_valve_position_percent",
        'AT09(%CH4)': "measured_ch4_percent",
        'ET12(V)': "voltage",
        'FT05(kg/h)': "diesel_mass_flow",
        'FT07(l/min)': "water_flow",
        'FT08(ln/min)': "ch4_volumeflow_raw",
        'IT13(A)': "current_phase_1",
        'IT15(A)': "current_phase_2",
        'JT11(kW)': "power_output",
        'PT04(bar abs)': "boost_pressure",
        'PT16(bar abs)': "exhaust_pressure",
        'Q CH4(ln/min)': "ch4_volumeflow",
        'Q CO2gd(ln/min)': "co2_volumeflow_raw",
        'Q CO2pd(ln/min)': "co2_volumeflow_processed",
        'ST14(Hz)': "generator_frequency",
        'TE02(°C)': "cooling_water_out_temp",
        'TE03(°C)': "cooling_water_in_temp",
        'TE10(°C)': "exhaust_temp",
        '% CH4 réel': "calculated_ch4_share_percent",
        'ṁ CH4 (kg/h) Formel': "ch4_mass_flow_calc",
        'DES (%)': "des_percent",
        'η elec (%)': "efficiency_electric",
        'η therm (%)': "efficiency_thermal",
        'Sheet': "sheet"
    }
    final_df.rename(columns=rename_columns, inplace=True)

    # Speichern
    os.makedirs("outputs", exist_ok=True)
    final_df.to_csv("outputs/digital_twin_cleaned_24cols.csv", index=False)

    return final_df