import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
# import json

from data_processing.extract_excel_data import create_final_dataframe
from data_processing.power_input_model_knnr import predict_efficiency_with_tuned_knnr
from data_processing.calculate_massflows import calculate_fuel_mass_flows
from data_processing.exhaust_temp_model import train_exhaust_temp_model


# Save function output to a txt file
def save_output_to_txt(output, filename):
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory

    # Define the full file path in the same folder as the script
    file_path = os.path.join(script_dir, filename)

    # Open the file in write mode and save the output
    with open(file_path, 'w') as file:
        file.write(output)

def run_interactive_gui():
    df = create_final_dataframe()
    exhaust_model = train_exhaust_temp_model(df)

    def run_calculations():
        try:
            des_percent = float(des_entry.get())
            power = float(power_entry.get())
            des = des_percent / 100

            result = predict_efficiency_with_tuned_knnr(df, power)
            print(result)
            # result_str = json.dumps(result, indent=4)
            # save_output_to_txt(result_str,'result.txt')
            predicted_eff = result["predicted_efficiency"] / 100
            mass_flows = calculate_fuel_mass_flows(power, predicted_eff, des)
            predicted_temp = exhaust_model.predict(np.array([[power]]))[0]

            df['abs_diff'] = abs(df['power_output'] - power)
            closest = df.loc[df['abs_diff'].idxmin()]
            real_eff = closest['efficiency_electric'] / 100
            real_temp = closest['exhaust_temp']

            voltage = 230
            rpm = 1500
            poles = 2
            current = (power * 1000) / voltage
            frequency = rpm * poles / 120
            real_current = (closest['power_output'] * 1000) / voltage
            real_frequency = frequency

            output_text = (
                f"{'Parameter':<20}{'Predicted/Calculated':<30}{'Measured (Closest)'}\n"
                f"{'-'*75}\n"
                f"{'DES (%)':<20}{des_percent:>18.2f}{closest['des_percent']:>25.2f}\n"
                f"{'Power Output (kW)':<20}{power:>18.2f}{closest['power_output']:>25.2f}\n"
                f"{'Efficiency (%)':<20}{predicted_eff*100:>18.2f}{closest['efficiency_electric']:>25.2f}\n"
                f"{'Diesel Flow (kg/h)':<20}{mass_flows['diesel_mass_flow_kg_h']:>18.2f}{closest['diesel_mass_flow']:>25.2f}\n"
                f"{'CH₄ Flow (kg/h)':<20}{mass_flows['ch4_mass_flow_kg_h']:>18.2f}{closest['ch4_mass_flow_calc']:>25.2f}\n"
                f"{'Exhaust Temp (°C)':<20}{predicted_temp:>18.2f}{real_temp:>25.2f}\n"
                f"{'Current (A)':<20}{current:>18.2f}{real_current:>25.2f}\n"
                f"{'Frequency (Hz)':<20}{frequency:>18.2f}{real_frequency:>25.2f}"
            )
            output_label.config(text=output_text)

            fig.clf()
            df_sorted = df.sort_values(by='power_output')

            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            # Plot 1: Efficiency
            ax1.scatter(df_sorted["power_output"], df_sorted["efficiency_electric"],
                        label="Measured", color="lightgray", s=25)
            ax1.scatter(power, predicted_eff * 100, color="blue", marker='X', s=100, label="Predicted")
            ax1.set_title("Efficiency vs Power")
            ax1.set_xlabel("Power [kW]")
            ax1.set_ylabel("Efficiency [%]")
            ax1.set_xlim(0, 15)
            ax1.set_ylim(0, 25)
            ax1.grid(True)
            ax1.legend()

            # Plot 2: Mass Flows
            ax2.scatter(df_sorted["power_output"], df_sorted["diesel_mass_flow"],
                        label="Diesel Measured", color="lightgray", s=25)
            ax2.scatter(df_sorted["power_output"], df_sorted["ch4_mass_flow_calc"],
                        label="CH₄ Measured", color="darkgray", s=25)
            ax2.scatter(power, mass_flows["diesel_mass_flow_kg_h"],
                        color="saddlebrown", marker='X', s=100, label="Diesel Predicted")
            ax2.scatter(power, mass_flows["ch4_mass_flow_kg_h"],
                        color="darkgreen", marker='X', s=100, label="CH₄ Predicted")
            ax2.annotate(f"DES: {des_percent:.1f}%", (power, mass_flows["diesel_mass_flow_kg_h"]),
                         textcoords="offset points", xytext=(5, -15), fontsize=9)
            ax2.set_title("Mass Flows vs Power")
            ax2.set_xlabel("Power [kW]")
            ax2.set_ylabel("Mass Flow [kg/h]")
            ax2.set_xlim(0, 15)
            ax2.set_ylim(0, 10)
            ax2.grid(True)
            ax2.legend()

            # Plot 3: Exhaust Temp
            ax3.scatter(df_sorted["power_output"], df_sorted["exhaust_temp"],
                        label="Measured", color="lightgray", s=25)
            ax3.scatter(power, predicted_temp, color="darkorange", marker='X', s=100, label="Predicted")
            ax3.set_title("Exhaust Temp vs Power")
            ax3.set_xlabel("Power [kW]")
            ax3.set_ylabel("Exhaust Temp [°C]")
            ax3.set_xlim(0, 15)
            ax3.set_ylim(df["exhaust_temp"].min() - 10, df["exhaust_temp"].max() + 10)
            ax3.grid(True)
            ax3.legend()

            canvas.draw()

        except Exception as e:
            output_label.config(text=f"⚠️ Error: {e}")

    # ==== GUI SETUP ====
    window = tk.Tk()
    window.title("Digital Twin – Interactive Dashboard")
    window.geometry("1500x980")

    font_large = ("Arial", 13)
    font_mono = ("Courier New", 11)

    tk.Label(window, text="Diesel Energy Share (DES %)", font=font_large).pack()
    des_entry = tk.Entry(window, font=font_large)
    des_entry.insert(0, "15.0")
    des_entry.pack()

    tk.Label(window, text="Power Output (kW)", font=font_large).pack()
    power_entry = tk.Entry(window, font=font_large)
    power_entry.insert(0, "10.0")
    power_entry.pack()

    tk.Button(window, text="Calculate & Update", command=run_calculations, font=font_large).pack(pady=10)

    output_label = tk.Label(window, text="", justify="left", anchor="w", font=font_mono)
    output_label.pack(padx=15, pady=10)

    fig = plt.Figure(figsize=(18, 5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack()

    tk.Button(window, text="Exit", command=window.destroy, font=font_large).pack(pady=10)

    window.mainloop()