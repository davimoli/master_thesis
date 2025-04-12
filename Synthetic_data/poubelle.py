# M. David Synthetic Data Analysis March 2025
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

print("Starting feature extraction from CSV file...")
start_time_total = time.perf_counter()

# ------------- Load Data from CSV File --------------------------
file_path = '/Users/moliconstruct/PycharmProjects/memoire_v2/.venv/0_8C_298K.csv'
raw_data_df = pd.read_csv(file_path)

# Verify the columns in the CSV file
print("Columns in the CSV file:", raw_data_df.columns.tolist())

# Extract raw simulation data from the CSV file
time = raw_data_df["Time_s"].values  # Time in seconds
time_hours = raw_data_df["Time_hours"].values  # Time in hours
current = raw_data_df["Current_A"].values  # Current in Amperes
voltage = raw_data_df["Terminal_Voltage_V"].values  # Terminal voltage in Volts
temperature_celsius = raw_data_df["Temperature_C"].values  # Temperature in Celsius
temperature = temperature_celsius + 273.15  # Convert to Kelvin
capacity_lost_side_reactions_percent = raw_data_df["Capacity_Lost_Percent"].values  # Capacity loss in percent
discharge_capacity = raw_data_df["Discharge_Capacity_Ah"].values  # Discharge capacity in A.h

# Nominal capacity (hardcoded as in the original code, since it's not in the CSV)
nominal_capacity = 5.0  # A.h (assumed value; adjust based on your simulation settings)

# Compute SoC and SoH
soc = 1 - (discharge_capacity / nominal_capacity)
soh = 1 - (capacity_lost_side_reactions_percent / 100)

# Threshold for CV end detection
c_100_threshold = nominal_capacity / 100

# ----------------------------------- Feature Extraction -----------------------------------
eps = 1e-2
v_cc_start_counting = 3.8
voltage_cv_threshold = 4.4 - 0.01
cccv_starts = []
cc_starts_41 = []
cv_starts = []
cv_ends = []
cc_durations = []
cv_durations = []
voltage_slopes_at_38V = []
avg_temps_during_cccv = []
avg_voltages_during_cc = []
time_between_cycles = []
soh_at_cycle_ends = []
time_at_cycle_ends = []
slope_window = 10  # Window size for slope calculation

# Feature extraction algorithm
counter_cycles = 1
v_low_cutoff = 2.5  # Assumed value; adjust based on your simulation settings
for i in range(1, len(voltage)):
    if voltage[i - 1] < v_low_cutoff + eps and voltage[i] > v_low_cutoff + eps and current[i] < 0:
        cccv_starts.append(i)
        print(f"Time of Cycle {counter_cycles} = {time[i]/3600:.2f} hours")
        counter_cycles += 1
        time_at_cycle_ends.append(time[i])

# Calculate time between cycles
for i in range(len(cccv_starts) - 1):
    time_diff = time[cccv_starts[i + 1]] - time[cccv_starts[i]]
    time_between_cycles.append(time_diff)

time_between_cycles.append(np.nan)  # For the last cycle

for cycle_num, start_idx in enumerate(cccv_starts, 1):
    cc_start_41_idx = None
    for i in range(start_idx, len(voltage)):
        if voltage[i] >= v_cc_start_counting and current[i] < 0:
            valid_cc_start = True
            for j in range(i, len(voltage)):
                if voltage[j] >= voltage_cv_threshold - eps:
                    break
                if current[j] >= 0:
                    valid_cc_start = False
                    break
            if valid_cc_start:
                cc_start_41_idx = i
                break

    cv_start_idx = None
    for i in range(start_idx, len(voltage) - 1):
        if (voltage[i] >= voltage_cv_threshold - eps and
                abs(voltage[i + 1] - voltage[i]) < 0.001 and
                current[i] < 0):
            cv_start_idx = i
            break

    cv_end_idx = None
    for i in range(cv_start_idx or start_idx, len(voltage) - 1):
        if (abs(current[i]) <= c_100_threshold or
                (current[i] > 0 and voltage[i] < 4.0)):
            cv_end_idx = i
            break

    if not (cc_start_41_idx and cv_start_idx and cv_end_idx and
            cc_start_41_idx < cv_start_idx < cv_end_idx):
        print(f"Cycle {cycle_num} failed condition: "
              f"CC Start = {cc_start_41_idx}, CV Start = {cv_start_idx}, CV End = {cv_end_idx}")
    else:
        cc_starts_41.append(cc_start_41_idx)
        cv_starts.append(cv_start_idx)
        cv_ends.append(cv_end_idx)
        cc_duration = time[cv_start_idx] - time[cc_start_41_idx]
        cv_duration = time[cv_end_idx] - time[cv_start_idx]
        cc_durations.append(cc_duration)
        cv_durations.append(cv_duration)

        # Calculate slope at 3.8 V
        window_start = max(0, cc_start_41_idx - slope_window // 2)
        window_end = min(len(voltage), cc_start_41_idx + slope_window // 2 + 1)
        time_window = time[window_start:window_end]
        voltage_window = voltage[window_start:window_end]
        if len(time_window) >= 2:
            coefficients = np.polyfit(time_window, voltage_window, 1)
            slope = coefficients[0]
        else:
            slope = 0
            print(f"  Warning: Not enough points to calculate slope for cycle {cycle_num + 1}")
        voltage_slopes_at_38V.append(slope)

        # Calculate average temperature during CCCV
        temp_cccv = temperature_celsius[start_idx:cv_end_idx + 1]
        avg_temp = np.mean(temp_cccv)
        avg_temps_during_cccv.append(avg_temp)

        # Calculate average voltage during CC phase (3.8 V to 4.4 V)
        voltage_cc = voltage[cc_start_41_idx:cv_start_idx + 1]
        avg_voltage_cc = np.mean(voltage_cc)
        avg_voltages_during_cc.append(avg_voltage_cc)

        soh_at_cycle_ends.append(soh[cv_end_idx])

# Print results
print(f"Number of CCCV cycles detected: {len(cccv_starts)}")
for i, (cc_dur, cv_dur, slope, avg_temp, avg_voltage, time_diff, soh_end) in enumerate(
        zip(cc_durations, cv_durations, voltage_slopes_at_38V, avg_temps_during_cccv, avg_voltages_during_cc, time_between_cycles, soh_at_cycle_ends), 1):
    time_diff_str = f"{time_diff:.2f}" if not np.isnan(time_diff) else "N/A"
    print(f"Cycle {i}: CC duration (3.8 V to CV) = {cc_dur:.2f} s, "
          f"CV duration = {cv_dur:.2f} s, Voltage slope at 3.8 V = {slope:.6f} V/s, "
          f"Avg Temp during CCCV = {avg_temp:.2f} °C, Avg Voltage during CC = {avg_voltage:.2f} V, "
          f"Time between cycles = {time_diff_str} s, SoH at cycle end = {soh_end*100:.2f}%")

# Store features in a CSV file
features_dict = {
    "Cycle": list(range(1, len(cc_durations) + 1)),
    "CC_Duration_seconds": cc_durations,
    "CV_Duration_seconds": cv_durations,
    "Voltage_Slope_at_38V": voltage_slopes_at_38V,
    "Avg_Temperature_during_CCCV_Celsius": avg_temps_during_cccv,
    "Avg_Voltage_during_CC": avg_voltages_during_cc,
    "Time_Between_Cycles_seconds": time_between_cycles,
    "SoH_at_Cycle_End": [soh * 100 for soh in soh_at_cycle_ends]
}
features_df = pd.DataFrame(features_dict)
features_df.to_csv("battery_features_0_8C_298K.csv", index=False)
print("Features saved to 'battery_features_0_8C_298K.csv'")

# Plot Voltage vs Time with CCCV, CC, and CV markers
plt.figure(figsize=(12, 6))
plt.plot(time_hours, voltage, label="Terminal Voltage [V]", color="blue")
for idx, start_idx in enumerate(cccv_starts):
    plt.axvline(x=time_hours[start_idx], color="red", linestyle="--", alpha=0.5,
                label=f"CCCV Cycle Start {idx + 1}" if idx == 0 else None)
for idx, (cc_idx, cv_start_idx, cv_end_idx) in enumerate(zip(cc_starts_41, cv_starts, cv_ends)):
    plt.axvline(x=time_hours[cc_idx], color="green", linestyle=":", alpha=0.7,
                label="CC Start (3.8 V)" if idx == 0 else None)
    plt.axvline(x=time_hours[cv_start_idx], color="purple", linestyle="-.", alpha=0.7,
                label="CV Start (4.4 V)" if idx == 0 else None)
    plt.axvline(x=time_hours[cv_end_idx], color="orange", linestyle="-", alpha=0.7,
                label="CV End" if idx == 0 else None)
plt.xlabel("Time [hours]")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs Time with CCCV, CC, and CV Markers (0_8C Discharge)")
plt.grid(True)
plt.legend()
plt.show()

# Plot Capacity Loss vs Time
plt.figure(figsize=(12, 6))
plt.plot(time_hours, capacity_lost_side_reactions_percent, label="Capacity Loss [%]", color="red")
plt.xlabel("Time [hours]")
plt.ylabel("Capacity Loss [%]")
plt.title("Capacity Loss vs Time")
plt.grid(True)
plt.legend()
plt.show()

# Plot Voltage and Current vs Time in separate subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
ax1.plot(time_hours, voltage, label="Terminal Voltage [V]", color="blue")
for idx, start_idx in enumerate(cccv_starts):
    ax1.axvline(x=time_hours[start_idx], color="red", linestyle="--", alpha=0.5,
                label=f"CCCV Cycle Start {idx + 1}" if idx == 0 else None)
for idx, (cc_idx, cv_start_idx, cv_end_idx) in enumerate(zip(cc_starts_41, cv_starts, cv_ends)):
    ax1.axvline(x=time_hours[cc_idx], color="green", linestyle=":", alpha=0.7,
                label="CC Start (3.8 V)" if idx == 0 else None)
    ax1.axvline(x=time_hours[cv_start_idx], color="purple", linestyle="-.", alpha=0.7,
                label="CV Start (4.4 V)" if idx == 0 else None)
    ax1.axvline(x=time_hours[cv_end_idx], color="orange", linestyle="-", alpha=0.7,
                label="CV End" if idx == 0 else None)
ax1.set_ylabel("Voltage [V]", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True)
ax1.legend(loc="upper right")

ax2.plot(time_hours, current, label="Current [A]", color="blue")
ax2.set_xlabel("Time [hours]")
ax2.set_ylabel("Current [A]", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
ax2.grid(True)
ax2.legend(loc="upper right")

fig.suptitle("Voltage and Current vs Time")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plot SoH vs Time
plt.figure(figsize=(12, 6))
plt.plot(time_hours, soh * 100, label="State of Health [%]", color="green")
plt.xlabel("Time [hours]")
plt.ylabel("SoH [%]")
plt.title("State of Health vs Time")
plt.grid(True)
plt.legend()
plt.show()

# Print verification info
print(f"Total simulation time: {time[-1]:.2f} seconds ({time_hours[-1]:.2f} hours)")
print(f"Voltage range: {min(voltage):.2f} V to {max(voltage):.2f} V")
print(f"SoH range: {min(soh)*100:.2f}% to {max(soh)*100:.2f}%")