# M. David Synthetic data generation March 2025
import pybamm
import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re

print(f"PyBaMM version: {pybamm.__version__}")
start_time_total = time.perf_counter()

# ---------------- FUNCTION ------------------------
def lambda_pos(T):
    return 2.063e-5 * T ** 2 - 0.01127 * T + 2.331

def lambda_neg(T):
    return -2.61e-4 * T ** 2 + 0.1726 * T - 24.49

def update_parameters(param_values, new_values, verbose=True):
    for param_name, new_value in new_values.items():
        if param_name in param_values:
            old_value = param_values[param_name]
            param_values[param_name] = new_value
            if verbose:
                print(f"Updated '{param_name}': {old_value} -> {new_value}")
        else:
            param_values.update({param_name: new_value}, check_already_exists=False)
            if verbose:
                print(f"Added new parameter '{param_name}': {new_value}")
    return param_values

# ------------- Initialization --------------------------
eps = 1e-2
parameter_values_ORegan = pybamm.ParameterValues("ORegan2022")
parameter_values_okane = pybamm.ParameterValues("OKane2022")
combined_dict = {**parameter_values_okane, **parameter_values_ORegan}
combined_parameters = pybamm.ParameterValues(combined_dict)
parameter_values_298 = combined_parameters

parameter_values_298["Positive electrode thermal conductivity [W.m-1.K-1]"] = pybamm.FunctionParameter(
    "Positive electrode thermal conductivity [W.m-1.K-1]",
    {"Temperature [K]": pybamm.InputParameter("Temperature [K]")},
    lambda_pos
)
parameter_values_298["Negative electrode thermal conductivity [W.m-1.K-1]"] = pybamm.FunctionParameter(
    "Negative electrode thermal conductivity [W.m-1.K-1]",
    {"Temperature [K]": pybamm.InputParameter("Temperature [K]")},
    lambda_neg
)

custom_parameters = {
    "Ratio of lithium moles to SEI moles": 2,
    "Lithium interstitial reference concentration [mol.m-3]": 15,
    "SEI resistivity [Ohm.m]": 2e5,
    "Inner SEI proportion": 0.5,
    "Initial inner SEI thickness [m]": 1.23625e-08,
    "Initial outer SEI thickness [m]": 1.23625e-08,
    "Initial concentration in electrolyte [mol.m-3]": 1000,
    "Initial EC concentration in electrolyte [mol.m-3]": 4541,
    "EC partial molar volume [m3.mol-1]": 6.667e-5,
    "Lithium plating transfer coefficient": 0.65,
    "Initial plated lithium concentration [mol.m-3]": 0,
    "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
    "Positive electrode LAM constant exponential term": 2,
    "Negative electrode LAM constant exponential term": 2,
    "Positive electrode stress intensity factor correction": 1.12,
    "Negative electrode stress intensity factor correction": 1.12,
    "Positive electrode Paris' law exponent m_cr": 2.2,
    "Negative electrode Paris' law exponent m_cr": 2.2,
    "Positive electrode number of cracks per unit area [m-2]": 3.18e15,
    "Negative electrode number of cracks per unit area [m-2]": 3.18e15,
    "Positive electrode initial crack length [m]": 2e-08,
    "Negative electrode initial crack length [m]": 2e-08,
    "Positive electrode initial crack width [m]": 1.5e-08,
    "Negative electrode initial crack width [m]": 1.5e-08,
    "Positive electrode critical stress [Pa]": 375e6,
    "Negative electrode critical stress [Pa]": 60e6,
}
parameter_values_298 = update_parameters(parameter_values_298, custom_parameters)
parameter_values_298.update({"Lithium plating kinetic rate constant [m.s-1]": 5e-11})
parameter_values_298.update({"Negative electrode cracking rate": 5.29e-25})
parameter_values_298.update({"Positive electrode cracking rate": 5.29e-25})

# Variable related to the parameters
v_up_cutoff = parameter_values_298["Upper voltage cut-off [V]"]
v_low_cutoff = parameter_values_298["Lower voltage cut-off [V]"]
nominal_capacity = parameter_values_298["Nominal cell capacity [A.h]"]

model_partially_reversible = pb.lithium_ion.DFN(options={
    "SEI": "interstitial-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "lithium plating porosity change": "true",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
    "thermal": "lumped",
})

file_path = '/Users/david/PycharmProjects/memoire/Synthetic_data/1C_298K.pkl'
solution = pb.load(file_path)
print("ok")

# Extract solutions outputs
time = solution["Time [s]"].entries  # Keep in seconds
voltage = solution["Terminal voltage [V]"].entries
current = solution["Current [A]"].entries
capacity_loss_sr = solution["Total capacity lost to side reactions [A.h]"].entries
capacity_lost_side_reactions_percent = (capacity_loss_sr / nominal_capacity) * 100

soh = 1 - (capacity_lost_side_reactions_percent / 100)

c_100_threshold = nominal_capacity / 100

# ----------------------------------- Feature extraction -----------------------------------
v_cc_start_counting = 3.8
voltage_cv_threshold = 4.4 - 0.01
cccv_starts = []
cc_starts_41 = []
cv_starts = []
cv_ends = []
cc_durations = []
cv_durations = []
cc_end_slopes = []
soh_at_cycle_ends = []

slope_window = 5

# Feature extraction algorithm
for i in range(1, len(voltage)):
    if voltage[i - 1] < v_low_cutoff + eps and voltage[i] > v_low_cutoff + eps and current[i] < 0:
        cccv_starts.append(i)

for start_idx in cccv_starts:
    cc_start_41_idx = None
    for i in range(start_idx, len(voltage)):
        if voltage[i] >= v_cc_start_counting and current[i] < 0:
            # Check if current remains negative until CV voltage is reached
            valid_cc_start = True
            for j in range(i, len(voltage)):
                if voltage[j] >= voltage_cv_threshold - eps:
                    break  # Stop checking once CV voltage is reached
                if current[j] >= 0:  # Current becomes non-negative before CV
                    valid_cc_start = False
                    break
            if valid_cc_start:
                cc_start_41_idx = i
                break

    cv_start_idx = None
    for i in range(start_idx, len(voltage) - 1):
        if (voltage[i] >= voltage_cv_threshold and
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

    if (cc_start_41_idx and cv_start_idx and cv_end_idx and
            cc_start_41_idx < cv_start_idx < cv_end_idx):
        cc_starts_41.append(cc_start_41_idx)
        cv_starts.append(cv_start_idx)
        cv_ends.append(cv_end_idx)
        cc_duration = time[cv_start_idx] - time[cc_start_41_idx]
        cv_duration = time[cv_end_idx] - time[cv_start_idx]
        cc_durations.append(cc_duration)
        cv_durations.append(cv_duration)

        window_start = max(cc_start_41_idx, cv_start_idx - slope_window)
        window_end = cv_start_idx + 1
        time_window = time[window_start:window_end]
        voltage_window = voltage[window_start:window_end]
        slope = np.gradient(voltage_window, time_window)[-1]
        cc_end_slopes.append(slope)

        # Store SoH at the end of the cycle (CV end)
        soh_at_cycle_ends.append(soh[cv_end_idx])

# Print results
print(f"Number of CCCV cycles detected: {len(cccv_starts)}")
for i, (cc_dur, cv_dur, slope, soh_end) in enumerate(zip(cc_durations, cv_durations, cc_end_slopes, soh_at_cycle_ends), 1):
    print(f"Cycle {i}: CC duration (3.8 V to CV) = {cc_dur:.2f} seconds, "
          f"CV duration = {cv_dur:.2f} seconds, Voltage slope at CC end = {slope:.6f} V/s, "
          f"SoH at cycle end = {soh_end*100:.2f}%")

# Plot Voltage vs Time with CCCV, CC, and CV markers
time_hours = time / 3600  # Convert to hours for plotting
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
plt.title("Voltage vs Time with CCCV, CC, and CV Markers (1C Discharge, 20 Cycles)")
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

# Voltage subplot
ax1.plot(time_hours, voltage, label="Terminal Voltage [V]", color="red")
ax1.set_ylabel("Voltage [V]", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.grid(True)
ax1.legend(loc="upper right")

# Current subplot
ax2.plot(time_hours, current, label="Current [A]", color="blue")
ax2.set_xlabel("Time [hours]")
ax2.set_ylabel("Current [A]", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
ax2.grid(True)
ax2.legend(loc="upper right")

# Overall figure title
fig.suptitle("Voltage and Current vs Time")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
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