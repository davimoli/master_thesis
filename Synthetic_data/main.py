# M. David Synthetic data generation March 2025
import pybamm
import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re
import pandas as pd

print(f"PyBaMM version: {pybamm.__version__}")
start_time_total = time.perf_counter()

# ---------------- FUNCTION ------------------------
def lambda_pos(T):
    return 2.063e-5 * T**2 - 0.01127 * T + 2.331

def lambda_neg(T):
    return -2.61e-4 * T**2 + 0.1726 * T - 24.49

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

# -------------------- MODEL ---------------------
model_partially_reversible = pb.lithium_ion.DFN(options={
    "SEI": "interstitial-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "lithium plating porosity change": "true",  # alias for "SEI porosity change"
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
    "thermal": "lumped"
})

var_pts = {
    "x_n": 5,
    "x_s": 5,
    "x_p": 30,
    "r_n": 30,
    "r_p": 30,
}

# -------------------------- Experiment --------------------------
cycle_number = 250
experiment_298 = pybamm.Experiment(
    [
        "Hold at 4.4 V until C/100 (5 minute period)",
        "Rest for 4 hours (5 minute period)",
        "Discharge at 0.1C until 2.5 V (5 minute period)",
        "Charge at 0.3C until 4.4 V (5 minute period)",
        "Hold at 4.4 V until C/100 (5 minute period)",
    ] + [
        (
            "Discharge at 0.8C until 2.5 V",
            "Charge at 0.3C until 3.8 V (5 minute period)",
            "Charge at 0.3C until 4.0 V (5 minute period)",
            "Charge at 0.3C until 4.2 V (5 minute period)",
            "Charge at 0.3C until 4.4 V (5 minute period)",
            "Hold at 4.4 V until C/100 (5 minute period)",
        )
    ] * cycle_number + ["Discharge at 0.1C until 2.5 V (10 second period)"],
)

# Create a progress bar callback
class ProgressCallback(pybamm.callbacks.Callback):
    def __init__(self, total_steps, model_name):
        super().__init__()
        self.total_steps = total_steps
        self.model_name = model_name
        self.pbar = tqdm(total=total_steps, desc=f"Simulation Progress ({self.model_name})", unit="step")
        self.current_step = 0

    def on_experiment_start(self, logs):
        self.pbar.set_description(f"Simulation Progress ({self.model_name} - Starting)")

    def on_step_start(self, logs):
        pass

    def on_step_end(self, logs):
        self.current_step += 1
        self.pbar.update(1)
        self.pbar.set_description(f"Simulation Progress ({self.model_name} - Step {self.current_step}/{self.total_steps})")

    def on_experiment_end(self, logs):
        self.pbar.set_description(f"Simulation Progress ({self.model_name} - Finished)")
        self.pbar.close()

    def on_experiment_error(self, logs):
        self.pbar.set_description(f"Simulation Progress ({self.model_name} - Error)")
        self.pbar.close()

total_steps = 5 + (6 * cycle_number) + 1
solver = pybamm.IDAKLUSolver()

callback_pr = ProgressCallback(total_steps, "Irreversible")
sim_298_pr = pb.Simulation(
    model=model_partially_reversible,
    parameter_values=parameter_values_298,
    experiment=experiment_298,
    solver=solver,
    var_pts=var_pts
)

# Measure the time for the simulation
start_time_simulation = time.perf_counter()
try:
    solution_298_pr = sim_298_pr.solve(callbacks=[callback_pr])
except Exception as e:
    print(f"Simulation failed for Irreversible model with error: {e}")
    solution_298_pr = sim_298_pr.solution  # Use partial solution if available

end_time_simulation = time.perf_counter()
simulation_time = end_time_simulation - start_time_simulation
print(f"\nSimulation took {simulation_time:.2f} seconds (or {simulation_time/60:.2f} min) to complete.")

# -------------------------- Results --------------------------
# Extract variables
try:
    time_298_pr = solution_298_pr["Time [s]"].entries
    current_298_pr = solution_298_pr["Current [A]"].entries
    voltage_298_pr = solution_298_pr["Terminal voltage [V]"].entries
    temperature_298_pr = solution_298_pr["Volume-averaged cell temperature [K]"].entries
    capacity_lost_side_reactions_298_pr = solution_298_pr["Total capacity lost to side reactions [A.h]"].entries
    discharge_capacity_298_pr = solution_298_pr["Discharge capacity [A.h]"].entries  # Added discharge capacity
except KeyError as e:
    print(f"Error extracting variable: {e}")
    print("Available variables in Irreversible solution:")
    print(list(solution_298_pr.variables.keys()))
    raise

time_298_pr_hours = time_298_pr / 3600
temperature_298_pr_celsius = temperature_298_pr - 273.15
nominal_capacity = parameter_values_298["Nominal cell capacity [A.h]"]
capacity_lost_side_reactions_percent = (capacity_lost_side_reactions_298_pr / nominal_capacity) * 100

# Save to CSV instead of .pkl
results_dict = {
    "Time_s": time_298_pr,
    "Time_hours": time_298_pr_hours,
    "Current_A": current_298_pr,
    "Terminal_Voltage_V": voltage_298_pr,
    "Temperature_C": temperature_298_pr_celsius,
    "Capacity_Lost_Percent": capacity_lost_side_reactions_percent,
    "Discharge_Capacity_Ah": discharge_capacity_298_pr  # Added discharge capacity
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv("0_8C_298K.csv", index=False)
print("Simulation results saved to '0_8C_298K.csv'")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(len(time_298_pr)), time_298_pr, 'b-', label="Time [s]")
plt.xlabel("Sample Index")
plt.ylabel("Time [s]")
plt.title("Time Vector Progression")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

# Plot Current vs Time
plt.subplot(2, 2, 1)
plt.plot(time_298_pr_hours, current_298_pr, label="Current [A]", color="blue")
plt.xlabel("Time [hours]")
plt.ylabel("Current [A]")
plt.title("Current vs Time")
plt.grid(True)
plt.legend()

# Plot Voltage vs Time
plt.subplot(2, 2, 2)
plt.plot(time_298_pr_hours, voltage_298_pr, label="Terminal Voltage [V]", color="orange")
plt.xlabel("Time [hours]")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs Time")
plt.grid(True)
plt.legend()

# Plot Surface Temperature vs Time (in Celsius)
plt.subplot(2, 2, 3)
plt.plot(time_298_pr_hours, temperature_298_pr_celsius, label="Temperature [°C]", color="green")
plt.xlabel("Time [hours]")
plt.ylabel("Temperature [°C]")
plt.title("Average Temperature vs Time")
plt.grid(True)
plt.legend()

# Plot Capacity Lost to Side Reactions vs Time (in Percentage)
plt.subplot(2, 2, 4)
plt.plot(time_298_pr_hours, capacity_lost_side_reactions_percent, label="Capacity Lost [%]", color="red")
plt.xlabel("Time [hours]")
plt.ylabel("Capacity Lost [%]")
plt.title("Capacity Loss due to Side Reactions vs Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Record the end time for the entire script and calculate total time
end_time_total = time.perf_counter()
total_time = end_time_total - start_time_total
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")