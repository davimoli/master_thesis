# M. David Synthetic data generation March 2025
import pybamm
import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time for measuring execution time
import re  # Import re for regular expressions

# Record the start time for the entire script
start_time_total = time.perf_counter()

# Print PyBaMM version
print(f"PyBaMM version: {pybamm.__version__}")


# Function to update parameters
def update_parameters(param_values, new_values, verbose=True):
    """
    Update parameter values in a PyBaMM ParameterValues object.

    Args:
        param_values (pybamm.ParameterValues): The ParameterValues object to update.
        new_values (dict): Dictionary of parameter names (keys) and new values (values).
        verbose (bool): If True, print the updated values for verification.

    Returns:
        pybamm.ParameterValues: The updated ParameterValues object (though it’s modified in-place).
    """
    for param_name, new_value in new_values.items():
        if param_name in param_values:
            old_value = param_values[param_name]
            param_values[param_name] = new_value
            if verbose:
                print(f"Updated '{param_name}': {old_value} -> {new_value}")
        else:
            param_values[param_name] = new_value
            if verbose:
                print(f"Added new parameter '{param_name}': {new_value}")
    return param_values


# ------------- Initialization --------------------------

# Load parameter values for Experiment 1 (298.15 K)
parameter_values_ORegan = pybamm.ParameterValues("ORegan2022")

# Load OKane2022 parameters to use for missing parameters
parameter_values_okane = pybamm.ParameterValues("OKane2022")

# Combine the parameter sets
combined_dict = {**parameter_values_okane, **parameter_values_ORegan}
combined_parameters = pybamm.ParameterValues(combined_dict)
parameter_values_298 = combined_parameters

'''
# Define custom parameter updates
custom_parameters = {
    #SEI Param
    "SEI lithium interstitial diffusivity [m2.s-1]": 9.81e-19,  # Update capacity to 5.5 A.h
    "SEI partial molar volume [m3.mol-1]": 5.22e-5,  # Update initial temperature to 300 K
    "SEI growth activation energy [J.mol-1]": 5e3,   # Update thickness to 0.00012 m

    #Lithium plating
    "Dead lithium decay constant [s-1]": 1e-7,
    "Lithium plating kinetic rate constant [m.s-1]": 1e-10,

    #LAM
    "Positive electrode LAM constant proportional term [s-1]": 2.98e-18,
    "Negative electrode LAM constant proportional term [s-1]": 2.84e-9,
    "Negative electrode cracking rate": 5.29e-25,

    #DFN model

}

# Apply the updates
parameter_values_298 = update_parameters(parameter_values_298, custom_parameters)
parameter_values_298.update({'Negative electrode diffusivity activation energy [J.mol-1]': 2e4}, check_already_exists=False)
'''

# Define the model for partially reversible lithium plating
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

# Define spatial discretization points
var_pts = {
    "x_n": 5,  # negative electrode
    "x_s": 5,  # separator
    "x_p": 30,  # positive electrode
    "r_n": 30,  # negative particle
    "r_p": 30,  # positive particle
}

#-------------------------- Experiment --------------------------

# Define the experiment: Cycling at 298.15 K
cycle_number = 10
experiment_298 = pybamm.Experiment(
[
        "Hold at 4.4 V until C/100 (5 minute period)",
        "Rest for 4 hours (5 minute period)",
        "Discharge at 0.1C until 2.5 V (5 minute period)",  # initial capacity check
        "Charge at 0.3C until 4.4 V (5 minute period)",
        "Hold at 4.4 V until C/100 (5 minute period)",
    ]
    + [
        (
            "Discharge at 1C until 2.5 V",  # ageing cycles 2C discharge
            "Charge at 0.3C until 4.4 V (5 minute period)",
            "Hold at 4.4 V until C/100 (5 minute period)",
        )
    ]
    * cycle_number
    + ["Discharge at 0.1C until 2.5 V (5 minute period)"],  # final capacity check
   # termination="80% capacity",
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

# Estimate the number of steps
total_steps = 5 + (3 * cycle_number) + 1

# Create and solve simulation
solver = pybamm.IDAKLUSolver()

# Simulation for Partially Reversible model
callback_pr = ProgressCallback(total_steps, "Partially Reversible")
sim_298_pr = pb.Simulation(model_partially_reversible, parameter_values=parameter_values_298, experiment=experiment_298, solver=solver, var_pts=var_pts)

# Measure the time for the simulation
start_time_simulation = time.perf_counter()
try:
    solution_298_pr = sim_298_pr.solve(callbacks=[callback_pr])
except Exception as e:
    print(f"Simulation failed for Partially Reversible model with error: {e}")
    raise
end_time_simulation = time.perf_counter()
simulation_time = end_time_simulation - start_time_simulation
print(f"\nSimulation took {simulation_time:.2f} seconds (or {simulation_time/60:.2f} min) to complete.")

#-------------------------- Results --------------------------

# Extract variables
try:
    time_298_pr = solution_298_pr["Time [s]"].entries
    current_298_pr = solution_298_pr["Current [A]"].entries
    voltage_298_pr = solution_298_pr["Terminal voltage [V]"].entries
    temperature_298_pr = solution_298_pr["Volume-averaged cell temperature [K]"].entries
    capacity_lost_side_reactions_298_pr = solution_298_pr["Total capacity lost to side reactions [A.h]"].entries
except KeyError as e:
    print(f"Error extracting variable: {e}")
    print("Available variables in Partially Reversible solution:")
    print(list(solution_298_pr.variables.keys()))
    raise

# Convert time to hours
time_298_pr_hours = time_298_pr / 3600

# Convert temperature from Kelvin to Celsius
temperature_298_pr_celsius = temperature_298_pr - 273.15

# Get nominal capacity from parameters and calculate capacity loss in percentage
nominal_capacity = parameter_values_298["Nominal cell capacity [A.h]"]
capacity_lost_side_reactions_percent = (capacity_lost_side_reactions_298_pr / nominal_capacity) * 100

# Plot
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
plt.plot(time_298_pr_hours, capacity_lost_side_reactions_percent, label="Capacity Lost [%] ", color="red")
plt.xlabel("Time [hours]")
plt.ylabel("Capacity Lost [%]")
plt.title("Capacity Loss due to Side Reactions vs Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

solution_298_pr.save("1C_298K.pkl")
print("Simulation solution saved to 'partially_reversible_solution.pkl'")
# Record the end time for the entire script and calculate total time
end_time_total = time.perf_counter()
total_time = end_time_total - start_time_total
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

