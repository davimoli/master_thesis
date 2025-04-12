import pybamm as pb
import pandas as pd
import pybamm

'''
Définition de mes variables 
'''

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

"""------------------------------------------------------------------------------------------------





------------------------------------- REAL CODE -------------------------------








------------------------------------------------------------------------------------------------"""

my_sim = pb.load('/Volumes/SANDISK128/0_8C_298K.pkl')

time = my_sim["Time [s]"].entries
current = my_sim["Current [A]"].entries
voltage = my_sim["Terminal voltage [V]"].entries
temperature = my_sim["Volume-averaged cell temperature [K]"].entries
capacity_lost_side_reactions = my_sim["Total capacity lost to side reactions [A.h]"].entries
discharge_capacity = my_sim["Discharge capacity [A.h]"].entries

time_hours = time / 3600
temperature_celsius = temperature - 273.15
nominal_capacity = 5
capacity_lost_side_reactions_percent = (capacity_lost_side_reactions / nominal_capacity) * 100

# Save to CSV instead of .pkl
results_dict = {
    "Time_s": time,
    "Time_hours": time_hours,
    "Current_A": current,
    "Terminal_Voltage_V": voltage,
    "Temperature_C": temperature_celsius,
    "Capacity_Lost_Percent": capacity_lost_side_reactions_percent,
    "Discharge_Capacity_Ah": discharge_capacity  # Added discharge capacity
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv("0_8C_298K.csv", index=False)
print("Simulation results saved to '1C_298K.csv'")