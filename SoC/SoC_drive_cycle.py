import pybamm
import pybamm as pb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from feature_extraction_drive_cycle import extract_battery_features


# ----------------------- PARAMETRISATION ----------------------

parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({"SEI kinetic rate constant [m.s-1]": 5e-15})  # To fit Okane's Deg
param_artemis = parameter_values.copy()
# ----------------------- Models ----------------------
model = pybamm.lithium_ion.DFN(options={"SEI": "ec reaction limited",
                                        "thermal": "lumped"})
model_spm = pybamm.lithium_ion.SPM()


print(model.variables["Voltage [V]"])
# ----------------------- EXPERIMENTS ----------------------

C = parameter_values['Nominal cell capacity [A.h]']  # To compute the C-rate

### DRIVE CYCLE ###
data_loader = pybamm.DataLoader()
drive_cycle = pd.read_csv(
        '/Users/david/PycharmProjects/memoire/SoC/ArtemisM_scaled.csv', comment="#", header=None
).to_numpy()
# Generate a time array (assuming 1-second intervals)
time = np.arange(0, len(drive_cycle), 1)  # [0, 1, 2, ..., 1771]
# Combine time and data into a 2-column array
drive_cycle_with_time = np.column_stack((time, drive_cycle))
# Create the interpolant
current_interpolant = pybamm.Interpolant(
    drive_cycle_with_time[:, 0],  # Time column
    drive_cycle_with_time[:, 1],  # Current (or other variable) column
    pybamm.t
)
# set drive cycle
print(len(drive_cycle)/3600)

### Caracterization experiment ###
capacity_check_exp = pybamm.Experiment(
    [
        (
            "Charge at 0.3C until 4.2V",
            "Hold at 4.2V until C/100",
            "Discharge at 1C for 0.5 hours",
            "Rest for 1 hour",

        )
    ]
)

### Artemis Experiment ###
artemis_exp = pb.Experiment([
    (
        pybamm.step.current(current_interpolant, duration=f"{len(drive_cycle)} seconds")
    )
])

### Custom experiment, after aging: AC current ###
def ac_current_step(t, C_rate):
    # return 0.5*C*np.sin(2 * np.pi * t / 3600 * 4)
    return C_rate * C


def neg_ac_current_step(t, C_rate):
    # return 0.5*C*np.sin(2 * np.pi * t / 3600 * 4)
    return -C_rate * C


ac_experiment = pybamm.Experiment(
    [
        (
        pybamm.step.current(ac_current_step(t= pb.t, C_rate=0.5), duration="0.1 hours"),
        pybamm.step.current(neg_ac_current_step(t= pb.t, C_rate=0.5), duration="0.1 hours")
        )
     * 2

    +
        (
        pybamm.step.current(ac_current_step(t= pb.t, C_rate=1), duration="0.05 hours"),
        pybamm.step.current(neg_ac_current_step(t= pb.t, C_rate=1), duration="0.05 hours")
        )
     * 2

    +
        (
            pybamm.step.current(ac_current_step(t=pb.t, C_rate=2), duration="0.025 hours"),
            pybamm.step.current(neg_ac_current_step(t=pb.t, C_rate=2), duration="0.025 hours"),

        )
     * 2]
)

# ----------------------- MY Simulations ----------------------
sim_capacity_check = pb.Simulation(model, parameter_values=parameter_values, experiment=capacity_check_exp)
sim_artemis = pb.Simulation(model, parameter_values=parameter_values, experiment=ac_experiment)





# ----------------------- Solving the sims  ----------------------
M = 70 #Number of cycles
initial_soc = 0.5
sol = sim_capacity_check.solve(initial_soc=initial_soc)

for i in range(M):
    if i != 0:
        sol = sim_artemis.solve(starting_solution=sol)
        sol = sim_capacity_check.solve(starting_solution=sol)


print(f"Sol length:", len(sol.cycles))

sol.save_data(
    "outputs.pickle",
    ["Time [h]", "Current [A]", "Voltage [V]", "Average negative particle concentration [mol.m-3]"],
)
### Solving the SPM model ###


# ------------ Feature extractions --------------
extract_battery_features(sol=sol, v_low_range=4.1, csv_filename='dynamic_cycle_2.csv')

# ------------ Analyzing the results --------------
x_100 = 0.9106121196114528
x_0 = 0.026347301451412015

t = sol['Time [h]'].data
voltage = sol['Voltage [V]'].data
current = sol['Current [A]'].data
x = sol['Negative electrode stoichiometry'].data
soc = (x - x_0) / (x_100 - x_0)


### In terms of concentration ###
cs_max = sol["Maximum negative particle concentration [mol.m-3]"].data
cs_avg = sol["Average negative particle concentration [mol.m-3]"].data
cs_min = sol["Minimum negative particle concentration [mol.m-3]"].data
cs_max = 33133.0
cs_0 = x_0*cs_max
cs_100 = x_100*cs_max

soc_conc = cs_avg - cs_0
soc_conc = soc_conc/(cs_100-cs_0)



### Coulomb Counting approach ###
initial_cap = 5.123
cc = sol["Discharge capacity [A.h]"].data
soc_cc = initial_soc - cc/initial_cap


plt.figure(figsize=(5.5, 3.4))
plt.plot(t, current, label='Current', linewidth=2)

for i, cycle in enumerate(sol.cycles):
    t_start_cycle = cycle['Time [h]'].data
    plt.axvline(x=t_start_cycle[0], linestyle='--', color='red', linewidth=1)

plt.title('Cycling protocol for dynamic current profile with cycle indicator', fontsize=12)
plt.xlabel('Time [h]')
plt.ylabel('Current [A]')
plt.grid()
plt.legend()
plt.show()

### Comparison with SPM ###
cycle_to_analyse = 57 * 2
soh_at_57 = 0.8883707192106897     #57th charge
FNN_57 = 0.8983707192106897        #Prediction from FNN

cycle_56 = sol.cycles[cycle_to_analyse ]
cycle_57 = sol.cycles[cycle_to_analyse + 1 ]        #+1 for python starting at 0
cs_avg = cycle_57["Average negative particle concentration [mol.m-3]"].data
cs_0 = x_0*cs_max
cs_100 = x_100 * cs_max * soh_at_57

time_cycle = cycle_57['Time [h]'].data
time_cycle = time_cycle - time_cycle[0]
current_cycle = cycle_57['Current [A]'].data
soc_DFN_conc = (cs_avg - cs_0)/(cs_100-cs_0)

### Lets build the SPM ###
parameter_values_spm = parameter_values.copy()
parameter_values_spm["Current function [A]"] = pybamm.Interpolant(time_cycle*3600, current_cycle, pybamm.t)
parameter_values_spm["Maximum concentration in negative electrode [mol.m-3]"] = cs_max * soh_at_57
sim_spm = pb.Simulation(model=model_spm, parameter_values=parameter_values_spm)

sol_spm = sim_spm.solve(initial_soc=soc_DFN_conc[0])


time_spm = sol_spm['Time [h]'].data
voltage_spm = sol_spm['Voltage [V]'].data
current_spm = sol_spm['Current [A]'].data
cs_avg_spm = sol_spm["Average negative particle concentration [mol.m-3]"].data
cs_100 = x_100 * cs_max * FNN_57
soc_spm_conc = (cs_avg_spm - cs_0)/(cs_100-cs_0) - (0.4876355 - 0.449722)

time_298_pr = sol["Time [s]"].entries
current_298_pr = sol["Current [A]"].entries
voltage_298_pr = sol["Terminal voltage [V]"].entries
temperature_298_pr = sol["Volume-averaged cell temperature [K]"].entries
capacity_lost_side_reactions_298_pr = sol["Total capacity lost to side reactions [A.h]"].entries
discharge_capacity_298_pr = sol["Discharge capacity [A.h]"].entries
neg_electrode_soc = sol["Negative electrode stoichiometry"].entries
lam = sol["Loss of active material in negative electrode [%]"].entries

# Save extracted variables to a CSV file
data_dict = {
    "Time [s]": time_298_pr,
    "Current [A]": current_298_pr,
    "Terminal voltage [V]": voltage_298_pr,
    "Volume-averaged cell temperature [K]": temperature_298_pr,
    "Total capacity lost to side reactions [A.h]": capacity_lost_side_reactions_298_pr,
    "Discharge capacity [A.h]": discharge_capacity_298_pr,
    "Negative electrode stoichiometry": neg_electrode_soc,
    "Loss of active material in negative electrode [%]": lam
}
df = pd.DataFrame(data_dict)
df.to_csv("dynamic_cycle_OKane_advices_x20r60_29_06.csv", index=False)

### classik SPM results ###
parameter_values_spm_klassik = parameter_values_spm.copy()
parameter_values_spm_klassik["Maximum concentration in negative electrode [mol.m-3]"] = cs_max
soc_spm_klassik_conc = (cs_avg_spm - cs_0)/(cs_100/FNN_57-cs_0) + (0.449722 - 0.43660)

plt.figure(figsize=(5.5, 3.4))
plt.plot(time_cycle, soc_DFN_conc*100, linewidth=2, color='blue', label='SoC')
plt.plot(time_spm, soc_spm_klassik_conc*100, linestyle='--', color='green', linewidth=2, label='SôC SPM')
plt.plot(time_spm, soc_spm_conc*100, linestyle='--', color='red', linewidth=2, label='SôC enhanced SPM')
plt.title('SoC estimation at Cycle 57', fontsize=14)
plt.legend(loc='upper right')
plt.xlabel('Time [h]')
plt.grid()
plt.plot()
plt.show()

plt.figure(figsize=(5.5, 3.4))
plt.plot(time_spm, current_spm, linewidth=2, label='Current [A]')
plt.xlabel('Time [h]')
plt.ylabel('Current [A]')
plt.title('Current at Cycle 57', fontsize=14)
plt.grid()
plt.plot()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(time_spm, current_spm, linewidth=2, label='True SoC')
plt.legend()
plt.grid()
plt.plot()
plt.show()

# ------------ Compute RMSE --------------
# Interpolate SPM predictions to DFN time steps
spm_enhanced_interp = interp1d(
    time_spm, soc_spm_conc,
    kind='linear', bounds_error=False, fill_value="extrapolate"
)(time_cycle)

spm_classic_interp = interp1d(
    time_spm, soc_spm_klassik_conc,
    kind='linear', bounds_error=False, fill_value="extrapolate"
)(time_cycle)


def compute_rmse(true_values, predicted_values):
    # Remove NaN values (if any due to extrapolation)
    mask = ~np.isnan(predicted_values)
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]

    # Compute RMSE
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    return rmse


# Calculate RMSE for both SPM models
rmse_enhanced = compute_rmse(soc_DFN_conc, spm_enhanced_interp)
rmse_classic = compute_rmse(soc_DFN_conc, spm_classic_interp)

print(f"RMSE (Enhanced SPM): {rmse_enhanced:.6f}")
print(f"RMSE (Classic SPM): {rmse_classic:.6f}")

soc_range = np.max(soc_DFN_conc) - np.min(soc_DFN_conc)
rmse_enhanced_percent = (rmse_enhanced / soc_range) * 100
rmse_classic_percent = (rmse_classic / soc_range) * 100

print(f"Normalized RMSE (Enhanced SPM): {rmse_enhanced_percent:.2f}%")
print(f"Normalized RMSE (Classic SPM): {rmse_classic_percent:.2f}%")


def compute_mae(true_values, predicted_values):
    # Remove NaN values (if any due to extrapolation)
    mask = ~np.isnan(predicted_values)
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]

    # Compute MAE (Mean Absolute Error)
    mae = np.mean(np.abs(true_values - predicted_values))
    return mae


# Calculate MAE for both SPM models
mae_enhanced = compute_mae(soc_DFN_conc, spm_enhanced_interp)
mae_classic = compute_mae(soc_DFN_conc, spm_classic_interp)

print(f"MAE (Enhanced SPM): {mae_enhanced:.6f}")
print(f"MAE (Classic SPM): {mae_classic:.6f}")

soc_range = np.max(soc_DFN_conc) - np.min(soc_DFN_conc)
mae_enhanced_percent = (mae_enhanced / soc_range) * 100
mae_classic_percent = (mae_classic / soc_range) * 100

print(f"Normalized MAE (Enhanced SPM): {mae_enhanced_percent:.2f}%")
print(f"Normalized MAE (Classic SPM): {mae_classic_percent:.2f}%")


# ------------ Additional plots --------------

### Current profile with dynamic cycling
'''
plt.figure(figsize=(5.5, 3.4))
plt.plot(t, current, label='Current', linewidth=2)

for i, cycle in enumerate(sol.cycles):
    t_start_cycle = cycle['Time [h]'].data
    plt.axvline(x=t_start_cycle[0], linestyle='--', color='red', linewidth=1)

plt.title('Cycling protocol for dynamic current profile with cycle indicator', fontsize=12)
plt.xlabel('Time [h]')
plt.ylabel('Current [A]')
plt.grid()
plt.legend()
plt.show()
'''

'''
### Cs_avg vs cs_max ###
plt.figure(figsize=(5.5, 3.4))
plt.plot(t, cs_avg, label='cs_avg', linewidth=2)
plt.title('cs_max evolution')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(5.5, 3.4))
plt.plot(t, soc_conc, label='SoC with x', linewidth=2)
plt.plot(t, soc_cc, linestyle='--', label='SoC with CC', linewidth=2)
plt.title('SoC methods comparison: CC vs proposed')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(5.5, 3.4))
plt.plot(t, current, label='Current')
plt.title('Current [A]')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(5.5, 3.4))
plt.plot(t, voltage, label='Voltage')
plt.title('Voltage')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(5.5, 3.4))
plt.plot(t, x, label='x')
plt.title('x')
plt.legend()
plt.grid()
plt.show()
'''

