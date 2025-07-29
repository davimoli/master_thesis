import pybamm
import pybamm as pb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------- PARAMETRISATION ----------------------

parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({"SEI kinetic rate constant [m.s-1]": 5e-15})  # To fit Okane's Deg
param_artemis = parameter_values.copy()
# ----------------------- Models ----------------------
model = pybamm.lithium_ion.SPM(options={"SEI": "ec reaction limited",
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
            "Discharge at 0.5C for 1 hour",

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
     * 3

    +
        (
        pybamm.step.current(ac_current_step(t= pb.t, C_rate=1), duration="0.05 hours"),
        pybamm.step.current(neg_ac_current_step(t= pb.t, C_rate=1), duration="0.05 hours")
        )
     * 3

    +
        (
            pybamm.step.current(ac_current_step(t=pb.t, C_rate=2), duration="0.025 hours"),
            pybamm.step.current(neg_ac_current_step(t=pb.t, C_rate=2), duration="0.025 hours"),

        )
     * 3]
)

# ----------------------- MY Simulations ----------------------
sim_capacity_check = pb.Simulation(model, parameter_values=parameter_values, experiment=capacity_check_exp)
sim_artemis = pb.Simulation(model, parameter_values=parameter_values, experiment=ac_experiment)



### SPM ###


# ----------------------- Solving the sims  ----------------------
M = 70 #Number of cycles
initial_soc = 0.5
sol = sim_capacity_check.solve(initial_soc=initial_soc)

for i in range(M):
    if i != 0:
        sol = sim_artemis.solve(starting_solution=sol)
        sol = sim_capacity_check.solve(starting_solution=sol)


print(f"Sol length:", len(sol.cycles))


### Solving the SPM model ###
#sol_spm = sim_spm.solve(initial_soc=1)

# ------------ Feature extractions --------------
for i, cycle in sol.cycles:
    if (i+1)%2 == 0:
        break


















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

