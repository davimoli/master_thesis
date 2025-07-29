import pybamm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------- PARAMETRISATION ----------------------

parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({"SEI kinetic rate constant [m.s-1]": 5e-15})  # To fit Okane's Deg
print(parameter_values)
# ----------------------- Models ----------------------
model = pybamm.lithium_ion.DFN(options={"SEI": "ec reaction limited",
                                        "thermal": "lumped"})
model = pybamm.lithium_ion.DFN()
model_spm = pybamm.lithium_ion.SPM()

# ----------------------- EXPERIMENTS ----------------------


C = parameter_values['Nominal cell capacity [A.h]']  # To compute the C-rate
N = 10
M = 2
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 2.5V",

            "Charge at 0.3C until 4.2V",
            "Hold at 4.2V until C/100",
            "Rest for 1 hour",

        )
    ]
    * N
    + [
        (
            "Discharge at 0.1C until 2.5V",

            "Charge at 0.3C until 4.2V",
            "Hold at 4.2V until C/50",

        )
    ],

    termination="90% capacity",
)

### Resting experiment ###
rest_experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C for 0.5 hours",
            "Rest for 3 hours",
        )
    ]
)


### Second experiment, after aging: AC current ###
def ac_current_step(t):
    # return 0.5*C*np.sin(2 * np.pi * t / 3600 * 4)
    return 0.5 * C


def neg_ac_current_step(t):
    # return 0.5*C*np.sin(2 * np.pi * t / 3600 * 4)
    return -0.5 * C


ac_experiment = pybamm.Experiment([

    (
        pybamm.step.current(ac_current_step, duration="0.1 hours"),

    )
])

# ----------------------- MY Simulations ----------------------

var_pts = {
    "x_n": 20,  # Negative electrode (x-direction)
    "x_s": 20,  # Separator (x-direction)
    "x_p": 20,  # Positive electrode (x-direction)
    "r_n": 60,  # Negative particle (radial)
    "r_p": 60,  # Positive particle (radial)
}

sim = pybamm.Simulation(
    model, experiment=experiment, parameter_values=parameter_values
)

sim_rest = pybamm.Simulation(
    model, experiment=rest_experiment, parameter_values=parameter_values
)

sim_ac = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=ac_experiment
)

### SPM ###
sim_spm = pybamm.Simulation(model_spm, parameter_values=parameter_values, experiment=experiment)

# ----------------------- Solving the sims  ----------------------
initial_soc = 0.5
sol = sim.solve(initial_soc= initial_soc)  # Running the experiment one time with initial SoC = 100%



# We did our first 10 cycles and the simulation is built, lets now loop over M:
for i in range(M):
    if i != 0:
        sol = sim.solve(starting_solution=sol)



### Solving the SPM model ###
sol_spm = sim_spm.solve(initial_soc=1)
for i in range(M):
    if i != 0:
        sol = sim.solve(starting_solution=sol)



# ------------ Analyzing the results --------------
x_100 = 0.9106121196114528
x_0 = 0.026347301451412015
second_x100 = 0.884
t = sol['Time [h]'].data
voltage = sol['Voltage [V]'].data
x = sol['Negative electrode stoichiometry'].data
soc = (x - x_0) / (x_100 - x_0)
print(model.variables["Negative electrode stoichiometry"])

### In terms of concentration ###
cs_max = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
cs_avg = sol["Average negative particle concentration [mol.m-3]"].data
cs_min = None
soc_cons = cs_avg/cs_max

### Coulomb Counting approach ###
second_rpt = 2.406 + 2.56
initial_cap = 5.123
cc = sol["Discharge capacity [A.h]"].data
soc_cc = initial_soc - 0.016 -  cc/second_rpt




print(np.max(x))
plt.figure(figsize=(5.5, 3.4))
plt.plot(t, soc, label='SoC with x', linewidth=2)
plt.plot(t, soc_cons, linestyle='--', label='SoC with CC', linewidth=2)
plt.title('SoC methods comparison: CC vs proposed')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(5.5, 3.4))
plt.plot(t, cc, label='cc')
plt.title('CC')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(5.5, 3.4))
plt.plot(t, x, label='x')
plt.title('x')
plt.legend()
plt.grid()
plt.show()

