import numpy as np
import pybamm
import pybamm as pb
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
import psutil
import os

print(f"PyBaMM version: {pybamm.__version__}")
# ------------- Initialization --------------------------
parameter_values_ORegan = pybamm.ParameterValues("ORegan2022")
parameter_values_okane = pybamm.ParameterValues("OKane2022")
combined_dict = {**parameter_values_okane, **parameter_values_ORegan}
combined_parameters = pybamm.ParameterValues(combined_dict)
parameter_values_sim = pybamm.ParameterValues("Chen2020")
parameter_values_sim.update({"SEI kinetic rate constant [m.s-1]": 5e-15})

# -------------------- MODEL ---------------------
model_DFN_deg = pb.lithium_ion.DFN(options={"SEI": "ec reaction limited",
                                        "thermal": "lumped"})

model_SPM_simple = pb.lithium_ion.SPM()

model_DFN_simple = pb.lithium_ion.DFN()

models = [model_SPM_simple, model_DFN_simple]

experiment = pb.Experiment(
    ["Discharge at 2 C until 2.5 V",
     "Charge at 0.3C until 4.2 V",
     "Hold at 4.2 V until C/100",
     ]
)

var_pts = {"x_n": 15, "x_s": 15, "x_p": 15, "r_n": 60, "r_p": 60}

# Simulate both models and collect data
results = {}
for model in models:
    print(f"Simulating {model.name}...")
    sim = pb.Simulation(
        model=model,
        experiment=experiment,
        parameter_values=parameter_values_sim,
        var_pts=var_pts
    )
    sol = sim.solve()

    time = sol["Time [h]"].entries
    voltage = sol["Terminal voltage [V]"].entries
    soc = sol["Negative electrode stoichiometry"].entries

    results[model.name] = {"time": time, "voltage": voltage, "soc": soc}

# Plot voltage comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for model_name, data in results.items():
    plt.plot(data["time"], data["voltage"], label=model_name, linewidth=2)  # Thicker lines
plt.xlabel("Time [h]", fontweight= 'bold')
plt.ylabel("Terminal Voltage [V]", fontweight= 'bold')
plt.title("Voltage Comparison: DFN vs SPM", fontweight= 'bold')
plt.legend()
plt.grid(True)

# Plot SOC comparison
plt.subplot(1, 2, 2)
for model_name, data in results.items():
    plt.plot(data["time"], data["soc"], label=model_name, linewidth=2)  # Thicker lines
plt.xlabel("Time [h]", fontweight= 'bold')
plt.ylabel("SoC", fontweight= 'bold')
plt.title("SoC Comparison: DFN vs SPM", fontweight= 'bold')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure as a PDF
plt.savefig('voltage_soc_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()