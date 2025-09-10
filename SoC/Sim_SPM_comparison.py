import pandas as pd
import matplotlib.pyplot as plt
import pybamm as pb
import pybamm

# Load the CSV file
df = pd.read_csv('/Users/david/PycharmProjects/memoire/Data/1C_OKane_advices_x15r60_11_07.csv')
#4C_298K_rpt_02_05.csv
# Extract time, voltage, current, and discharge capacity
time = df['Time_s'].values
voltage = df['Terminal_Voltage_V'].values
current = df['Current_A'].values
discharge_capacity = df['Discharge_Capacity_Ah'].values
soc = df['State_of_Charge'].values
cap_lost_sr_prcent = df['Capacity_Lost_Percent'].values /100
cap_lost_lam_prcent = df['LAM_neg_electrode'].values/100

total_cap_lost_prcent = cap_lost_sr_prcent + cap_lost_lam_prcent

# Nominal capacity (assumed; adjust based on your simulation)
nominal_capacity = 5.0  # A.h

# Threshold for CV end detection
c_100_threshold = nominal_capacity / 100
cycle_to_plot = 49


# ---------------------------------------------- Data SOC nTh Cycle -------------------------
# Parameters for CCCV detection
eps = 1e-2
v_cc_start_counting = 3.8
voltage_cv_threshold = 4.4 - 0.01
v_low_cutoff = 2.5  # Assumed discharge cutoff
cccv_starts = []

# Detect CCCV charging starts
cccv_starts = []
counter_cycles = 1

for i in range(1, len(voltage)):
    if voltage[i - 1] < v_low_cutoff + eps and voltage[i] > v_low_cutoff + eps and current[i] < 0:
        cccv_starts.append(i)
        print(f"Time of Cycle {counter_cycles} = {time[i]/3600:.2f} hours")
        counter_cycles += 1


soc_to_plot = []

idx_begin = cccv_starts[cycle_to_plot]
idx_end = cccv_starts[cycle_to_plot + 1]

soc_to_plot = soc[idx_begin:idx_end]
time_to_plot = time[idx_begin:idx_end] - time[idx_begin]
current_to_plot = current[idx_begin:idx_end]
voltage_to_plot = voltage[idx_begin:idx_end]
total_cap_lost_cycle_to_plot = total_cap_lost_prcent[int((idx_begin))]
soh_cycle_to_plot = 1 - total_cap_lost_cycle_to_plot/2

# ---------------------------------------------- SPM SOC nTh Cycle -------------------------
parameter_values_ORegan = pybamm.ParameterValues("ORegan2022")
parameter_values_okane = pybamm.ParameterValues("OKane2022")
combined_dict = {**parameter_values_okane, **parameter_values_ORegan}
combined_parameters = pybamm.ParameterValues(combined_dict)
parameter_values_sim = combined_parameters

var_pts = {"x_n": 15, "x_s": 15, "x_p": 15, "r_n": 60, "r_p": 60}


spm = pb.lithium_ion.SPM()

parameter_values_sim["Current function [A]"] = pybamm.Interpolant(time_to_plot, current_to_plot, pybamm.t, interpolator="linear")
spm.initial_conditions.update()

sim = pybamm.Simulation(spm, parameter_values=parameter_values_sim, var_pts= var_pts)
sol = sim.solve(initial_soc= soc_to_plot[0] -0.027)

# Extract simulated SoC and time
sim_time = sol["Time [s]"].entries
sim_soc = sol["Negative electrode stoichiometry"].entries
sim_voltage = sol["Terminal voltage [V]"].entries
sim_current = sol["Current [A]"].entries
sim_soc_updated = sim_soc/soh_cycle_to_plot


print(soc_to_plot[0])
print(sim_soc[0])

# Plot comparison: Original SoC vs SPM simulated SoC
plt.figure(figsize=(10, 5))
plt.plot(time_to_plot, soc_to_plot, linewidth=2, label='Original SoC')
plt.plot(sim_time, sim_soc_updated, linewidth=2, linestyle='--', label='SPM capacity update Simulated SoC')
plt.plot(sim_time, sim_soc, linewidth=2, linestyle='--', label='SPM Simulated SoC')
plt.xlabel('Time within Cycle (s)')
plt.ylabel('State of Charge')
plt.title(f'SoC Comparison: Original vs SPM Model for Cycle {cycle_to_plot + 1}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(time_to_plot, voltage_to_plot, linewidth=2, label= 'SPM Voltage')
plt.plot(sim_time, sim_voltage, linewidth=2, label= 'SPM Voltage')
plt.title(f'Voltage during cycle {cycle_to_plot+1}')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time_to_plot, current_to_plot, linewidth=2, label='Original Current')
plt.plot(sim_time, sim_current, linewidth=2, label='SPM current')
plt.title('CURRENT DURING the cycle')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(time, soh_cycle_to_plot, linewidth=2)
plt.axvline(idx_begin)
plt.legend()
plt.grid()
plt.show()