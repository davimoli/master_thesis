import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('/Users/david/PycharmProjects/memoire/Data/1C_OKane_advices_x15r60_11_07.csv')
#4C_298K_rpt_02_05.csv
# Extract time, voltage, current, and discharge capacity
time = df['Time_s'].values
voltage = df['Terminal_Voltage_V'].values
current = df['Current_A'].values
discharge_capacity = df['Discharge_Capacity_Ah'].values
plt.plot(discharge_capacity, marker='o')
plt.show()

# Nominal capacity (assumed; adjust based on your simulation)
nominal_capacity = 5.0  # A.h

# Threshold for CV end detection
c_100_threshold = nominal_capacity / 100

# Parameters for CCCV detection
eps = 1e-2
v_cc_start_counting = 3.8
voltage_cv_threshold = 4.2 - 0.01
v_low_cutoff = 2.5  # Assumed discharge cutoff
cccv_starts = []
cc_starts_41 = []
cv_starts = []
cv_ends = []
cc_durations = []
cv_durations = []
voltage_slopes_at_38V = []
avg_temps_during_cccv = []
avg_current_during_cccv = []
avg_voltages_during_cc = []
energy_during_cc = []
time_between_cycles = []
soh_at_cycle_ends = []
time_at_cycle_ends = []

# Detect CCCV charging starts
cccv_starts = []
counter_cycles = 1

# Capacity detection at the end of each rpt cycles
initial_cap = None
rpt_capacities = []
lowest_cap = min(discharge_capacity)
print(lowest_cap)
for i in range(1, len(voltage)):
    if voltage[i - 1] < v_low_cutoff + eps and voltage[i] > v_low_cutoff + eps and current[i] < 0:
        cccv_starts.append(i)
        print(f"Time of Cycle {counter_cycles} = {time[i]/3600:.2f} hours")
        counter_cycles += 1

# Feature extraction partial algo
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
                distance_to_41_i = abs(voltage[i] - v_cc_start_counting)
                ditance_to_41_i_1 = abs(voltage[i-1] - v_cc_start_counting)
                if min(ditance_to_41_i_1, distance_to_41_i) == ditance_to_41_i_1:
                    cc_start_41_idx = i-1
                    print("Took the it before, we were doing something wrong blud")
                else:
                    cc_start_41_idx = i
                break

    cv_start_idx = None
    for i in range(start_idx, len(voltage) - 1):
        if (voltage[i] >= voltage_cv_threshold  and
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

for (cycle_numb, cccv_start) in enumerate(cccv_starts):
    if cycle_numb == 0:
        initial_cap = max(discharge_capacity[0: cccv_start]) - lowest_cap
        rpt_capacities.append(initial_cap)
        print(f'Initial Capacity: {initial_cap}')

    elif (cycle_numb) %11 == 0 :

        d_c_range = discharge_capacity[cv_ends[cycle_numb-10]:cccv_start+5]
        if len(d_c_range) < 1:
            break
        rpt_capacity = max(d_c_range)  - lowest_cap
        rpt_capacities.append(rpt_capacity)
        print(f'RPT CAPACITY AT CYCLE {cycle_numb}: {rpt_capacity}')

soh_vector = rpt_capacities/initial_cap


plt.plot(soh_vector)
plt.show()

plt.plot(voltage)
plt.show()










# Select cycles to plot (1, 20, 50, 100, 250)
selected_cycles = [1, 50, 70, 75, 90]
selected_starts = [cccv_starts[i-1] for i in selected_cycles if i-1 < len(cccv_starts)]

# Detect CC and CV phases for each cycle
cc_starts_41 = []
cv_starts = []
cv_ends = []
for cycle_num, start_idx in enumerate(selected_starts, 1):
    # Find CC start (3.8 V)
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
                distance_to_41_i = abs(voltage[i] - v_cc_start_counting)
                distance_to_41_i_1 = abs(voltage[i-1] - v_cc_start_counting)
                if min(distance_to_41_i_1, distance_to_41_i) == distance_to_41_i_1:
                    cc_start_41_idx = i-1
                    print(f"Cycle {cycle_num}: Took the index before for CC start")
                else:
                    cc_start_41_idx = i
                break
    if cc_start_41_idx is None:
        print(f"Cycle {cycle_num}: CC start not found")
        continue

    # Find CV start (4.4 V)
    cv_start_idx = None
    for i in range(start_idx, len(voltage) - 1):
        if (voltage[i] >= voltage_cv_threshold and
                abs(voltage[i + 1] - voltage[i]) < 0.001 and
                current[i] < 0):
            cv_start_idx = i
            break
    if cv_start_idx is None:
        print(f"Cycle {cycle_num}: CV start not found")
        continue

    # Find CV end (current <= C/100 or discharge starts)
    cv_end_idx = None
    for i in range(cv_start_idx, len(voltage) - 1):
        if (abs(current[i]) <= c_100_threshold or
                (current[i] > 0 and voltage[i] < 4.0)):
            cv_end_idx = i
            break
    if cv_end_idx is None:
        print(f"Cycle {cycle_num}: CV end not found")
        continue

    # Store the detected indices
    cc_starts_41.append(cc_start_41_idx)
    cv_starts.append(cv_start_idx)
    cv_ends.append(cv_end_idx)

# Plot CCCV voltage for selected cycles, superimposed
plt.figure(figsize=(10, 6))
colors = ['blue', 'orange', 'green', 'red', 'purple']  # Colors for each cycle
for i, (cycle_num, cc_start, cv_start, cv_end) in enumerate(zip(selected_cycles, cc_starts_41, cv_starts, cv_ends)):
    # Extract CCCV phase (from CC start to CV end)
    time_cccv = time[cc_start:cv_end + 1]
    voltage_cccv = voltage[cc_start:cv_end + 1]
    # Normalize time to start at 0 for this cycle
    time_cccv_normalized = (time_cccv - time_cccv[0]) / 3600  # Convert to hours
    # Plot
    plt.plot(time_cccv_normalized, voltage_cccv, label=f'Cycle {cycle_num}', color=colors[i], linewidth=2)

# Customize the plot
plt.xlabel('Time [hours]', fontsize=20)
plt.ylabel('Voltage [V]', fontsize=20)
plt.title('CCCV Voltage profile (1 data)', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()