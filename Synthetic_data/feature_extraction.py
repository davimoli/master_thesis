''' Features should be extracted from a CCCV charging protocol. The function takes as input the cycles from PyBaMM experiment
  Let denote sol, the PyBaMM solution. cycles = sol.cycles. The function saves the features in a csv file and returns a
  lis of capacities from RPT's experiments '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def extract_battery_features(sol, v_upp_cutoff=4.2, v_low_range=3.8, tolerance=0.01,
                                      tolerance_cv=0.001, x_100_fresh=0.9106121196114528,
                                      save_plot=False, plot_filename='all_cycles_voltage_cumulative.png',
                                      save_csv=True, csv_filename='battery_features_1C_27_07.csv'):
    """
    Parameters:
    -----------
    sol : object
        Object containing cycle data with fields like 'Time [h]', 'Voltage [V]', etc.
    v_upp_cutoff : float
        Upper voltage cutoff for CV phase (default: 4.2 V).
    v_low_range : float
        Lower voltage range for CC phase start (default: 3.8 V).
    tolerance : float
        Tolerance for detecting CC start (default: 0.01 V).
    tolerance_cv : float
        Tolerance for detecting CV phase (default: 0.001 V).
    x_100_fresh : float
        Reference stoichiometry for fresh cell (default: 0.9106121196114528).
    save_plot : bool
        Whether to save the plot (default: True).
    plot_filename : str
        Filename for saving the plot (default: 'all_cycles_voltage_cumulative.png').
    save_csv : bool
        Whether to save features to CSV (default: True).
    csv_filename : str
        Filename for saving the features CSV (default: 'battery_features_1C_27_07.csv').

    Returns:
    --------
    dict
        Dictionary containing extracted features for each valid cycle.
    """

    # Initialize lists for feature extraction
    cc_duration = []
    cv_duration = []
    v_slope_38 = []
    avg_temps_during_cccv = []
    avg_voltages_during_cc = []
    avg_current_during_cccv = []
    energy_during_cc = []
    max_stoichiometries = []
    cycle_times = []
    cycle_voltages = []
    cycle_labels = []

    # Feature extraction loop
    for i, cycle in enumerate(sol.cycles):
        time_cycle = cycle["Time [h]"].data
        t = cycle["Time [s]"].data - cycle["Time [s]"].data[0]
        voltage = cycle["Voltage [V]"].data
        voltage_grad = np.gradient(voltage, t)
        current = cycle['Current [A]'].data
        temperature_celsius = cycle['Volume-averaged cell temperature [K]'].data - 273.15
        neg_electrode_stoichiometry = cycle['Negative electrode stoichiometry'].data
        delta_t_vector = np.diff(t)

        cc_38_start_indices = np.where(np.abs(voltage - v_low_range) < tolerance)[0]
        cv_start_indices = np.where(np.abs(voltage - v_upp_cutoff) < tolerance_cv)[0]
        cv_end_indices = np.where(np.abs(voltage - v_upp_cutoff) < tolerance_cv)[0]

        cc38_start_idx = None
        cv_start_idx = None
        cv_end_idx = None

        for idx in cc_38_start_indices:
            if idx < len(voltage) - 1 and voltage[idx + 1] > voltage[idx]:
                cc38_start_idx = idx
                break

        for idx in cv_start_indices:
            if np.abs(voltage[idx + 1] - voltage[idx]) < tolerance_cv:
                print(f"Cycle {i + 1}: CV start detected")
                cv_start_idx = idx
                break

        for idx in cv_end_indices:
            if idx < len(voltage) - 1 and voltage[idx + 1] < voltage[idx] - tolerance_cv:
                cv_end_idx = idx
                break

        if (cc38_start_idx is not None and cv_start_idx is not None and cv_end_idx is not None and
                cc38_start_idx < cv_start_idx < cv_end_idx):
            cc_duration.append(t[cv_start_idx] - t[cc38_start_idx])
            cv_duration.append(t[cv_end_idx] - t[cv_start_idx])
            max_stoichiometry = np.max(neg_electrode_stoichiometry[cv_start_idx:cv_end_idx + 1])
            max_stoichiometries.append(max_stoichiometry / x_100_fresh)

            delta_t = t[cc38_start_idx + 2] - t[cc38_start_idx - 2]
            delta_v = voltage[cc38_start_idx + 2] - voltage[cc38_start_idx - 2]
            slope = delta_v / delta_t
            v_slope_38.append(voltage_grad[cc38_start_idx])

            time_cc = t[cc38_start_idx:cv_end_idx + 1]
            total_t = time_cc[-1] - time_cc[0]

            # Average temperature during CCCV
            temp_cccv = temperature_celsius[cc38_start_idx:cv_end_idx + 1]
            avg_temp = np.trapz(temp_cccv, time_cc) / total_t if total_t > 0 else 0
            avg_temps_during_cccv.append(avg_temp)

            # Average current during CCCV
            current_cccv = current[cc38_start_idx:cv_end_idx + 1]
            avg_curr = np.trapz(current_cccv, time_cc) / total_t if total_t > 0 else 0
            avg_current_during_cccv.append(avg_curr)

            # Average voltage during CCCV
            voltage_cccv = voltage[cc38_start_idx:cv_end_idx + 1]
            avg_voltage = np.trapz(voltage_cccv, time_cc) / total_t if total_t > 0 else 0
            avg_voltages_during_cc.append(avg_voltage)

            # Energy during CCCV
            current_cc = abs(current[cc38_start_idx:cv_end_idx + 1])
            voltage_cc = voltage[cc38_start_idx:cv_end_idx + 1]
            time_cc = t[cc38_start_idx:cv_end_idx + 1]
            energy = np.trapz(voltage_cc * current_cc, time_cc)
            energy_during_cc.append(energy)

            # Store data for plotting
            cycle_times.append(t)
            cycle_voltages.append(voltage)
            cycle_labels.append(f'Cycle {i + 1}')

    # Plot all cycles with cumulative time
    plt.figure(figsize=(12, 6))
    time_offset = 0
    for t, v, label in zip(cycle_times, cycle_voltages, cycle_labels):
        plt.plot((t + time_offset) / 3600, v, label=label)
        time_offset += t[-1]
    plt.title('Voltage vs. Cumulative Time for All Cycles', fontsize=16)
    plt.xlabel('Cumulative Time [h]', fontsize=14)
    plt.ylabel('Voltage [V]', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Plot overall voltage
    plt.figure(figsize=(10, 6))
    plt.plot(sol["Voltage [V]"].data)
    plt.title('Overall Voltage Profile', fontsize=16)
    plt.xlabel('Time Index', fontsize=14)
    plt.ylabel('Voltage [V]', fontsize=14)
    plt.grid(True)
    plt.show()

    # RPT capacity extraction
    rpt_max_capacities = []
    rpt_cycle_numbers = []
    for i, cycle in enumerate(sol.cycles):
        if (i + 1) % 11 == 0 and cycle is not None:
            print(f"Cycle detected: Cycle {i + 1}")
            t = cycle["Time [h]"].data - cycle["Time [h]"].data[0]
            cycle_cap = cycle["Discharge capacity [A.h]"].data
            capacity_max_idx = np.argmax(cycle_cap)
            max_capacity = np.max(cycle_cap)
            rpt_cycle_num = int((i + 1) / 11)
            rpt_max_capacities.append(max_capacity)
            rpt_cycle_numbers.append(rpt_cycle_num)

    # Print debugging information
    print(f"# of Cycles: {len(sol.cycles)}")
    print(f"Length of cc_duration: {len(cc_duration)}")
    print(f"Length of cv_duration: {len(cv_duration)}")
    print(f"Length of v_slope_38: {len(v_slope_38)}")
    print(f"Length of avg_temps_during_cccv: {len(avg_temps_during_cccv)}")
    print(f"Length of avg_voltages_during_cc: {len(avg_voltages_during_cc)}")
    print(f"Length of avg_current_during_cccv: {len(avg_current_during_cccv)}")
    print(f"Length of energy_during_cc: {len(energy_during_cc)}")
    print(f"Length of max_stoichiometries: {len(max_stoichiometries)}")
    print(f"Length of rpt_max_capacities: {len(rpt_max_capacities)}")

    # Create features dictionary
    n_cycles = len(cc_duration)
    features_dict = {
        "Cycle": list(range(1, n_cycles + 1)),
        "CC_Duration_seconds": cc_duration,
        "CV_Duration_seconds": cv_duration,
        "Voltage_Slope_at_38V": v_slope_38,
        "Avg_Temperature_during_CCCV_Celsius": avg_temps_during_cccv,
        "Avg_Current_during_CCCV": avg_current_during_cccv,
        "Avg_Voltage_during_CC": avg_voltages_during_cc,
        "Energy_during_CCCV_VAs": energy_during_cc,
        "SoH_at_Cycle_End": max_stoichiometries
    }

    # Save to CSV if requested
    if save_csv:
        features_df = pd.DataFrame(features_dict)
        features_df.to_csv(csv_filename, index=False)
        print(f"Features saved to {csv_filename}")

    return rpt_max_capacities