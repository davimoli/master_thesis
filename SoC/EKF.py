import pybamm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time

#### Obtain Inputs from User ###

#Crate = float(input("C_rate? (ex: 1, 0.5, 0.1) \n"))

#temp = float(input("Temperature? [°C] \n"))

#show = input("Would you like to see DFN result? (Y/N) \n")
Crate = 0.1

temp = 20
show = "N"


if show!='Y' and show!='N':
    print("%s not recognized. Please use Y or N"%show)

#### Simulate "Real" Data Using DFN Model ####
dfn_model = pybamm.lithium_ion.DFN(options={"thermal": "x-full"}) #Define DFN model with lumped thermal eqn
pv = pybamm.ParameterValues("Chen2020") #Use LGM50 Cell Parameters (NMC-SiGraphite-LiPF6)
current = Crate*pv['Nominal cell capacity [A.h]']

#Define Ambient/Initial temperatures
pv["Ambient temperature [K]"] = temp+273
pv["Initial temperature [K]"] = temp+273

#Add undefined thermal parameters to set
custom_params = {
    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Negative tab heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Negative tab width [m]": 0.01,
    "Edge heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Positive tab width [m]": 0.01,
}
pv.update(custom_params,check_already_exists=False)


#Apply a dynamic current profile
fs = 100
Ts = 1/fs           # Sampling time
total_time = 60   # In seconds
n_ti = int(total_time/Ts)      # Number of sampling points
time = np.linspace(0, total_time, 1000)  #1-hour simulation with n_ti + 1 sampling points since we are taking 0
time_sampled = np.linspace(0, total_time, n_ti+1)
current_profile = lambda t: current*np.sin(2 * np.pi * t / total_time) #Sinusoidal current [A] to model realistic conditions
pv["Current function [A]"] = current_profile
sim_dfn = pybamm.Simulation(dfn_model,parameter_values=pv) #Create simulation with model
sim_dfn.set_initial_soc(0.5)
#Run DFN simulation
sol_dfn = sim_dfn.solve(t_eval= time, t_interp=time_sampled)

print(time[-1])
print("Solution was generated at times: ", sol_dfn.t)
print("Time_sampled vector:", time_sampled)
print("Solution time duration: ", sol_dfn.total_time)
print("Number of sampling points", n_ti)

#Extract "measured" terminal voltage, SOC, and temperature
voltage_dfn = sim_dfn.solution["Terminal voltage [V]"].data

#Calculate SOC based on surface lithium concentration
c_s_n = sim_dfn.solution["Average negative particle concentration"].data
avg_neg_part_stoich = sol_dfn['Average negative particle stoichiometry'].data
avg_pos_part_stoich = sol_dfn['Average positive particle stoichiometry'].data
max_neg_part_stoich = sol_dfn['Maximum negative particle stoichiometry'].data
min_neg_part_stoich = sol_dfn['Minimum negative particle stoichiometry'].data
neg_electrode_stoich = sol_dfn['Negative electrode stoichiometry'].data

soc_dfn = neg_electrode_stoich

#Extract OCV vs SOC from DFN Model
ocv_dfn = sim_dfn.solution["Battery open-circuit voltage [V]"].data

#Compute dV/dSOC using finite differences
H_values = np.gradient(ocv_dfn, soc_dfn)
H_interp = lambda soc: np.interp(soc, soc_dfn, H_values)

temp_dfn = sim_dfn.solution["X-averaged cell temperature [K]"].data

#Add measurement noise
voltage_measured = voltage_dfn + np.random.normal(0, 0.001, size=voltage_dfn.shape)  #10 mV noise
temp_measured = temp_dfn + np.random.normal(0, 0.5, size=temp_dfn.shape)  - 273#0.5 K noise
time_sim = sim_dfn.solution["Time [s]"].data




#### Set up the SPM for EKF #####
spm_model = pybamm.lithium_ion.SPM()
sim_spm = pybamm.Simulation(spm_model,parameter_values=pv)
sim_spm.set_initial_soc(0.5)

##### Implement EKF for SOC Estimation #####

#EKF parameters
Q = np.array([[1e-5]])  # Process noise covariance (SOC)
R = np.array([[1e-4]])  # Measurement noise covariance (Voltage)

#Initial estimates
state_est = np.zeros(len(time_sampled))  #SOC
state_est[0] = 0.5  #Initial SOC guess
P = np.array([[0.01]])  #Initial error covariance
execution_time = np.zeros(len(time_sampled)) #Average execution time per step
RMSE = 0 #Root-mean-squared error


for k in range(1, len(time_sampled)):
    start_time = datetime.now()
    dt = Ts
    current = current_profile(k*Ts)
#     current = current_profile[k]

    #Prediction step using SPM model
    sim_spm_sol = sim_spm.solve(t_eval=[0, k*Ts])
    c_s_n_spm = sim_spm.solution["X-averaged negative particle surface concentration [mol.m-3]"].data[-1]
    voltage_pred = sim_spm.solution["Terminal voltage [V]"].data[-1]
#    soc_pred = c_s_n_spm /33133.0 #33133.0 is maximum conc on surface from Chen2020 paramter set
    soc_pred = sim_spm_sol['Negative electrode stoichiometry'].data[-1]
    P_pred = P + Q

    #Correction step
    z_k = voltage_measured[k]  #Measurement
    h_x = voltage_pred  #Model prediction
    y_k = z_k - h_x  #Innovation

    #Measurement sensitivity matrix H computed from OCV-SOC curve
    H = np.array([[H_interp(soc_pred)]])

    #Kalman Gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    #State update
    state_est[k] = soc_pred + (K * y_k).item()
    P = (np.eye(1) - K @ H) @ P_pred
    end_time = datetime.now()
    execution_t = end_time - start_time
    execution_time[k] = execution_t.total_seconds()

    #Calculate Error b/t Estimator and DFN
    RMSE += (soc_dfn[k] - state_est[k])**2

#Plot SOC estimation results
plt.figure(figsize=(10, 5))
plt.plot(time_sim, soc_dfn, label="True SOC (DFN)", color='g',linewidth=2)
plt.plot(time_sampled, state_est, label="Estimated SOC (SPM+EKF)", linestyle='--', linewidth=0.5)
plt.xlabel("Time [min]")
plt.ylabel("State of Charge")
plt.title("SOC Estimate at %gC and %g°C"%(Crate,temp))
plt.legend()
plt.show()
print("Average Execution Time: %g | #Estimates/sec: %g | RMSE_t: %g" %(np.mean(execution_time),1/np.mean(execution_time),np.sqrt(RMSE)/time_sim[-1]))

print("Len soc_dfn:",len(soc_dfn))
print("Len soc_etimated:", len(state_est))
