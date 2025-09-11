import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, max_error

df_03C = pd.read_csv('/Users/david/PycharmProjects/memoire/Synthetic_data/battery_features_0_3C_298K.csv')
df_05C = pd.read_csv('/Users/david/PycharmProjects/memoire/Synthetic_data/battery_features_0_5C_298K.csv')
df_08C = pd.read_csv('/Users/david/PycharmProjects/memoire/Synthetic_data/battery_features_0_8C_298K.csv')
df_1C = pd.read_csv("/Users/david/PycharmProjects/memoire/Synthetic_data/battery_features_1C_298K.csv")

df_03C = df_03C[df_03C['SoH_at_Cycle_End'] >= 80]
df_05C = df_05C[df_05C['SoH_at_Cycle_End'] >= 80]
df_08C = df_08C[df_08C['SoH_at_Cycle_End'] >= 80]
df_1C = df_1C[df_1C['SoH_at_Cycle_End'] >= 80]


df = pd.concat([df_03C, df_05C, df_1C])

features = ['CV_Duration_seconds', 'Avg_Temperature_during_CCCV_Celsius', 'Avg_Voltage_during_CC']
target = 'SoH_at_Cycle_End'

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------  Mon Modèle ------- #

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

rmse = root_mean_squared_error(y_pred, y_test)
print(f"RMSE on Cross validation dataset (20% dataset @ 0.3C + 0.5C + 1C): {rmse}")

mse = mean_squared_error(y_pred, y_test)
print(f"MSE on test (20% dataset) on 1C: {mse}")

mae = max_error(y_pred, y_test)
print(f'Maximal error : {mae} % ')

# -------  Interprétation graphique des résultats sur ma cross-validation ------- #

residuals = y_pred - y_test

plt.figure(figsize=(12,5))

# Plot 1: Residual distribution
plt.subplot(1, 2, 1)
sns.histplot(residuals, bins = 30, kde = True, color='blue')
plt.axvline(x = 0, color = 'red', linestyle = '--')
plt.title('Résidual Distribution')
plt.xlabel('Residual (y_pred - y_test)')
plt.ylabel('frequency')

# Plot 2: Regression fit
plt.subplot(1,2,2)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Regression Fit: Actual vs Predicted")
plt.xlabel("Actual SoH (%)")
plt.ylabel("Predicted SoH (%)")

plt.tight_layout()
plt.show()

# -------  Prediction @ 0_8C ------- #

X_08C = df_08C[features].values
y_08C = df_08C[target].values

X_08C_scaled = scaler.transform(X_08C)

y_pred_08C = model.predict(X_08C_scaled)

rmse_08C = root_mean_squared_error(y_pred_08C, y_08C)
print(f"RMSE 0.8C pred: {rmse_08C}")

mse_08C = mean_squared_error(y_pred_08C, y_08C)
print(f'MSE 0.8C pred: {mse_08C}')

mae_08C = max_error(y_pred_08C, y_08C)
print(f'Maximal error on 0.8C pred: {mae_08C}')

plt.figure(figsize=(12,5))

# Plot 1: Residual distribution
residuals_08C = y_08C - y_pred_08C
plt.subplot(1, 2, 1)
sns.histplot(residuals_08C, bins = 30, kde = True, color='blue')
plt.axvline(x = 0, color = 'red', linestyle = '--')
plt.title('Résidual Distribution')
plt.xlabel('Residual (y_pred - y_test)')
plt.ylabel('frequency')

# Plot 2: Regression fit
plt.subplot(1,2,2)
sns.scatterplot(x=y_08C, y=y_pred_08C, alpha=0.5)
plt.plot([min(y_08C), max(y_08C)], [min(y_08C), max(y_08C)], color='red', linestyle='--')
plt.title("Regression Fit: Actual vs Predicted")
plt.xlabel("Actual SoH (%)")
plt.ylabel("Predicted SoH (%)")

plt.tight_layout()
plt.show()

#Plot 3: Prediction vs Actual SoH
plt.figure(figsize=(12, 8))
plt.plot(y_pred_08C, label = 'Prediction with lin. regression')
plt.plot(y_08C, label = 'Actual SoH')
plt.title('Prediction with linear regression')
plt.ylabel('SoH (%)')
plt.xlabel('Cycle number')
plt.legend()


plt.show()



