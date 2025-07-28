import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

file_path = '/Users/david/PycharmProjects/memoire/Synthetic_data/battery_features_1C_27_07.csv'
df = pd.read_csv(file_path)

# Drop rows with NaN in Time_Between_Cycles_seconds for correlation analysis
df_clean = df.dropna()
print(df_clean.head())

# Prepare features and target for correlation analysis (unscaled)
features = df_clean.drop(['SoH_at_Cycle_End'], axis=1)
target = df_clean['SoH_at_Cycle_End']
cycle = df_clean['Cycle']

# Scale features for plotting
scaler = MinMaxScaler()
# Scale features for "Features vs SoH"
features_scaled = pd.DataFrame(features, columns=features.columns, index=features.index)
# Scale all features in df except 'SoH_at_Cycle_End' for "Features vs Cycle"
df_scaled = df.copy()
df_scaled[features.columns] = scaler.fit_transform(df[features.columns])

# Plot scaled features vs SoH in one figure with SoH decreasing from left to right
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 13))
fig.suptitle('Scaled Features vs SoH', fontsize=20)
axes = axes.flatten()

for idx, feature in enumerate(features_scaled.columns):
    axes[idx].plot(target, features_scaled[feature], alpha=0.5, linewidth = 3)
   # axes[idx].set_xlabel('SoH_at_Cycle_End')
    axes[idx].set_ylabel(f'{feature} ', fontsize = 11)
    axes[idx].grid(True)
    axes[idx].invert_xaxis()  # Highest SoH on left

# Remove extra subplots (7 features < 9 slots)
for idx in range(len(features_scaled.columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Compute correlations (using unscaled features as before)
correlation_matrix = features.corr()

pearson_correlations = {}
for column in features.columns:
    corr, p_value = pearsonr(features[column], target)
    pearson_correlations[column] = {'correlation': corr, 'p_value': p_value}

pearson_df = pd.DataFrame({
    'Feature': list(pearson_correlations.keys()),
    'Correlation': [values['correlation'] for values in pearson_correlations.values()]
}).set_index('Feature')

print("Cross-correlation between features:")
print(correlation_matrix)
print("\nPearson correlation with target (SoH):")
for feature, values in pearson_correlations.items():
    print(f"{feature}: correlation = {values['correlation']:.4f}, p-value = {values['p_value']:.4f}")

# Plot scaled features vs Cycle in one figure
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 12))
fig.suptitle('Scaled Features vs Cycle', fontsize=20)
axes = axes.flatten()

for idx, feature in enumerate(df_scaled.drop(['SoH_at_Cycle_End'], axis=1).columns):
    axes[idx].plot(df['Cycle'], df_scaled[feature], alpha=0.5)
   # axes[idx].set_title(f'{feature} (Scaled) vs Cycle')
    axes[idx].set_xlabel('Cycle', fontsize= 12)
    axes[idx].set_ylabel(f'{feature}', fontsize= 12)
    axes[idx].grid(True)

for idx in range(len(df_scaled.drop('SoH_at_Cycle_End', axis=1).columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Visualize Pearson correlations with SoH as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pearson_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Pearson Correlation with SoH (Heatmap)')
plt.xlabel('')
plt.tight_layout()
plt.show()