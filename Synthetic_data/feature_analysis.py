import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/david/PycharmProjects/memoire/Synthetic_data/battery_features_0_5C_298K.csv'
df = pd.read_csv(file_path)

# Drop rows with NaN in Time_Between_Cycles_seconds for correlation analysis
df_clean = df.dropna()

features = df_clean.drop('SoH_at_Cycle_End', axis=1)
target = df_clean['SoH_at_Cycle_End']
cycle = df_clean['Cycle']


# Plot all features vs SoH in one figure with SoH decreasing from left to right
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))  # Updated to 3x3 for 7 features
fig.suptitle('Features vs SoH', fontsize=12)
axes = axes.flatten()

for idx, feature in enumerate(features.columns):
    axes[idx].scatter(target, features[feature], alpha=0.5)
    axes[idx].set_title(f'{feature} vs SoH')
    axes[idx].set_xlabel('SoH_at_Cycle_End')
    axes[idx].set_ylabel(feature)
    axes[idx].grid(True)
    axes[idx].invert_xaxis()  # Highest SoH on left

# Remove extra subplots (7 features < 9 slots)
for idx in range(len(features.columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

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



'''
# Plot all features vs Cycle in one figure
# Use original df (with NaN) to show all cycles
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))  # Updated to 3x3
fig.suptitle('Features vs Cycle', fontsize=16)
axes = axes.flatten()

for idx, feature in enumerate(df.drop('SoH_at_Cycle_End', axis=1).columns):
    axes[idx].scatter(df['Cycle'], df[feature], alpha=0.5)
    axes[idx].set_title(f'{feature} vs Cycle')
    axes[idx].set_xlabel('Cycle')
    axes[idx].set_ylabel(feature)
    axes[idx].grid(True)

for idx in range(len(features.columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Visualize cross-correlation matrix
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Cross-correlation between Features')
plt.show()

# Visualize Pearson correlations with SoH as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pearson_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Pearson Correlation with SoH (Heatmap)')
plt.xlabel('')
plt.tight_layout()
plt.show()

# Feature selection suggestion based on correlation thresholds
print("\nFeature Selection Suggestion:")
print("Features with high correlation with target (|corr| > 0.7):")
for feature, values in pearson_correlations.items():
    if abs(values['correlation']) > 0.7 and values['p_value'] < 0.05:
        print(f"{feature}: correlation = {values['correlation']:.4f}")

'''