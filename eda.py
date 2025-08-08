import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the combined dataset
wine_data = pd.read_csv('combined_wine_data.csv')

# Check for missing values
print('\nMissing values:\n', wine_data.isnull().sum())

# Visualize distributions of numerical features
wine_data.hist(bins=20, figsize=(15, 10))
plt.suptitle('Distribution of Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('feature_distributions.png')

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(wine_data.drop('type', axis=1).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Wine Features')
plt.savefig('correlation_matrix.png')

# Box plots to identify outliers
plt.figure(figsize=(15, 10))
for i, column in enumerate(wine_data.drop(['quality', 'type'], axis=1).columns):
    plt.subplot(3, 4, i + 1) # Adjust subplot grid based on number of features
    sns.boxplot(y=wine_data[column])
    plt.title(column)
plt.tight_layout()
plt.savefig('outlier_boxplots.png')

print('Exploratory data analysis complete. Visualizations saved as PNG files.')


