
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load actual and predicted values
y_test_actual = pd.read_csv("y_test_actual.csv").values
y_test_predicted = pd.read_csv("y_test_predicted.csv").values

# Flatten the arrays for plotting
y_test_actual = y_test_actual.flatten()
y_test_predicted = y_test_predicted.flatten()

# Create a scatter plot of predicted vs actual values
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_actual, y=y_test_predicted, alpha=0.6)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], color='red', linestyle='--', lw=2)
plt.xlabel('Actual Quality')
plt.ylabel("Predicted Quality")
plt.title("Actual vs. Predicted Wine Quality")
plt.grid(True)
plt.savefig("actual_vs_predicted.png")

print("Visualization of actual vs. predicted values saved as actual_vs_predicted.png")


