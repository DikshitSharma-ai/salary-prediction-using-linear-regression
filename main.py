import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("salary_data.csv")

print("First 5 Rows:")
print(df.head())

# =====================================
# Dataset Information
# =====================================

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# =====================================
# Features and Label
# =====================================

X = df[["Experience", "Hours_Worked"]]

y = df["Salary"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =====================================
# Create Model
# =====================================

model = LinearRegression()

# =====================================
# Train Model
# =====================================

model.fit(X_train, y_train)

print("\nModel Training Completed")

# =====================================
# Prediction on Test Data
# =====================================

predictions = model.predict(X_test)

# =====================================
# Model Evaluation
# =====================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")

print("MAE :", mae)

print("MSE :", mse)

print("R2 Score :", r2)

# =====================================
# Actual vs Predicted Values
# =====================================

results = pd.DataFrame({
    "Actual Salary": y_test,
    "Predicted Salary": predictions
})

print("\nActual vs Predicted")
print(results)

# =====================================
# Future Salary Prediction
# =====================================

new_employee = [[5, 8]]

future_salary = model.predict(new_employee)

print(
    "\nPredicted Salary for Employee "
    "(5 Years Experience, 8 Hours Worked):"
)

print(future_salary)

# =====================================
# Visualization 1
# Experience vs Salary
# =====================================

plt.figure(figsize=(6,4))

plt.scatter(
    df["Experience"],
    df["Salary"]
)

plt.xlabel("Experience")

plt.ylabel("Salary")

plt.title("Experience vs Salary")

plt.show()

# =====================================
# Visualization 2
# Actual vs Predicted
# =====================================

plt.figure(figsize=(6,4))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Salary")

plt.ylabel("Predicted Salary")

plt.title("Actual vs Predicted Salary")

plt.show()