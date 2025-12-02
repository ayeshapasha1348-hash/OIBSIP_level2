import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# STEP 1: Load Dataset
# ============================
df = pd.read_csv("house_price_regression_dataset.csv")
print("✅ Dataset Loaded Successfully")

# ============================
# STEP 2: Data Exploration
# ============================
print("\n🔹 First 5 Rows:")
print(df.head())

print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Missing Values:")
print(df.isnull().sum())

print("\n🔹 Summary Statistics:")
print(df.describe())

# ============================
# STEP 3: Duplicate Rows Check
# ============================
duplicate_count = df.duplicated().sum()
print(f"\n🔹 Duplicate Rows Found: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("✅ Duplicate rows removed!")
else:
    print("✔ No duplicate rows.")

# ============================
# STEP 4: Wrong Data Types Check
# ============================
print("\n🔹 Checking Wrong Data Types:")
print(df.dtypes)

# ============================
# STEP 5: Invalid Values Check
# ============================
invalid_values = (df < 0).sum()
print("\n🔹 Invalid (Negative) Values:")
print(invalid_values)
df = df[df >= 0].dropna()
print("✔ Negative values handled.")

# ============================
# STEP 6: Outlier Detection (IQR Method)
# ============================
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("\n🔹 Outliers Count:")
print(outliers)
# Note: We do not remove outliers for regression

# ============================
# STEP 7: Missing Value Treatment
# ============================
df['Square_Footage'] = df['Square_Footage'].fillna(df['Square_Footage'].mean())
df['Lot_Size'] = df['Lot_Size'].fillna(df['Lot_Size'].mean())
df['Neighborhood_Quality'] = df['Neighborhood_Quality'].fillna(df['Neighborhood_Quality'].mean())
df['House_Price'] = df['House_Price'].fillna(df['House_Price'].mean())
print("\n✔ Missing values filled successfully!")

# ============================
# STEP 8: Feature Selection
# ============================
corr_matrix = df.corr()
print("\n🔹 Correlation with House_Price:")
print(corr_matrix['House_Price'].sort_values(ascending=False))

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Select features
X = df[['Square_Footage', 'Lot_Size', 'Neighborhood_Quality']]
y = df['House_Price']
print("\n✔ Selected Features for Model:")
print(X.head())

# ============================
# STEP 9: Train-Test Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✔ Data split into train and test sets")

# ============================
# STEP 10: Train Linear Regression Model
# ============================
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✔ Linear Regression Model Trained")

# ============================
# STEP 11: Predictions
# ============================
y_pred = model.predict(X_test)

# ============================
# STEP 12: Model Evaluation
# ============================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# ============================
# STEP 13: Visualization
# ============================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Optional: Feature Coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n🔹 Feature Coefficients:")
print(coeff_df)   

df.to_csv("house_price_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'house_price_cleaned.csv'")

