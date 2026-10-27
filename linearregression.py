#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[112]:


df = pd.read_csv("student-por.csv")


# In[114]:


print("Dataset preview:")
print(df.head())


# In[116]:


print("\nMissing values:")
print(df.isnull().sum())


# In[118]:


# Exploratory Data Analysis (EDA)
print("\nSummary statistics:")
print(df.describe())


# In[120]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='studytime', y='G3', hue='failures', palette='coolwarm')
plt.title("Study Time vs Final Grade (G3) with Failures Highlighted")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.show()


# In[122]:


# Ensure only numeric columns are used for correlation
numeric_df = df.select_dtypes(include=['number'])

# Calculate and visualize the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[124]:


X = df[['studytime', 'G1', 'G2', 'failures']]  # Adjust features as needed
y = df['G3']


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[130]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[132]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[134]:


print("\nModel Coefficients:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# In[136]:


y_pred = model.predict(X_test)


# In[138]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)


# In[140]:


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Actual vs Predicted Grades")
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.show()


# In[142]:


residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='green')
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()


# In[144]:


df = pd.read_csv("winequality-white.csv", sep=';')


# In[146]:


print(df.head())
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()


# In[148]:


X = df.drop('quality', axis=1)
y = df['quality']


# In[152]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load datasets
student_file = "student-por.csv"
wine_file = "winequality-white.csv"

# --- 1. STUDENT PERFORMANCE LINEAR REGRESSION ---
df_students = pd.read_csv(student_file)

# Select relevant features
X_students = df_students[['studytime']]
y_students = df_students['G3']

# Train-Test Split
X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(X_students, y_students, test_size=0.2, random_state=42)

# Feature Scaling
scaler_st = StandardScaler()
X_train_scaled_st = scaler_st.fit_transform(X_train_st)
X_test_scaled_st = scaler_st.transform(X_test_st)

# Train the Linear Regression Model
model_st = LinearRegression()
model_st.fit(X_train_scaled_st, y_train_st)

# Make predictions
y_pred_st = model_st.predict(X_test_scaled_st)

# Model Evaluation
mse_st = mean_squared_error(y_test_st, y_pred_st)
rmse_st = np.sqrt(mse_st)
mae_st = mean_absolute_error(y_test_st, y_pred_st)

print("\nStudent Performance - Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_st:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_st:.2f}")
print(f"Mean Absolute Error (MAE): {mae_st:.2f}")

# --- 2. WINE QUALITY LINEAR REGRESSION ---
df_wine = pd.read_csv(wine_file)

# Select relevant feature
X_wine = df_wine[['alcohol']]
y_wine = df_wine['quality']

# Train-Test Split
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Feature Scaling
scaler_w = StandardScaler()
X_train_scaled_w = scaler_w.fit_transform(X_train_w)
X_test_scaled_w = scaler_w.transform(X_test_w)

# Train the Linear Regression Model
model_w = LinearRegression()
model_w.fit(X_train_scaled_w, y_train_w)

# Make predictions
y_pred_w = model_w.predict(X_test_scaled_w)

# Model Evaluation
mse_w = mean_squared_error(y_test_w, y_pred_w)
rmse_w = np.sqrt(mse_w)
mae_w = mean_absolute_error(y_test_w, y_pred_w)

print("\nWine Quality - Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_w:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_w:.2f}")
print(f"Mean Absolute Error (MAE): {mae_w:.2f}")

# --- VISUALIZATION WITH REGRESSION LINES ---

# Student Performance Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_st['studytime'], y=y_test_st, label="Actual Data", alpha=0.7)
sns.lineplot(x=np.sort(X_test_st['studytime']), 
             y=np.sort(y_pred_st), 
             color="red", 
             label="Regression Line")
plt.xlabel("Study Time")
plt.ylabel("Final Exam Score (G3)")
plt.title("Linear Regression: Study Time vs G3")
plt.legend()
plt.show()

# Wine Quality Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_w['alcohol'], y=y_test_w, label="Actual Data", alpha=0.7)
sns.lineplot(x=np.sort(X_test_w['alcohol']), 
             y=np.sort(y_pred_w), 
             color="red", 
             label="Regression Line")
plt.xlabel("Alcohol Content")
plt.ylabel("Wine Quality")
plt.title("Linear Regression: Alcohol Content vs Wine Quality")
plt.legend()
plt.show()


# In[154]:


df_wine = pd.read_csv(wine_file, delimiter=';')  # Adjusting delimiter
print(df_wine.columns)

