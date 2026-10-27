#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
plt.show()


# In[5]:


import pandas as pd

# Load dataset from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first 5 rows
df.head()


# In[8]:


df.columns


# In[12]:


# Display data types of each column
df.dtypes


# In[14]:


# Check for missing values
df.isnull().sum()


# In[18]:


# Descriptive Statistics
df.describe()


# In[35]:


sns.set(style="whitegrid")


# In[37]:


# Histogram of passenger ages
plt.figure(figsize=(8, 5))
df['Age'].hist(bins=30, edgecolor='black')
plt.title('Histogram of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[39]:


# Bar chart showing number of passengers 
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', data=df, palette='viridis')
plt.title('Number of Passengers in Each Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()


# In[41]:


# Scatter plot of fare vs age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='coolwarm')
plt.title('Fare vs. Age Colored by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


# In[43]:


# Boxplot of fare of each passenger class
plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set2')
plt.title('Box Plot of Fare by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()


# In[45]:


# Countplot of survived
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', hue='Sex', data=df, palette='pastel')
plt.title('Survival Count by Sex')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# In[49]:


# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Display the correlation matrix
print(corr_matrix)


# In[51]:


# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[53]:


# Create new feature "FamilySize"
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


# In[57]:


# Create new feature "IsAlone"
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)


# In[59]:


# Verify by displaying first 5 rows
print(df[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head())


# In[ ]:





# In[ ]:




