# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 2

# Experiment Name

# Exploratory Data Analysis (EDA) on a Real-World Business Dataset using Python

# Aim

# To perform Exploratory Data Analysis (EDA) on a real-world dataset using Python in order to understand the dataset's structure, identify trends, detect anomalies, analyze feature relationships, and generate meaningful business insights through descriptive statistics and visualizations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

from google.colab import files

uploaded = files.upload()

df = pd.read_csv(next(iter(uploaded)))

print("Dataset loaded successfully!")

df.head()

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("Column names:")
print(df.columns.tolist())

df.info()

df.describe()

df.describe(include='object')

for column in df.select_dtypes(include='object').columns:
    print("\n", column)
    print(df[column].value_counts().head())

print("Missing values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Sales'], kde=True)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Profit'], kde=True)
plt.title("Distribution of Profit")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Category')
plt.title("Number of Orders by Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Region')
plt.title("Number of Orders by Region")
plt.xlabel("Region")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Sales', y='Profit')
plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()

category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)

print(category_sales)

category_sales.plot(kind='bar', figsize=(8, 5))

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.show()

region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)

print(region_sales)

region_sales.plot(kind='bar', figsize=(8, 5))

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.show()

category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)

print(category_profit)

category_profit.plot(kind='bar', figsize=(8, 5))

plt.title("Total Profit by Category")
plt.xlabel("Category")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.show()

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Sales'])
plt.title("Box Plot of Sales")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Profit'])
plt.title("Box Plot of Profit")
plt.ylabel("Profit")
plt.show()

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Sales',
    y='Profit',
    hue='Category',
    size='Discount'
)

plt.title("Sales, Profit, Category and Discount Analysis")
plt.show()

top_products = (
    df.groupby('Product Name')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 Products by Sales:")
print(top_products)

top_profit_products = (
    df.groupby('Product Name')['Profit']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 Products by Profit:")
print(top_profit_products)

print("Highest Sales Category:", category_sales.idxmax())
print("Highest Profit Category:", category_profit.idxmax())
print("Highest Sales Region:", region_sales.idxmax())

# Questions and Answers
# Q1. What is Exploratory Data Analysis (EDA), and why is it performed before machine learning?
# Answer: Exploratory Data Analysis (EDA) is the process of examining and understanding a dataset using statistical methods and visualizations. It helps identify patterns, trends, relationships, outliers, missing values, and unusual observations.

# EDA is performed before machine learning because it helps understand the data and identify potential problems before building a model. It also helps in selecting useful features and choosing appropriate preprocessing techniques.

# Q2. Differentiate between univariate, bivariate, and multivariate analysis with suitable examples.
# Answer:

# Univariate Analysis: It analyzes one variable at a time. Example: Studying the distribution of Sales using a histogram.

# Bivariate Analysis: It analyzes the relationship between two variables. Example: Studying the relationship between Sales and Profit using a scatter plot.

# Multivariate Analysis: It analyzes three or more variables together. Example: Studying Sales, Profit, Category, and Discount simultaneously.

# Q3. What insights can be obtained from a correlation heatmap?
# Answer: A correlation heatmap shows the strength and direction of relationships between numerical variables. Correlation values generally range from -1 to +1.

# A value close to +1 indicates a strong positive relationship.
# A value close to -1 indicates a strong negative relationship.
# A value close to 0 indicates a weak or no linear relationship.
# It can help identify strongly related features and possible redundant variables.

# Q4. Explain the purpose of histograms, box plots, and scatter plots in EDA.
# Answer:

# Histogram: Shows the distribution and frequency of numerical data. It helps understand the spread and shape of the data.
# Box Plot: Shows the median, quartiles, spread, and potential outliers in numerical data.
# Scatter Plot: Shows the relationship between two numerical variables and helps identify trends, patterns, and unusual observations.
# Q5. How can EDA help identify data quality issues before analysis?
# Answer: EDA helps identify data quality problems such as missing values, duplicate records, incorrect data types, unusual values, outliers, and inconsistent categories.

# Functions such as info(), isnull(), describe(), and value_counts() along with graphs help analysts detect these problems before performing further analysis.

# Q6. Why is correlation important in predictive analytics? Can correlation imply causation?
# Answer: Correlation is important because it helps identify the strength and direction of relationships between variables. It can help in feature selection and understanding which variables may be useful for prediction.

# However, correlation does not imply causation. Two variables may be correlated without one causing the other. A third factor or coincidence may explain the relationship.

# Q7. Which visualization would you use to analyze categorical and numerical variables? Justify your choice.
# Answer: For categorical variables, bar charts and count plots are useful because they show the frequency or count of different categories.

# For numerical variables, histograms and box plots are useful because they show the distribution, spread, central tendency, and outliers.

# For relationships between two numerical variables, a scatter plot is suitable.

# Q8. What business insights can be derived from the Superstore dataset through EDA?
# Answer: EDA of the Superstore dataset can provide several useful business insights, such as:

# Which category generates the highest sales.
# Which category generates the highest profit.
# Which region has the highest sales.
# Which products are top performers.
# The relationship between sales and profit.
# The presence of high discounts or unusual transactions.
# Which areas may require improvement in terms of profitability.
# These insights can help businesses improve sales strategies and resource allocation.

# Q9. How does EDA contribute to feature selection and model building?
# Answer: EDA helps identify important features by studying their distributions, correlations, and relationships with the target variable. It can also reveal irrelevant or highly correlated features.

# The insights obtained from EDA help in selecting useful features, choosing preprocessing techniques, detecting data problems, and building better machine learning models.

# Q10. What challenges might arise while performing EDA on large-scale real-world datasets?
# Answer: Some common challenges are:

# Large data size – Processing huge datasets can require significant memory and computing power.
# Missing and inconsistent data – Real-world datasets may contain many data quality problems.
# High number of features – Understanding hundreds or thousands of variables can be difficult.
# Outliers and noise – Large datasets may contain many unusual observations.
# Visualization limitations – Plotting millions of records can be slow and difficult to interpret.
# Therefore, sampling, efficient data processing, and appropriate visualization techniques may be required.
