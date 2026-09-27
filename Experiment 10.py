# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 10

# Experiment Name
# End-to-End Data Analytics Project: Business Intelligence and Predictive Analytics using a Real-World Dataset

# Aim
# To implement a complete data analytics workflow on a real-world dataset by performing data collection, preprocessing, exploratory data analysis, visualization, statistical analysis, predictive modeling, and presentation of actionable business insights.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

url = "https://raw.githubusercontent.com/yannie28/Global-Superstore/master/Global_Superstore(CSV).csv"

df = pd.read_csv(url, encoding="utf-8-sig")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

df.head()

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

# Convert numeric columns
for col in ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicate records
df = df.drop_duplicates()

# Remove missing values from important columns
df = df.dropna(
    subset=["Sales", "Quantity", "Discount", "Profit"]
)

# Create shipping days
df["Shipping Days"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

# Remove invalid shipping days
df = df[df["Shipping Days"] >= 0]

print("Preprocessing completed!")
print("New Shape:", df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

display(
    df[
        ["Sales", "Quantity", "Discount",
         "Profit", "Shipping Cost"]
    ].describe().round(2)
)

# Sales by Category
plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Category",
    y="Sales",
    estimator="sum"
)

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.show()


# Sales Distribution
plt.figure(figsize=(8,5))

sns.histplot(
    df["Sales"],
    bins=30,
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

numeric_cols = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Shipping Cost",
    "Shipping Days"
]

corr = df[numeric_cols].corr()

plt.figure(figsize=(9,6))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

technology = df[
    df["Category"] == "Technology"
]["Sales"]

furniture = df[
    df["Category"] == "Furniture"
]["Sales"]

t_stat, p_value = stats.ttest_ind(
    technology,
    furniture,
    equal_var=False,
    nan_policy="omit"
)

print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 6))

if p_value < 0.05:
    print("Result: Significant difference exists.")
else:
    print("Result: No significant difference found.")

features = [
    "Quantity",
    "Discount",
    "Shipping Cost",
    "Shipping Days"
]

X = df[features]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression model trained successfully!")

print("\nCoefficients:")
for feature, coef in zip(features, model.coef_):
    print(feature, ":", round(coef, 4))

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print("----------------------")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 4))

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.show()

print("Average Sales:", round(df["Sales"].mean(), 2))
print("Average Profit:", round(df["Profit"].mean(), 2))
print("Model R²:", round(r2, 4))

print("""
BUSINESS RECOMMENDATIONS
------------------------

1. Use the predictive model to estimate future sales.
2. Monitor discount levels and their effect on sales.
3. Analyse product categories regularly.
4. Track sales, profit and shipping performance.
5. Improve the model by adding more relevant features.
6. Use dashboards to communicate important business insights.

CONCLUSION
----------

An end-to-end data analytics project was completed using
the Global Superstore dataset.

The project included data preprocessing, EDA,
visualization, statistical analysis, predictive modelling,
model evaluation and business recommendations.
""")


# Experiment 10 — Questions & Answers
# 1. What are the major stages involved in an end-to-end data analytics project?
# Answer: The major stages are data collection, data preprocessing, Exploratory Data Analysis (EDA), visualization, statistical analysis, machine learning, model evaluation, interpretation of results, and business recommendations. These stages help convert raw data into useful information for decision-making.

# 2. Why is data preprocessing considered the foundation of successful analytics and machine learning?
# Answer: Data preprocessing is important because real-world data may contain missing values, duplicate records, outliers, and categorical variables. Cleaning and transforming the data improves its quality and makes it suitable for analysis and machine learning models.

# 3. How does Exploratory Data Analysis (EDA) contribute to model development?
# Answer: EDA helps us understand the characteristics of the dataset using descriptive statistics and visualizations. It helps identify patterns, trends, relationships, unusual values, and important variables. These findings help in selecting suitable features and developing an appropriate model.

# 4. What factors should be considered while selecting a machine learning algorithm for a business problem?
# Answer: The algorithm should be selected according to the type of business problem, type of target variable, dataset size, available features, and required output. The complexity and performance of the algorithm should also be considered. For example, Linear Regression can be used for predicting a numerical value such as Sales.

# 5. Why is model evaluation essential before deploying a predictive model?
# Answer: Model evaluation helps determine how well a model performs on unseen data. It shows the accuracy and prediction errors of the model and helps identify whether the model is suitable for the business problem. For regression models, MAE, RMSE, and R² can be used for evaluation.

# 6. Explain how Business Intelligence tools complement machine learning in data analytics projects.
# Answer: Business Intelligence tools such as Tableau and Power BI help present analytical and machine-learning results using dashboards, charts, and interactive visualizations. They make complex results easier for managers and stakeholders to understand and help them use the insights for decision-making.

# 7. What challenges are commonly encountered while working with real-world datasets?
# Answer: Common challenges include missing values, duplicate records, outliers, categorical variables, inconsistent data, and large datasets. These problems can affect the quality of analysis and model performance, so proper preprocessing is required before analysis.

# 8. How can analytical insights be converted into actionable business recommendations?
# Answer: Analytical insights can be converted into recommendations by identifying important trends, opportunities, and risks from the data. These findings can then be connected with practical business actions, such as improving sales strategies, monitoring discounts, or improving operational performance.

# 9. Why is effective presentation and data storytelling important in analytics projects?
# Answer: Effective presentation and data storytelling make analytical findings easier to understand. They help communicate the methodology, important trends, visualizations, model performance, and recommendations clearly to stakeholders. This allows the results to support business strategy and decision-making.

# 10. Suggest future improvements or advanced techniques that could enhance the accuracy and effectiveness of the developed analytics solution.
# Answer: Future improvements can include using more relevant features, feature engineering, trying different machine-learning algorithms, hyperparameter tuning, and using larger or better-quality datasets. BI dashboards can also be improved to present results more effectively and support better decision-making.
