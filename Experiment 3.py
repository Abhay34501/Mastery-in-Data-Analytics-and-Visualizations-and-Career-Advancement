# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1
# Experiment No. 3

# Experiment Name
# Statistical Analysis and Hypothesis Testing using Python

# Aim
# To perform statistical analysis and hypothesis testing on a real-world dataset using Python in order to identify relationships between variables, validate assumptions, and support data-driven decision-making.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from scipy.stats import f_oneway

import statsmodels.api as sm
from statsmodels.formula.api import ols

print("All libraries imported successfully!")

from google.colab import files

uploaded = files.upload()

import io

file_name = list(uploaded.keys())[0]

df = pd.read_csv(io.BytesIO(uploaded[file_name]))

print("Dataset loaded successfully!")
print("File Name:", file_name)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

df.head()

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("Missing Values in Each Column:")
print(df.isnull().sum())

numeric_cols = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "YearsAtCompany",
    "TotalWorkingYears"
]

print("Mean:")
print(df[numeric_cols].mean())

print("\nMedian:")
print(df[numeric_cols].median())

print("\nMode:")
print(df[numeric_cols].mode().iloc[0])

print("\nVariance:")
print(df[numeric_cols].var())

print("\nStandard Deviation:")
print(df[numeric_cols].std())

df[numeric_cols].describe()

correlation = df[numeric_cols].corr(method="pearson")

print("Pearson Correlation Matrix:")
display(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Pearson Correlation Heatmap")
plt.show()

print("H0: There is no significant difference in Monthly Income")
print("between employees who left and employees who stayed.")

print("\nH1: There is a significant difference in Monthly Income")
print("between employees who left and employees who stayed.")

left = df[df["Attrition"] == "Yes"]["MonthlyIncome"]

stayed = df[df["Attrition"] == "No"]["MonthlyIncome"]

t_stat, p_value = ttest_ind(
    left,
    stayed,
    equal_var=False
)

print("Independent Sample t-test")
print("--------------------------")
print("t-statistic:", t_stat)
print("p-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Decision: Reject H0")
    print("Conclusion: There is a statistically significant difference")
    print("in Monthly Income between the two groups.")
else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: There is no statistically significant difference")
    print("in Monthly Income between the two groups.")
  job_roles = df["JobRole"].unique()

groups = []

for role in job_roles:
    group = df[df["JobRole"] == role]["MonthlyIncome"]
    groups.append(group)

f_stat, p_value_anova = f_oneway(*groups)

print("One-Way ANOVA")
print("-------------")
print("F-statistic:", f_stat)
print("p-value:", p_value_anova)

alpha = 0.05

if p_value_anova < alpha:
    print("Decision: Reject H0")
    print("Conclusion: There is a statistically significant")
    print("difference in Monthly Income among Job Roles.")
else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: There is no statistically significant")
    print("difference in Monthly Income among Job Roles.")

X = df[["TotalWorkingYears"]]

y = df["MonthlyIncome"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

print("Regression Coefficients:")
print(model.params)

print("R² Score:", model.rsquared)

confidence_interval = model.conf_int()

print("95% Confidence Intervals:")
display(confidence_interval)

plt.figure(figsize=(9, 6))

sns.regplot(
    data=df,
    x="TotalWorkingYears",
    y="MonthlyIncome"
)

plt.title("Linear Regression: Total Working Years vs Monthly Income")
plt.xlabel("Total Working Years")
plt.ylabel("Monthly Income")

plt.show()

print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("\n1. Dataset:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. T-Test:")
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Result: Significant difference found.")
else:
    print("Result: No significant difference found.")

print("\n3. ANOVA:")
print("F-statistic:", f_stat)
print("p-value:", p_value_anova)

if p_value_anova < 0.05:
    print("Result: Significant difference among Job Roles.")
else:
    print("Result: No significant difference among Job Roles.")

print("\n4. Linear Regression:")
print("R² Score:", model.rsquared)

print("\nExperiment Completed Successfully!")

# VIVA / ASSIGNMENT QUESTIONS — ANSWERS

# Q1. What is the difference between descriptive statistics and inferential statistics?
# Answer: Descriptive statistics are used to summarize, organize, and describe the important characteristics of a given dataset. They provide a simple representation of the data using measures such as mean, median, mode, variance, standard deviation, minimum, and maximum values. Descriptive statistics only describe the data that has been collected and do not make predictions about a larger population.

# Q2. Explain Null Hypothesis (H₀) and Alternative Hypothesis (H₁).
# Answer:The Alternative Hypothesis (H₁) is a statement that suggests that a significant difference, effect, or relationship exists between the variables.

# During hypothesis testing, statistical tests are performed to determine whether there is enough evidence to reject the null hypothesis. If the evidence is strong enough, H₀ is rejected in favor of H₁. The Null Hypothesis (H₀) is a statement that assumes there is no significant difference, effect, or relationship between the variables being studied. It represents the default assumption that researchers try to test using statistical evidence. H₀ (Null Hypothesis): States that there is no significant difference or relationship between variables.

# H₁ (Alternative Hypothesis): States that a significant difference or relationship exists.

# Q3. What is a p-value?
# Answer:A p-value is a statistical value that helps determine whether the observed result provides enough evidence against the null hypothesis. It represents the probability of obtaining a result as extreme as the observed result, assuming that the null hypothesis is true.

# The p-value is commonly compared with a significance level (α), which is often set to 0.05.

# A p-value indicates the probability of obtaining the observed result if the null hypothesis is true. Generally, if p < 0.05, H₀ is rejected; if p ≥ 0.05, H₀ is not rejected.

# Q4. Differentiate between t-test and ANOVA.
# Answer:A t-test and ANOVA (Analysis of Variance) are statistical hypothesis-testing methods used to compare group means.

# A t-test is generally used when we want to compare the means of two groups. For example, we can use a t-test to compare the average income of employees who left a company with the average income of employees who stayed.

# ANOVA is used when we want to compare the means of three or more groups. For example, ANOVA can be used to determine whether the average income is significantly different among employees working in different job roles. t-test compares two groups, while ANOVA compares three or more groups. A t-test produces a t-statistic, whereas ANOVA produces an F-statistic.

# Example: t-test: Comparing income of two groups. ANOVA: Comparing income across different job roles.

# Q5. What does Pearson correlation coefficient indicate?
# Answer:The Pearson correlation coefficient measures the strength and direction of the linear relationship between two numerical variables. It is represented by the symbol r and its value ranges from -1 to +1.

# A positive correlation means that when one variable increases, the other variable tends to increase as well. A negative correlation means that when one variable increases, the other tends to decrease.

# The interpretation is generally: Pearson correlation measures the strength and direction of a linear relationship between two numerical variables. Its value ranges from -1 to +1.

# +1 → Perfect positive correlation

# 0 → No linear correlation

# -1 → Perfect negative correlation

# Q6. Explain R² in Linear Regression.
# Answer:R², also known as the coefficient of determination, is an important measure used in linear regression to evaluate how well the independent variable(s) explain the variation in the dependent variable.

# R² represents the proportion of variation in the dependent variable that is explained by the regression model. Its value generally ranges from 0 to 1.

# For example, if a regression model has an R² value of 0.80, it means that approximately 80% of the variation in the dependent variable is explained by the variables included in the model, while the remaining variation is due to other factors and random variation. R², or coefficient of determination, represents the proportion of variation in the dependent variable explained by the independent variable. Its value generally ranges from 0 to 1. A higher R² indicates that the model explains more of the variation.

# Q7. Why is statistical analysis important before Machine Learning?
# Answer:Statistical analysis is important before building a Machine Learning model because it helps us understand the dataset and identify important patterns and problems within the data.

# It helps in understanding the distribution of variables, relationships between features, missing values, outliers, unusual observations, and data variability. Statistical techniques can also help determine whether certain variables are strongly related and whether some features may be useful for prediction.

# Statistical analysis can also help check assumptions and select appropriate preprocessing techniques before training a model. Statistical analysis helps understand the dataset, identify relationships, detect unusual values, check assumptions and determine important variables before building a machine-learning model.

# Q8. What assumptions should be satisfied before t-test or ANOVA?
# Answer:Before applying a t-test or ANOVA, certain important statistical assumptions should generally be checked to ensure that the results are reliable.

# Important assumptions include:

# Independence of observations.
# Approximately normally distributed data within groups.
# Homogeneity of variance between groups.
# Appropriate measurement scale for the variables.
# Appropriate measurement scale: The dependent variable should generally be numerical and measured on an appropriate scale.
# Random or appropriate sampling: The sample should be collected appropriately so that the statistical conclusions are meaningful.
# Q9. How can regression help organizations?
# Answer:Regression analysis helps organizations understand relationships between variables and make predictions about future outcomes. It is widely used in areas such as business, finance, marketing, sales, human resources, and forecasting.

# For example, a company can use regression to estimate an employee's income based on factors such as years of experience, education, and job role. Similarly, a business can use regression to forecast future sales based on advertising expenditure, customer demand, and previous sales data.

# Regression can also help organizations identify which factors have a strong relationship with a target variable. This information can support better decision-making, planning, forecasting, and resource allocation

# Q10. Give two real-world applications of hypothesis testing.
# Answer:Hypothesis testing is used in many real-world situations to determine whether an observed difference or relationship is statistically significant.

# Business: A company can use hypothesis testing to determine whether a new marketing campaign significantly increases sales compared with the previous marketing strategy. The null hypothesis may state that the new campaign has no significant effect on sales, while the alternative hypothesis states that it has a significant effect.

# Human Resources: An organization can use hypothesis testing to determine whether employee income differs significantly between employees who leave the organization and those who stay. This can help the HR department understand whether compensation may be associated with employee turnover.

# Business: Testing whether a new marketing campaign significantly increases sales.

# Thus, hypothesis testing helps organizations make decisions based on statistical evidence rather than assumptions or observations alone.
