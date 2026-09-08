# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 6

# Experiment Name
# Predictive Analytics using Linear Regression and Model Performance Evaluation

# Aim
# To develop a Linear Regression model using Python for predicting continuous outcomes and evaluate its performance using appropriate regression metrics.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8,5)

np.random.seed(42)

n = 200
TV = np.round(np.random.uniform(5, 300, n), 2)
Radio = np.round(np.random.uniform(0, 50, n), 2)
Newspaper = np.round(np.random.uniform(0, 100, n), 2)

# Sales depends mainly on TV & Radio with some noise
Sales = 4.5 + 0.045*TV + 0.19*Radio + 0.01*Newspaper + np.random.normal(0, 2, n)
Sales = np.round(Sales, 2)

df = pd.DataFrame({
    'TV': TV,
    'Radio': Radio,
    'Newspaper': Newspaper,
    'Sales': Sales
})

print(df.shape)
df.head()

print(df.isnull().sum())
print(df.describe())

# No missing values, no categorical variables here.
# Selecting features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))

y_pred = model.predict(X_test)

results = pd.DataFrame({'Actual': y_test.values, 'Predicted': np.round(y_pred, 2)})
results.head(10)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='teal', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.tight_layout()
plt.show()

X_tv = df[['TV']]
y_tv = df['Sales']

model_tv = LinearRegression()
model_tv.fit(X_tv, y_tv)
y_tv_pred = model_tv.predict(X_tv)

plt.figure(figsize=(8,6))
plt.scatter(X_tv, y_tv, color='skyblue', alpha=0.6, label='Actual Data')
plt.plot(X_tv, y_tv_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.title('Regression Line: TV Budget vs Sales')
plt.legend()
plt.tight_layout()
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MODEL PERFORMANCE METRICS")
print("="*40)
print(f"Mean Absolute Error (MAE)   : {mae:.3f}")
print(f"Mean Squared Error (MSE)    : {mse:.3f}")
print(f"Root Mean Squared Error    : {rmse:.3f}")
print(f"R² Score                    : {r2:.3f}")

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print(coef_df)

plt.figure(figsize=(7,5))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.title('Impact of Each Feature on Sales')
plt.tight_layout()
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Distribution of Residuals')
plt.xlabel('Residual (Actual - Predicted)')
plt.tight_layout()
plt.show()

print("INTERPRETATION SUMMARY")
print("="*50)
print(f"R² Score of {r2:.2f} indicates that the model explains {r2*100:.1f}% "
      f"of the variance in Sales.")
print(f"RMSE of {rmse:.2f} shows the average prediction error in the same units as Sales.")
top_feature = coef_df.iloc[0]['Feature']
print(f"'{top_feature}' has the strongest positive impact on Sales based on coefficient magnitude.")
print("Linear Regression appears suitable here as the relationship between "
      "advertising spend and sales is approximately linear.")

# Question & Answer Section

# Q1. What is Predictive Analytics, and how is it used in real-world applications?

# Predictive analytics uses historical data, statistical algorithms, and machine learning techniques to forecast future outcomes or trends. It's widely used in business for sales forecasting, in finance for credit risk assessment and fraud detection, in healthcare for predicting disease risk, and in marketing for customer churn prediction — helping organizations make proactive, data-driven decisions.

# Q2. Explain the working principle of the Linear Regression algorithm.

# Linear Regression models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a straight line (or hyperplane in multiple dimensions) that minimizes the sum of squared differences between actual and predicted values. It estimates coefficients for each feature using the Ordinary Least Squares (OLS) method, producing an equation like y = b0 + b1x1 + b2x2 + ... + bn*xn that can then predict new outcomes.

# Q3. Differentiate between dependent and independent variables with suitable examples.

# The dependent variable (also called target or outcome) is the value being predicted — for example, Sales. Independent variables (also called features or predictors) are the inputs used to make that prediction — for example, TV advertising budget, Radio budget, and Newspaper budget. Changes in independent variables are assumed to influence the dependent variable, not the other way around.

# Q4. Why is it necessary to split the dataset into training and testing sets?

# Splitting the data ensures the model is evaluated on data it hasn't seen during training, giving an honest measure of how well it generalizes to new, unseen data. Without this split, a model could simply memorize the training data (overfitting) and appear highly accurate while actually performing poorly on real-world predictions.

# Q5. What is the significance of the R² Score in regression analysis?

# The R² Score (coefficient of determination) indicates the proportion of variance in the dependent variable that is explained by the independent variables, ranging from 0 to 1 (or negative for very poor models). An R² of 0.85, for instance, means 85% of the variability in the target variable is explained by the model — higher values indicate a better fit.

# Q6. Differentiate between MAE, MSE, and RMSE. Which metric is more sensitive to large prediction errors?

# MAE (Mean Absolute Error) calculates the average of absolute differences between actual and predicted values, treating all errors equally. MSE (Mean Squared Error) squares the errors before averaging, which penalizes larger errors more heavily. RMSE (Root Mean Squared Error) is the square root of MSE, bringing the error back to the same unit as the target variable while still being sensitive to large errors. MSE (and consequently RMSE) is more sensitive to large prediction errors because squaring amplifies bigger deviations disproportionately compared to MAE.

# Q7. What assumptions should be satisfied before applying Linear Regression?

# Key assumptions include: a linear relationship between independent and dependent variables, independence of errors (no autocorrelation), homoscedasticity (constant variance of residuals across all levels of the independent variables), normality of residuals, and no or minimal multicollinearity among independent variables.

# Q8. How can overfitting and underfitting affect the performance of a regression model?

# Overfitting occurs when a model learns the training data too precisely, including noise, resulting in excellent training performance but poor generalization to new data. Underfitting occurs when the model is too simple to capture the underlying pattern, leading to poor performance on both training and testing data. Both reduce the model's real-world predictive reliability — the goal is a balanced model that generalizes well.

# Q9. Mention any three real-world applications of Linear Regression in business or industry.

# Sales forecasting — predicting future sales based on advertising spend, seasonality, or economic indicators.

# Real estate pricing — estimating house prices based on features like area, location, and number of rooms.

# Risk assessment in insurance/finance — predicting claim amounts or credit risk scores based on customer attributes.

# Q10. How can feature selection improve the accuracy and interpretability of a predictive model?

# Feature selection removes irrelevant, redundant, or noisy variables from the model, which reduces overfitting, improves computational efficiency, and often increases predictive accuracy. It also makes the model more interpretable by focusing only on the features that genuinely impact the target variable, making it easier for stakeholders to understand which factors drive the predicted outcome.
