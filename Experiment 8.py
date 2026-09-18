# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 8

# Experiment Name
# Customer Churn Prediction using Decision Tree Classification

# Aim
# To build and evaluate a Decision Tree Classification model for predicting customer churn using a real-world dataset and analyze the factors influencing customer retention.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from google.colab import files

uploaded = files.upload()

import io

filename = list(uploaded.keys())[0]

df = pd.read_csv(io.BytesIO(uploaded[filename]))

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

print("Dataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
display(df.describe(include="all"))

df = df.drop("CustomerID", axis=1)

# Check missing values
print("Missing values before preprocessing:")
print(df.isnull().sum())

# Handle missing values
df = df.dropna()

print("\nDataset after preprocessing:")
display(df.head())

label_encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Contract",
    "InternetService",
    "TechSupport",
    "PaymentMethod",
    "Churn"
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("Categorical variables encoded successfully!")

display(df.head())

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Features:")
print(X.columns)

print("\nTarget Variable:")
print("Churn")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("Decision Tree Model Trained Successfully!")

y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("Model Evaluation Results")
print("------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()

plt.figure(figsize=(22, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Visualization")
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance:")
display(importance)

plt.figure(figsize=(10, 5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Customer Churn Prediction")

plt.tight_layout()
plt.show()

new_customer = pd.DataFrame([{
    "Gender": 1,
    "SeniorCitizen": 0,
    "TenureMonths": 5,
    "MonthlyCharges": 95.0,
    "Contract": 0,
    "InternetService": 1,
    "TechSupport": 0,
    "PaymentMethod": 0
}])

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")

df.head()

# QUESTION AND ANSWER
# Q1. What is classification, and how does it differ from regression?
# Answer: Classification is a supervised machine learning technique used to predict categorical output classes, such as Yes or No. Regression predicts continuous numerical values, such as salary or house price.

# Q2. Explain the working principle of the Decision Tree algorithm.
# Answer: A Decision Tree divides data into smaller groups using feature-based conditions. It starts from a root node and creates branches based on splitting criteria. The process continues until leaf nodes produce predictions.

# Q3. Differentiate between Gini Index and Entropy.
# Answer: Gini Index measures impurity based on the probability of incorrect classification. Entropy measures the uncertainty or disorder in data using information theory. Both can be used to select suitable splits in a Decision Tree.

# Q4. What is a Confusion Matrix?
# Answer: A Confusion Matrix evaluates classification predictions using four components: True Positive, True Negative, False Positive, and False Negative. It helps identify correct and incorrect predictions.

# Q5. Define Accuracy, Precision, Recall, and F1-Score.
# Answer:

# Accuracy: Proportion of all predictions that are correct.
# Precision: Proportion of predicted positive cases that are actually positive.
# Recall: Proportion of actual positive cases correctly identified.
# F1-Score: Harmonic mean of precision and recall.
# These metrics help evaluate different aspects of model performance.

# Q6. What is overfitting in a Decision Tree?
# Answer: Overfitting occurs when a Decision Tree learns training data too closely, including noise, and performs poorly on unseen data. It can be reduced by limiting tree depth, increasing minimum samples per leaf, and using pruning.

# Q7. Why is customer churn prediction important?
# Answer: Customer churn prediction helps organizations identify customers who may leave their services. Businesses can use these insights to develop retention strategies, improve customer satisfaction, and reduce potential revenue loss.

# Q8. What are the advantages and limitations of Decision Trees?
# Answer:

# Advantages:

# Easy to understand and interpret.
# Can handle numerical and categorical features after suitable preprocessing.
# Requires relatively little data preparation.
# Limitations:

# Can overfit training data.
# Small data changes can produce different trees.
# A single tree may have lower predictive performance than some ensemble methods.
# Q9. Mention three real-world applications of Decision Tree Classification.
# Answer:

# Medical disease classification.
# Loan approval and credit risk assessment.
# Email spam detection.
# Q10. How can churn prediction improve customer retention?
# Answer: Organizations can identify customers who may leave and analyze the factors associated with churn. They can then offer suitable support, improve services, and develop targeted retention programs. This may help improve customer loyalty and business profitability.
