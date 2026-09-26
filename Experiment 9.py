# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 9

# Experiment Name
# Data Storytelling and Business Insight Generation using Interactive Visualizations

# Aim
# To create a compelling data-driven story by analyzing a real-world dataset, developing meaningful visualizations, and presenting actionable business insights to support strategic decision-making.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid")

url = "https://raw.githubusercontent.com/yannie28/Global-Superstore/master/Global_Superstore(CSV).csv"

df = pd.read_csv(url, encoding="utf-8-sig")

print("Dataset Loaded Successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

df.head()

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("Missing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

for col in ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates()

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

print("Data Cleaning Completed!")
print("New Shape:", df.shape)

df[[
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Shipping Cost"
]].describe().round(2)

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()

profit_margin = (total_profit / total_sales) * 100
average_order_value = total_sales / total_orders

print("========== KPIs ==========")
print("Total Sales       :", round(total_sales, 2))
print("Total Profit      :", round(total_profit, 2))
print("Total Orders      :", total_orders)
print("Total Customers   :", total_customers)
print("Average Order Value:", round(average_order_value, 2))
print("Profit Margin     :", round(profit_margin, 2), "%")

yearly = df.groupby("Year")[["Sales", "Profit"]].sum().reset_index()

display(yearly.round(2))

plt.figure(figsize=(10,5))

plt.plot(yearly["Year"], yearly["Sales"],
         marker="o", label="Sales")

plt.plot(yearly["Year"], yearly["Profit"],
         marker="o", label="Profit")

plt.title("Yearly Sales and Profit")
plt.xlabel("Year")
plt.ylabel("Amount")
plt.legend()
plt.show()

region = df.groupby("Region")[["Sales", "Profit"]].sum().reset_index()

display(region.round(2))

plt.figure(figsize=(10,5))

sns.barplot(data=region, x="Region", y="Sales")

plt.title("Sales by Region")
plt.xticks(rotation=45)
plt.show()

category = df.groupby("Category")[["Sales", "Profit"]].sum().reset_index()

display(category.round(2))

category.plot(
    x="Category",
    y=["Sales", "Profit"],
    kind="bar",
    figsize=(10,5)
)

plt.title("Sales and Profit by Category")
plt.ylabel("Amount")
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df.sample(min(5000, len(df)), random_state=42),
    x="Discount",
    y="Profit"
)

plt.title("Discount vs Profit")
plt.show()


# Correlation Heatmap

plt.figure(figsize=(8,6))

corr = df[
    ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

best_region = region.loc[region["Sales"].idxmax()]
best_category = category.loc[category["Sales"].idxmax()]
profit_category = category.loc[category["Profit"].idxmax()]
worst_region = region.loc[region["Profit"].idxmin()]

print("========== KEY INSIGHTS ==========\n")

print("1. Highest Sales Region:",
      best_region["Region"])

print("2. Highest Sales Category:",
      best_category["Category"])

print("3. Highest Profit Category:",
      profit_category["Category"])

print("4. Lowest Profit Region:",
      worst_region["Region"])

print("\nOverall Profit Margin:",
      round(profit_margin, 2), "%")

print("""
========== RECOMMENDATIONS ==========

1. Focus on regions generating strong sales and profit.

2. Identify and review loss-making products or categories.

3. Monitor discount levels because high discounts may reduce profit.

4. Use monthly and yearly sales trends for better planning.

5. Track KPIs such as Sales, Profit, Orders and Profit Margin regularly.

========== CONCLUSION ==========

Data storytelling combines data analysis, visualization and
narrative to communicate meaningful business insights.

The Global Superstore dataset was analyzed using Python,
Pandas, Matplotlib and Seaborn.

The analysis identified sales trends, regional performance,
category performance, profitability and the relationship between
discount and profit.

These insights can support business planning and decision-making.
""")


# Experiment 9 — Question Answers
# Q1. What is data storytelling? How is it different from data visualization?
# Answer: Data storytelling is the process of combining data, visualization, and narrative to communicate meaningful insights effectively. Data visualization mainly represents data using charts and graphs, whereas data storytelling explains the meaning and importance of those visualizations and connects them with business decisions.

# Q2. Why is storytelling important in business analytics and decision-making?
# Answer: Storytelling makes complex analytical results easier to understand. It helps stakeholders identify important trends, problems, opportunities, and risks. It also connects data findings with business objectives and supports better decision-making.

# Q3. What are the essential components of an effective data story?
# Answer: The essential components are:

# Clear business problem or objective
# Relevant and reliable data
# Data analysis
# Meaningful visualizations
# Key findings and insights
# Context and explanation
# Actionable recommendations
# Q4. How do Key Performance Indicators (KPIs) enhance business reporting?
# Answer: KPIs provide a quick summary of important business performance measures. Examples include Sales, Profit, Number of Orders, Average Order Value, and Profit Margin. They help managers monitor performance, compare results, identify gaps, and make informed decisions.

# Q5. Why should visualizations be arranged in a logical sequence while presenting analytical findings?
# Answer: Visualizations should be arranged logically so that the audience can easily follow the data story. A proper sequence can move from problem → data → analysis → findings → recommendations. This makes the presentation clear, organized, and easier to understand.

# Q6. What factors should be considered while selecting visualizations for a business presentation?
# Answer: The following factors should be considered:

# Type of data
# Purpose of analysis
# Comparison required
# Number of categories
# Audience
# Readability
# Simplicity
# Business relevance
# For example, line charts are useful for trends, bar charts for comparisons, and heatmaps for relationships or correlations.

# Q7. Explain how dashboards and storytelling complement each other in Business Intelligence.
# Answer: Dashboards provide an interactive view of important KPIs, trends, and business metrics. Storytelling explains the meaning of these results and highlights the most important findings. Therefore, dashboards provide the visual evidence, while storytelling provides the context and explanation.

# Q8. What challenges may arise while communicating analytical insights to non-technical stakeholders?
# Answer: Some common challenges are:

# Complex technical terminology
# Too many charts or excessive information
# Lack of business context
# Difficulty understanding statistical concepts
# Poorly designed visualizations
# Difficulty identifying the most important finding
# Analysts should use simple language, clear charts, and focus on business-relevant insights.

# Q9. Give two real-world examples where data storytelling has influenced business or policy decisions.
# Answer:

# Example 1 — Retail: Retail companies use sales dashboards and data stories to identify product demand, sales trends, and regional performance. This information can support decisions about inventory, promotions, and product planning.

# Example 2 — Public Health: Public-health organizations use data dashboards and visual stories to communicate disease trends, cases, and resource requirements. Such information can support planning and allocation of resources.

# Q10. How can effective data storytelling improve strategic planning and organizational performance?
# Answer: Effective data storytelling helps organizations understand their current performance, identify trends and performance gaps, and communicate findings clearly to stakeholders. It supports better planning, resource allocation, monitoring, and decision-making, which can contribute to improved organizational performance.
