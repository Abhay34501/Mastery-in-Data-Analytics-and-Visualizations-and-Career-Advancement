# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 4

# Experiment Name
# Advanced Data Visualization using Matplotlib and Seaborn

# Aim
# To create and analyze various data visualizations using Python libraries such as Matplotlib and Seaborn in order to identify trends, patterns, correlations, and insights from real-world datasets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

np.random.seed(42)

n = 500
regions = ['East', 'West', 'North', 'South']
categories = ['Furniture', 'Office Supplies', 'Technology']
sub_categories = ['Chairs', 'Tables', 'Phones', 'Binders', 'Storage', 'Accessories']

dates = pd.date_range(start='2022-01-01', end='2023-12-31', periods=n)

df = pd.DataFrame({
    'Order_Date': dates,
    'Region': np.random.choice(regions, n),
    'Category': np.random.choice(categories, n),
    'Sub_Category': np.random.choice(sub_categories, n),
    'Sales': np.round(np.random.gamma(shape=2, scale=150, size=n), 2),
    'Quantity': np.random.randint(1, 15, n),
    'Discount': np.round(np.random.uniform(0, 0.5, n), 2),
    'Profit': np.round(np.random.normal(50, 80, n), 2),
    'Customer_Age': np.random.randint(18, 70, n)
})

df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)

print(df.shape)
df.head()

df.info()
df.describe()

region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=region_sales.index, y=region_sales.values, palette='viridis')
plt.title('Total Sales by Region', fontsize=14, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.show()

monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(12,5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='teal', linewidth=2)
plt.title('Monthly Sales Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Sales'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Sales', fontsize=14, fontweight='bold')
plt.xlabel('Sales ($)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Category', y='Sales', data=df, palette='Set2')
plt.title('Sales Distribution by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category')
plt.ylabel('Sales ($)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Discount', y='Profit', hue='Category', data=df, palette='deep', s=60)
plt.title('Discount vs Profit', fontsize=14, fontweight='bold')
plt.xlabel('Discount')
plt.ylabel('Profit ($)')
plt.legend(title='Category')
plt.tight_layout()
plt.show()

numeric_cols = df[['Sales', 'Quantity', 'Discount', 'Profit', 'Customer_Age']]
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sales by sub-category
sub_sales = df.groupby('Sub_Category')['Sales'].sum().sort_values()
axes[0,0].barh(sub_sales.index, sub_sales.values, color='coral')
axes[0,0].set_title('Sales by Sub-Category')
axes[0,0].set_xlabel('Total Sales ($)')

# Quantity distribution
axes[0,1].hist(df['Quantity'], bins=14, color='mediumseagreen', edgecolor='black')
axes[0,1].set_title('Quantity Distribution')
axes[0,1].set_xlabel('Quantity')

# Profit by region boxplot
sns.boxplot(x='Region', y='Profit', data=df, ax=axes[1,0], palette='pastel')
axes[1,0].set_title('Profit by Region')

# Customer age distribution
sns.kdeplot(df['Customer_Age'], fill=True, color='purple', ax=axes[1,1])
axes[1,1].set_title('Customer Age Distribution')

plt.suptitle('Superstore Dataset — Summary Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Auto-generated insights based on actual data
top_region = region_sales.idxmax()
top_region_sales = region_sales.max()

trend = "increasing" if monthly_sales.iloc[-1] > monthly_sales.iloc[0] else "decreasing"

category_variance = df.groupby('Category')['Sales'].std().idxmax()

corr_sales_profit = corr_matrix.loc['Sales', 'Profit']
corr_discount_profit = corr_matrix.loc['Discount', 'Profit']

print("KEY INSIGHTS")
print("=" * 50)
print(f"1. {top_region} generated the highest total sales (${top_region_sales:,.2f}), suggesting stronger market demand there.")
print(f"2. Sales show a {trend} trend across the two-year period.")
print(f"3. Most transactions fall within a moderate sales range, with a few high-value outliers.")
print(f"4. {category_variance} shows the widest spread in sales, indicating inconsistent order values.")
print(f"5. Correlation between Discount and Profit is {corr_discount_profit:.2f}, "
      f"suggesting {'a negative relationship — higher discounts reduce profit' if corr_discount_profit < 0 else 'no strong negative impact'}.")
print(f"6. Correlation between Sales and Profit is {corr_sales_profit:.2f}.")
