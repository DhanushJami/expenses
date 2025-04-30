import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

file_path = 'expense_data_with_nans.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

print("Dimensions of the dataset:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nData types of columns:")
print(df.dtypes)

print("\nMissing values in each column:")
print(df.isnull().sum())

df.dropna(thresh=len(df.columns)//2, inplace=True)

amount_mean_value = df['Amount'].mean()
df.fillna(value=amount_mean_value, inplace=True)

for col in df.select_dtypes(include='object'):
    df[col] = df[col].astype(str)

df.drop_duplicates(subset=['Description'], keep='first', inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)

print("\nBasic statistics of the dataset:")
print(df.describe())

amount_mean = df['Amount'].mean()
print("\nMean for Amount:", amount_mean)

total_expense = df['Amount'].sum()
print("Total Expense:", total_expense)

category_expense = df.groupby('Category')['Amount'].sum()
print("\nTotal Expense by Category:")
print(category_expense)

print("\nTop 5 Highest Expenses:")
print(df.sort_values(by='Amount', ascending=False).head(5))

plt.figure(figsize=(16, 12))

plt.subplot(3, 2, 1)
plt.hist(df['Amount'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.grid(True)

category_counts = df['Category'].value_counts()
plt.subplot(3, 2, 2)
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Expense Distribution by Category')
plt.legend(category_counts.index, loc="best")

plt.subplot(3, 2, 3)
df_sorted = df.sort_values(by='Date')
plt.plot(df_sorted['Date'], df_sorted['Amount'], color='green')
plt.title('Total Expense over Time')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.grid(True)

plt.subplot(3, 2, 4)
category_expense.plot(kind='bar', color='salmon')
plt.title('Total Expense by Department (Category)')
plt.xlabel('Category')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()
