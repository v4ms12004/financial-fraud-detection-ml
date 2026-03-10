import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("transaction_data.csv")

counts = df['isFraud'].value_counts()

labels = ['Non-Fraud','Fraud']

plt.figure(figsize=(5,5))
plt.pie(counts, labels=labels, autopct='%1.3f%%', startangle=90)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()