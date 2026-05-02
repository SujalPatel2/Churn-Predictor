from data_loader import load_data, get_summary

df = load_data()
summary = get_summary(df)

print(df.shape)        # Should print (7032, 20)
print(df.head())       # First 5 rows
print(summary)         # Churn summary stats