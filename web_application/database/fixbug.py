import pandas as pd

# Load CSV file
df = pd.read_csv('owners_sample.csv')

# Drop duplicate plates, keep first occurrence
df_unique = df.drop_duplicates(subset='plate', keep='first')

# Save the result to a new CSV file
df_unique.to_csv('owners_sample.csv', index=False)
