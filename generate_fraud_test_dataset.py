import pandas as pd
df = pd.read_csv("creditcard.csv")
fraud_df = df[df['Class'] == 1].sample(5, random_state=42)
fraud_df.to_csv("data/synthetic_fraud.csv", index=False)
