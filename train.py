import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import os

input_path = sys.argv[1]
model_path = sys.argv[2]

print(f"📥 Loading processed data from {input_path}...")
df = pd.read_csv(input_path)

X = df[['feature1', 'feature2']]
y = df['label']

model = LogisticRegression()
model.fit(X, y)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model trained and saved to {model_path}")
