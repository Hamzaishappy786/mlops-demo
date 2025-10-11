import pandas as pd
import sys
import os

# read input + output paths from command line args
input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"📥 Reading data from {input_path}...")
df = pd.read_csv(input_path)

# Just a dumb "preprocessing": normalize feature1 & feature2
df['feature1'] = df['feature1'] / df['feature1'].max()
df['feature2'] = df['feature2'] / df['feature2'].max()

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Saved cleaned data to {output_path}")