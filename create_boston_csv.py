import pandas as pd
import numpy as np
import os
import ssl
from sklearn.datasets import fetch_openml

# Disable SSL verification (temporary fix for certificate issue)
ssl._create_default_https_context = ssl._create_unverified_context

print("📥 Fetching Boston housing dataset from OpenML (SSL verification disabled)...")

# Fetch dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Save dataset to local file
os.makedirs("data", exist_ok=True)
df.to_csv("data/boston.csv", index=False)

print("✅ Boston dataset successfully created and saved to: data/boston.csv")
