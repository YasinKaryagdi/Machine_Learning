#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

df = pd.read_csv('newcard.csv')


# In[5]:


# Shuffle the DataFrame randomly
df_shuffled = df.sample(frac=1, random_state=42)  # Using a fixed random state for reproducibility

# Get the total number of rows
total_rows = len(df_shuffled)

# Split the DataFrame into 50 equal parts
num_parts = 50
rows_per_part = total_rows // num_parts
remainder = total_rows % num_parts

# Calculate the start and end indices for each part
start_idx = 0
data_parts = []
for i in range(num_parts):
    end_idx = start_idx + rows_per_part
    if i < remainder:
        end_idx += 1
    data_parts.append(df_shuffled.iloc[start_idx:end_idx])
    start_idx = end_idx

# Write each part into separate CSV files
for i, part in enumerate(data_parts):
    part.to_csv(f'dataset_{i+1}.csv', index=False)


# In[ ]:




