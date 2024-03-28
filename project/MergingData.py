#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
# List to store individual datasets
dataset_list = []

# Read each dataset and append it to the list
for i in range(1, 51):
    filename = f"dataset_{i}.csv"
    dataset = pd.read_csv(filename)
    dataset_list.append(dataset)

# Concatenate the datasets
merged_df = pd.concat(dataset_list, ignore_index=True)

# Verify the merged dataframe
print(merged_df.head())

# Save the merged dataframe to a new CSV file
#merged_df.to_csv('merged_data.csv', index=False)

print(merged_df.describe)


# In[ ]:




