#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:
# List to store individual datasets
dataset_list = []

# Read each dataset and append it to the list
for i in range(1, 51):
    filename = f"dataset_{i}.csv"
    dataset = pd.read_csv(filename)
    dataset_list.append(dataset)

# Concatenate the datasets
df = pd.concat(dataset_list, ignore_index=True)


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve


# In[ ]:


# Extracting features (X) and target variable (y)
# Assuming 'Is Fraud?' is your target column and you have dropped other irrelevant features
X = df.drop(columns=['Is Fraud?'])  # Drop the target column to get features
y = df['Is Fraud?']  # Get the target column


# In[ ]:


#Data is stored in X (features) and y (target)

# Step 1: Split the dataset into training, validation, and test sets while maintaining class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42, stratify=y_train)
#training set 70%, test set 20%, validation set 10%
#for this final run X_train becomes 80% again


# In[ ]:


# Initialize and train the SVM classifier with C = 0.1
svm_classifier = SVC(kernel='linear', C=0.1, random_state=42)
svm_classifier.fit(X_train, y_train)

# Initialize and train the Logistic Regression classifier with C = 100
LR_classifier = LogisticRegression(C = 100, max_iter=1000)
LR_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_classifier.predict(X_test)
y_pred_lr = LR_classifier.predict(X_test)

# Calculate precision
precision_svm = precision_score(y_test, y_pred_svm)
precision_lr = precision_score(y_test, y_pred_lr)

# Calculate recall
recall_svm = recall_score(y_test, y_pred_svm)
recall_lr = recall_score(y_test, y_pred_lr)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Calculate ROC-AUC score
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr)

# Compute the average precision score
auprc_svm = average_precision_score(y_test, y_pred_svm)
auprc_lr = average_precision_score(y_test, y_pred_lr)

# Print the results
print("Logistic Regression Metrics:")
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("Accuracy:", accuracy_lr)
print("ROC AUC:", roc_auc_lr)
print("AUPRC:", auprc_lr)

print("\nSVM Metrics:")
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("Accuracy:", accuracy_svm)
print("ROC AUC:", roc_auc_svm)
print("AUPRC:", auprc_svm)


# In[ ]:


#Plotting ROC curves
# Get predicted probabilities from the model
lr_probs = LR_classifier.predict_proba(X_test)[:, 1]
svm_probs = svm_classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve for logistic regression
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# Compute ROC curve for SVM
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)

# Plot ROC curves
plt.figure(figsize=(10, 5))
plt.plot(lr_fpr, lr_tpr, linestyle='-', label='Logistic Regression')
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:


#Plotting precision-recall curve
# Compute precision-recall curve for logistic regression
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)

# Compute precision-recall curve for SVM
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)

# Plot Precision-Recall curves
plt.figure(figsize=(10, 5))
plt.plot(lr_recall, lr_precision, linestyle='-', label='Logistic Regression')
plt.plot(svm_recall, svm_precision, linestyle='-', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

