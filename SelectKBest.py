#!/usr/bin/env python
# coding: utf-8
# Steps of preprocessing of data
# Step 1: Import necessary library
# Step 2: Read Dataset
# Step 3: Sanity check of data
# step 4: Exploratory Data Analysis(EDA)
# Step 5: Outliers treatments
# Step 6: Duplicates and garbage value treatments
# Step 7: Encoding of data
# In[1]:


#Import necessary library

import numpy as np
import pandas as pd
import csv
import seaborn as sns
import math
import missingno as msno

import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import RFE

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

import time

from imblearn.over_sampling import SMOTE



# In[2]:


# Read Dataset
#filename_read = "UDP.csv"
df1 = pd.read_csv("UDP.csv",low_memory=False) #read the csv file in df
df2 = pd.read_csv("LDAP.csv",low_memory=False) #read the csv file in df
df3 = pd.read_csv("Portmap.csv",low_memory=False) #read the csv file in df
#df4 = pd.read_csv("NetBIOS.csv",low_memory=False) #read the csv file in df


# In[3]:


data_list = [df1, df2, df3]

print('Data dimensions: ')
for i, data in enumerate(data_list, start = 1):
  rows, cols = data.shape
  print(f'Data{i} -> {rows} rows, {cols} columns')


# In[4]:


df = pd.concat(data_list)
rows, cols = df.shape

print('New dimension:')
print(f'Number of rows: {rows}')
print(f'Number of columns: {cols}')
print(f'Total cells: {rows * cols}')


# In[5]:


# Deleting dataframes after concating to save memory
for d in data_list: del d
     


# In[6]:


# Renaming the columns by removing leading/trailing whitespace
col_names = {col: col.strip() for col in df.columns}
df.rename(columns = col_names, inplace = True)


# In[7]:


# data = pd.concat(data_list)
rows, cols = df.shape   # rows and column count

print('New dimension:')
print(f'Number of rows: {rows}')
print(f'Number of columns: {cols}')
print(f'Total cells: {rows * cols}')


# In[8]:


df.info()   # information about the datatypes


# In[9]:


#finding missing value
df.isnull().sum()


# In[10]:


#finding percentage of missing value
df.isnull().sum()/df.shape[0]*100


# In[11]:


#finding duplicate value
df.duplicated().sum()


# In[12]:


df = df.drop_duplicates() # dropping all duplicate rows

df


# In[13]:


#identify garbage value (if there is garbage value it is in form of object)

for i in df.select_dtypes(include="object").columns:
  print(df[i].value_counts())
  print("***"*10)


# In[14]:


df.columns


# In[15]:


#descriptive statistics
df.describe().T  #T is transpose


# In[16]:


df.describe(include ="object")


# In[17]:


# Checking for infinity values
# Select numeric columns
numeric_cols = df.select_dtypes(include = np.number).columns

# Check for infinity values and count them
inf_count = np.isinf(df[numeric_cols]).sum()

print(inf_count[inf_count > 0])


# In[18]:


# Replacing any infinite values (positive or negative) with NaN (not a number)
print(f'Initial missing values: {df.isna().sum().sum()}')

df.replace([np.inf, -np.inf], np.nan, inplace = True)

print(f'Missing values after processing infinite values: {df.isna().sum().sum()}')


# In[19]:


missing = df.isna().sum()
print(missing.loc[missing > 0])


# In[20]:


# Calculating missing value percentage in the dataset
mis_per = (missing / len(df)) * 100
mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
mis_table = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage of Total Values'})

print(mis_table.loc[mis_per > 0])


# In[21]:


sns.set_palette('pastel')
colors = sns.color_palette()

missing_vals = [col for col in df.columns if df[col].isna().any()]

fig, ax = plt.subplots(figsize = (2, 6))
msno.bar(df[missing_vals], ax = ax, fontsize = 12, color = colors)
ax.set_xlabel('Features', fontsize = 12)
ax.set_ylabel('Non-Null Value Count', fontsize = 12)
ax.set_title('Missing Value Chart', fontsize = 12)
plt.show()


# In[22]:


colors = sns.color_palette('Reds')
plt.hist(df['Flow Bytes/s'], color = colors[3])
plt.title('Histogram of Flow Bytes/s')
plt.xlabel('Flow Bytes/s')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[23]:


colors = sns.color_palette('Reds')
plt.hist(df['Flow Packets/s'], color = colors[1])
plt.title('Histogram of Flow Packets/s')
plt.xlabel('Flow Packets/s')
plt.ylabel('Frequency')
plt.show()


# In[24]:


med_flow_bytes = df['Flow Bytes/s'].median()
med_flow_packets = df['Flow Packets/s'].median()

print('Median of Flow Bytes/s: ', med_flow_bytes)
print('Median of Flow Packets/s: ', med_flow_packets)


# In[25]:


# Filling missing values with median
df['Flow Bytes/s'].fillna(med_flow_bytes, inplace = True)
df['Flow Packets/s'].fillna(med_flow_packets, inplace = True)

print('Number of \'Flow Bytes/s\' missing values:', df['Flow Bytes/s'].isna().sum())
print('Number of \'Flow Packets/s\' missing values:', df['Flow Packets/s'].isna().sum())


# In[ ]:





# In[ ]:





# In[ ]:





# # Analysing Patterns using Visualisations
# 

# In[26]:


df['Label'].unique()


# In[27]:


# Types of attacks & normal instances (BENIGN)
df['Label'].value_counts()


# In[28]:


# Creating a dictionary that maps each label to its attack type
attack_map = {'UDP':'UDP', 'MSSQL': 'MSSQL','BENIGN': 'BENIGN', 'NetBIOS' :'NetBIOS','LDAP':'LDAP','Portmap':'Portmap' }

# Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
df['Attack Type'] = df['Label'].map(attack_map)
     


# In[ ]:





# # Binary Classification

# In[31]:


# Create balanced dataset for binary classification
normal_traffic = df[df['Attack Type'] == 'BENIGN']
intrusions = df[df['Attack Type'] != 'BENIGN']
# Determine the number of samples needed
sample_size = min(len(normal_traffic), len(intrusions))

# Sample from both datasets to match the smaller size
normal_traffic = normal_traffic.sample(n=sample_size, replace=False, random_state=42)
intrusions = intrusions.sample(n=sample_size, replace=False, random_state=42)

# Combine the datasets and create binary labels
ids_data = pd.concat([intrusions, normal_traffic])
ids_data['Attack Type'] = np.where(ids_data['Attack Type'] == 'BENIGN', 0, 1)

# Adjust the sample size based on available data
target_sample_size = min(15000, len(ids_data))
if target_sample_size > 0:
    bc_data = ids_data.sample(n=target_sample_size, random_state=42)
    print(bc_data['Attack Type'].value_counts())
else:
    print("Insufficient data even after balancing.")



# In[32]:


#Identify non-numeric columns
non_numeric_columns = bc_data.select_dtypes(include=['object']).columns

# Convert non-numeric data to numeric using LabelEncoder
le = LabelEncoder()
for col in non_numeric_columns:
   bc_data[col] = le.fit_transform(bc_data[col])

# Split the data into features (X) and target (y)
X = bc_data.drop('Attack Type', axis=1)
y = bc_data['Attack Type']

# Split the data into training and test sets
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X, y, test_size=0.25, random_state=0)

# Displaying the shapes of the resulting datasets
print(f'X_train_bc shape: {X_train_bc.shape}')
print(f'X_test_bc shape: {X_test_bc.shape}')
print(f'y_train_bc shape: {y_train_bc.shape}')
print(f'y_test_bc shape: {y_test_bc.shape}')

# Scale numeric data
ss = StandardScaler()
X_train_bc = ss.fit_transform(X_train_bc)  # Fit on train data
X_test_bc = ss.transform(X_test_bc)        # Transform test data using the same scaler


# In[31]:


models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()


# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()


# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()


# In[32]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    
    # Fit the classifier
    models[key].fit(X_train_bc, y_train_bc)
    
    # Make predictions
    predictions = models[key].predict(X_test_bc)
    
    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test_bc)
    precision[key] = precision_score(predictions, y_test_bc)
    recall[key] = recall_score(predictions, y_test_bc)
    


# In[33]:


import pandas as pd

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

df_model


# In[34]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machines': LinearSVC(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbor': KNeighborsClassifier()
}

# Dictionaries to store metrics
accuracy, precision, recall, f1_scores= {}, {}, {}, {}

# Fit models and calculate metrics
for key in models.keys():
    # Fit the classifier
    models[key].fit(X_train_bc, y_train_bc)
    
    # Make predictions
    predictions = models[key].predict(X_test_bc)
    
    # Calculate metrics
    accuracy[key] = accuracy_score(y_test_bc, predictions)
    precision[key] = precision_score(y_test_bc, predictions)
    recall[key] = recall_score(y_test_bc, predictions)
    f1_scores[key] = f1_score(y_test_bc, predictions)


# Create a DataFrame to display metrics
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model['f1 Score'] = f1_scores.values()

print(df_model)

# Plot confusion matrices
for key in models.keys():
    # Make predictions
    predictions = models[key].predict(X_test_bc)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test_bc, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix for {key}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# In[30]:


df.drop('Label', axis = 1, inplace = True)


# In[31]:


class_counts = df['Attack Type'].value_counts()
selected_classes = class_counts[class_counts > 1500]
class_names = selected_classes.index
selected = df[df['Attack Type'].isin(class_names)]

dfs = []
for name in class_names:
    df = selected[selected['Attack Type'] == name]
    if len(df) > 5000:
        df = df.sample(n=5000, replace=False, random_state=0)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['Attack Type'].value_counts()


# In[32]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# Drop columns that are not numeric or that are not useful for the model
# This example assumes you want to convert non-numeric columns to numeric
non_numeric_columns = df.select_dtypes(include=['object']).columns

# Convert categorical features to numeric using Label Encoding
feature_encoders = {}
for column in non_numeric_columns:
    if column != 'Attack Type':  # Exclude the target column
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        feature_encoders[column] = le

# Encode the target variable
attack_label_encoder = LabelEncoder()
df['Attack Type_encoded'] = attack_label_encoder.fit_transform(df['Attack Type'])
class_labels = attack_label_encoder.classes_  # Get the class labels

# Define features and target
X = df.drop('Attack Type', axis=1)
y = df['Attack Type']

# Apply SMOTE to upsample the minority classes
smote = SMOTE(sampling_strategy='auto', random_state=0)
X_upsampled, y_upsampled = smote.fit_resample(X, y)

# Create a new DataFrame with the upsampled data
blnc_data = pd.DataFrame(X_upsampled, columns=X.columns)
blnc_data['Attack Type'] = y_upsampled
blnc_data = blnc_data.sample(frac=1, random_state=0)  # Shuffle the DataFrame

# Check the class distribution to verify balance
print(blnc_data['Attack Type'].value_counts())


# In[33]:


df['Attack Type'].value_counts()


# In[34]:


features = blnc_data.drop('Attack Type', axis = 1)
labels = blnc_data['Attack Type']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)
     


# In[35]:


# Printing corresponding attack type for each encoded value
##encoded_values = df['Attack Number'].unique()
##for val in sorted(encoded_values):
##    print(f"{val}: {le.inverse_transform([val])[0]}")
     


# In[36]:


import warnings
warnings.filterwarnings("ignore")


# # without Feature Selection

# In[42]:


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=0),
    'SVM with RBF kernel': SVC(kernel='rbf', C=1, gamma=0.1, random_state=0, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Function to evaluate models
def evaluate_model_no_fs(model, model_name, X_train, X_test, y_train, y_test):
    # Start timer for training
    start_train_time = time.time()
    
    # Cross-validation scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Fit the model on the entire training data
    model.fit(X_train, y_train)
    
    # End timer for training
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    
    # Start timer for testing
    start_test_time = time.time()

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else "N/A"
    
    
    # End timer for testing
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print results
    print(f"--- {model_name} ---")
    print(f"Cross-validation scores: {scores.tolist()}")
    print(f"Mean cross-validation score: {scores.mean():.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")

    # Plot confusion matrix with original labels
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")
    evaluate_model_no_fs(model, name, X_train, X_test, y_train, y_test)


# # Feature Selection -SelectKBest

# In[40]:


from sklearn.feature_selection import SelectKBest, f_classif


# Apply SelectKBest to select top 20 features using f_classif
k_best = SelectKBest(score_func=f_classif, k=20)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=0),
    'SVM with RBF kernel': SVC(kernel='rbf', C=1, gamma=0.1, random_state=0, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Function to evaluate models with SelectKBest feature selection
def evaluate_model_selectkbest(model, model_name, X_train, X_test, y_train, y_test):
    # Start timer for training
    start_train_time = time.time()
    
    # Cross-validation scores
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Fit the model on the entire training data
    model.fit(X_train, y_train)

    # End timer for training
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    
    # Start timer for testing
    start_test_time = time.time()
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else "N/A"
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

     # End timer for testing
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    
    
    # Print results
    print(f"--- {model_name} with SelectKBest ---")
    print(f"Cross-validation scores: {scores.tolist()}")
    print(f"Mean cross-validation score: {scores.mean():.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")
    
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Evaluate each model with SelectKBest feature selection
for name, model in models.items():
    print(f"Evaluating {name}...")
    evaluate_model_selectkbest(model, name, X_train_kbest, X_test_kbest, y_train, y_test)


# In[41]:


from sklearn.feature_selection import SelectKBest, f_classif


# Apply SelectKBest to select top 10 features using f_classif
k_best = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=0),
    'SVM with RBF kernel': SVC(kernel='rbf', C=1, gamma=0.1, random_state=0, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Function to evaluate models with SelectKBest feature selection
def evaluate_model_selectkbest(model, model_name, X_train, X_test, y_train, y_test):
    # Start timer for training
    start_train_time = time.time()
    
    # Cross-validation scores
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Fit the model on the entire training data
    model.fit(X_train, y_train)

    # End timer for training
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    
    # Start timer for testing
    start_test_time = time.time()
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else "N/A"
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

     # End timer for testing
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    
    
    # Print results
    print(f"--- {model_name} with SelectKBest ---")
    print(f"Cross-validation scores: {scores.tolist()}")
    print(f"Mean cross-validation score: {scores.mean():.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")
    
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Evaluate each model with SelectKBest feature selection
for name, model in models.items():
    print(f"Evaluating {name}...")
    evaluate_model_selectkbest(model, name, X_train_kbest, X_test_kbest, y_train, y_test)


# In[42]:


from sklearn.feature_selection import SelectKBest, f_classif


# Apply SelectKBest to select top 30 features using f_classif
k_best = SelectKBest(score_func=f_classif, k=30)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=0),
    'SVM with RBF kernel': SVC(kernel='rbf', C=1, gamma=0.1, random_state=0, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Function to evaluate models with SelectKBest feature selection
def evaluate_model_selectkbest(model, model_name, X_train, X_test, y_train, y_test):
    # Start timer for training
    start_train_time = time.time()
    
    # Cross-validation scores
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Fit the model on the entire training data
    model.fit(X_train, y_train)

    # End timer for training
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    
    # Start timer for testing
    start_test_time = time.time()
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else "N/A"
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

     # End timer for testing
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    
    
    # Print results
    print(f"--- {model_name} with SelectKBest ---")
    print(f"Cross-validation scores: {scores.tolist()}")
    print(f"Mean cross-validation score: {scores.mean():.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")
    
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Evaluate each model with SelectKBest feature selection
for name, model in models.items():
    print(f"Evaluating {name}...")
    evaluate_model_selectkbest(model, name, X_train_kbest, X_test_kbest, y_train, y_test)


# In[43]:


from sklearn.feature_selection import SelectKBest, f_classif

# Apply SelectKBest to select top 20 features using f_classif
k_best = SelectKBest(score_func=f_classif, k=20)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Get feature scores and selected features
feature_scores = k_best.scores_   # Scores of each feature
selected_feature_indices = k_best.get_support(indices=True)  # Indices of selected features

# Print feature scores and selected feature indices
print("Feature Scores:", feature_scores)
print("Selected Feature Indices:", selected_feature_indices)

# If you have feature names (for example in a DataFrame), print them
if hasattr(X_train, 'columns'):  # Assuming X_train is a DataFrame
    selected_feature_names = X_train.columns[selected_feature_indices]
    print("Selected Feature Names:", selected_feature_names)


# In[37]:


from sklearn.feature_selection import SelectKBest, f_classif

# Apply SelectKBest to select top 10 features using f_classif
k_best = SelectKBest(score_func=f_classif, k=30)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Get feature scores and selected features
feature_scores = k_best.scores_   # Scores of each feature
selected_feature_indices = k_best.get_support(indices=True)  # Indices of selected features

# Print feature scores and selected feature indices
print("Feature Scores:", feature_scores)
print("Selected Feature Indices:", selected_feature_indices)

# If you have feature names (for example in a DataFrame), print them
if hasattr(X_train, 'columns'):  # Assuming X_train is a DataFrame
    selected_feature_names = X_train.columns[selected_feature_indices]
    print("Selected Feature Names:", selected_feature_names)


# Feature Selection -RFE

# In[ ]:


# Function to evaluate models with RFE feature selection
def evaluate_model_rfe(model, model_name, X_train, X_test, y_train, y_test):
    # Initialize RFE with the model
    rfe = RFE(estimator=model, n_features_to_select=20)  # Adjust number of features as needed
    rfe.fit(X_train, y_train)

    # Transform data using the fitted RFE
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # Check shapes after RFE
    print(f"X_train_rfe shape: {X_train_rfe.shape}")
    print(f"X_test_rfe shape: {X_test_rfe.shape}")

    # Cross-validation scores
    scores = cross_val_score(model, X_train_rfe, y_train, cv=cv, scoring='accuracy')
    
    # Fit the model on the entire training data
    model.fit(X_train_rfe, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_rfe)
    y_pred_proba = model.predict_proba(X_test_rfe) if hasattr(model, 'predict_proba') else None
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else "N/A"
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print results
    print(f"--- {model_name} with RFE ---")
    print(f"Cross-validation scores: {scores.tolist()}")
    print(f"Mean cross-validation score: {scores.mean():.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#Evaluate each model with RFE feature selection
for name, model in models.items():
    print(f"Evaluating {name}...")
    evaluate_model_rfe(model, name, X_train, X_test, y_train, y_test)


# In[ ]:




