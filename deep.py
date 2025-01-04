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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


# # Analysing Patterns using Visualisations
# 

# In[26]:


df['Label'].unique()


# In[27]:


# Types of attacks & normal instances (BENIGN)
df['Label'].value_counts()


# In[28]:


#@title Standardtext für Titel
df[df['Label'] != "BENIGN"]['Label'].value_counts().plot(kind='bar')


# In[29]:


np.unique(df['Label'])


# In[30]:


df.shape


# In[31]:


df['Label'].unique()


# In[32]:


# Types of attacks & normal instances (BENIGN)
df['Label'].value_counts()


# In[33]:


# Creating a dictionary that maps each label to its attack type
attack_map = {'UDP':'UDP', 'MSSQL': 'MSSQL','BENIGN': 'BENIGN', 'NetBIOS' :'NetBIOS','LDAP':'LDAP','Portmap':'Portmap' }

# Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
df['Attack Type'] = df['Label'].map(attack_map)
     


# In[34]:


df.drop('Label', axis = 1, inplace = True)


# In[35]:


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


# In[36]:


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


# In[37]:


df['Attack Type'].value_counts()


# In[39]:


features = blnc_data.drop('Attack Type', axis = 1)
labels = blnc_data['Attack Type']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)
     


# ## Model Training 

# # CNN

# In[40]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# In[41]:


# In order to ignore FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from sklearn import metrics
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from hyperopt.plotting import main_plot_history, main_plot_vars
import uuid
import gc
from tensorflow import keras
import tensorflow as tf
import pydot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout



# In[60]:


model = Sequential()
model.add(Convolution1D(filters=128, kernel_size=6, input_shape=(87, 1)))
model.add(Activation('relu'))
model.add(Convolution1D(filters=256, kernel_size=6))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary() 

tf.keras.utils.plot_model(
    model,
        show_shapes=True,
    show_dtype=False,
      show_layer_names=False,
)



# In[61]:


# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[43]:


# Ensure 'Attack Type_encoded' is used for the target variable
X = df.drop(['Attack Type', 'Attack Type_encoded'], axis=1)
y = df['Attack Type_encoded']

# Apply SMOTE to upsample the minority classes
smote = SMOTE(sampling_strategy='auto', random_state=0)
X_upsampled, y_upsampled = smote.fit_resample(X, y)

# Create a new DataFrame with the upsampled data
blnc_data = pd.DataFrame(X_upsampled, columns=X.columns)
blnc_data['Attack Type_encoded'] = y_upsampled
blnc_data = blnc_data.sample(frac=1, random_state=0)  # Shuffle the DataFrame

# Check the class distribution to verify balance
print(blnc_data['Attack Type_encoded'].value_counts())

# Define features and target with encoded labels
features = blnc_data.drop('Attack Type_encoded', axis=1)
labels = blnc_data['Attack Type_encoded']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[62]:


# Reshape data for CNN
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# One-hot encode target variables
y_train_cnn = tf.keras.utils.to_categorical(y_train, num_classes=len(class_labels))
y_test_cnn = tf.keras.utils.to_categorical(y_test, num_classes=len(class_labels))

# Train the model
start_train_time = time.time()

history = model.fit(X_train_cnn, y_train_cnn, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

end_train_time = time.time()
train_time = end_train_time - start_train_time

# Evaluate the model
start_test_time = time.time()

y_pred_cnn = model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred_cnn, axis=1)
y_test_classes = np.argmax(y_test_cnn, axis=1)

end_test_time = time.time()
test_time = end_test_time - start_test_time

# Calculate metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
auc = roc_auc_score(y_test_cnn, y_pred_cnn, multi_class='ovr')

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC: {auc:.2f}")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Testing Time: {test_time:.2f} seconds")

# Plot confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix for CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# # LSTM

# In[44]:


# Reshape data for LSTM
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# One-hot encode target variables
y_train_lstm = tf.keras.utils.to_categorical(y_train, num_classes=len(class_labels))
y_test_lstm = tf.keras.utils.to_categorical(y_test, num_classes=len(class_labels))

# Build LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(class_labels), activation='softmax'))


# In[45]:


# Plot the model architecture using the requested method
tf.keras.utils.plot_model(
    model,
    show_shapes=True,     # Shows the shape of input and output for each layer
    show_dtype=False,      # Hides data types of tensors (False by request)
    show_layer_names=False # Hides layer names (False by request)
)


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
start_train_time = time.time()

history = model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

end_train_time = time.time()
train_time = end_train_time - start_train_time

# Evaluate the model
start_test_time = time.time()

y_pred_lstm = model.predict(X_test_lstm)
y_pred_classes = np.argmax(y_pred_lstm, axis=1)
y_test_classes = np.argmax(y_test_lstm, axis=1)

end_test_time = time.time()
test_time = end_test_time - start_test_time

# Calculate metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
auc = roc_auc_score(y_test_lstm, y_pred_lstm, multi_class='ovr')

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC: {auc:.2f}")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Testing Time: {test_time:.2f} seconds")

# Plot confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix for LSTM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




