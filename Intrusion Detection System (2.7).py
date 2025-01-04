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


# # Read Dataset

# In[2]:


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


df.head()  # read first five rows


# In[8]:


df.tail()  #bottom five rows


# # Sanity check of data

# In[9]:


# data = pd.concat(data_list)
rows, cols = df.shape   # rows and column count

print('New dimension:')
print(f'Number of rows: {rows}')
print(f'Number of columns: {cols}')
print(f'Total cells: {rows * cols}')


# In[10]:


df.info()   # information about the datatypes


# In[11]:


#finding missing value
df.isnull().sum()


# In[12]:


#finding percentage of missing value
df.isnull().sum()/df.shape[0]*100


# In[13]:


#finding duplicate value
df.duplicated().sum()


# In[14]:


df = df.drop_duplicates() # dropping all duplicate rows

df


# In[15]:


#identify garbage value (if there is garbage value it is in form of object)

for i in df.select_dtypes(include="object").columns:
  print(df[i].value_counts())
  print("***"*10)


# In[16]:


df.columns


# In[17]:


#descriptive statistics
df.describe().T  #T is transpose

The df.describe(include="object") function in pandas provides descriptive statistics of categorical (object) columns in the DataFrame df. This typically includes count, unique, top, and freq for each categorical column

count:  The number of non-null entries in the column.
unique: The number of unique categories in the column.
top:    The most frequent category in the column.
freq:   The frequency of the most frequent category.
# In[18]:


df.describe(include ="object")

The df.select_dtypes(include="number").corr() function in pandas is used to compute the pairwise correlation of columns with numerical data types in the DataFrame df.
# In[19]:


# Compute the correlation matrix for numerical columns
s=df.select_dtypes(include="number").corr()


# In[20]:


plt.figure(figsize = (80,80))
sns.heatmap(s,annot=True)

Checking for infinity values in the numeric columns of a DataFrame and prints the count of such values for each column where at least one infinity value is found. 

1. Select Numeric Columns: It selects all columns in the DataFrame that have numeric data types.
2. Check for Infinity Values: It checks each of these numeric columns for infinity values (both positive and negative).
3. Count Infinity Values: It counts the number of infinity values in each column.
4. Print Results: It prints the counts of infinity values for columns that have more than zero such values.
# In[21]:


# Checking for infinity values
# Select numeric columns
numeric_cols = df.select_dtypes(include = np.number).columns

# Check for infinity values and count them
inf_count = np.isinf(df[numeric_cols]).sum()

print(inf_count[inf_count > 0])


# In[22]:


# Replacing any infinite values (positive or negative) with NaN (not a number)
print(f'Initial missing values: {df.isna().sum().sum()}')

df.replace([np.inf, -np.inf], np.nan, inplace = True)

print(f'Missing values after processing infinite values: {df.isna().sum().sum()}')


# In[23]:


missing = df.isna().sum()
print(missing.loc[missing > 0])


# In[24]:


# Calculating missing value percentage in the dataset
mis_per = (missing / len(df)) * 100
mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
mis_table = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage of Total Values'})

print(mis_table.loc[mis_per > 0])


# Visualisation of missing data
# 

# In[25]:


sns.set_palette('pastel')
colors = sns.color_palette()

missing_vals = [col for col in df.columns if df[col].isna().any()]

fig, ax = plt.subplots(figsize = (2, 6))
msno.bar(df[missing_vals], ax = ax, fontsize = 12, color = colors)
ax.set_xlabel('Features', fontsize = 12)
ax.set_ylabel('Non-Null Value Count', fontsize = 12)
ax.set_title('Missing Value Chart', fontsize = 12)
plt.show()


# Dealing with missing values (Columns with missing data)

# In[26]:


plt.figure(figsize = (8, 3))
sns.boxplot(x = df['Flow Bytes/s'])
plt.xlabel('Boxplot of Flow Bytes/s')
plt.grid()
plt.show()
     


# In[27]:


colors = sns.color_palette('Reds')
plt.hist(df['Flow Bytes/s'], color = colors[3])
plt.title('Histogram of Flow Bytes/s')
plt.xlabel('Flow Bytes/s')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[28]:


plt.figure(figsize = (8, 3))
sns.boxplot(x = df['Flow Packets/s'])
plt.xlabel('Boxplot of Flow Bytes/s')
plt.grid()
plt.show()
     


# In[29]:


colors = sns.color_palette('Reds')
plt.hist(df['Flow Packets/s'], color = colors[1])
plt.title('Histogram of Flow Packets/s')
plt.xlabel('Flow Packets/s')
plt.ylabel('Frequency')
plt.show()


# In[30]:


med_flow_bytes = df['Flow Bytes/s'].median()
med_flow_packets = df['Flow Packets/s'].median()

print('Median of Flow Bytes/s: ', med_flow_bytes)
print('Median of Flow Packets/s: ', med_flow_packets)


# In[31]:


# Filling missing values with median
df['Flow Bytes/s'].fillna(med_flow_bytes, inplace = True)
df['Flow Packets/s'].fillna(med_flow_packets, inplace = True)

print('Number of \'Flow Bytes/s\' missing values:', df['Flow Bytes/s'].isna().sum())
print('Number of \'Flow Packets/s\' missing values:', df['Flow Packets/s'].isna().sum())

Analysing Patterns using Visualisations

# In[32]:


df['Label'].unique()


# In[33]:


# Types of attacks & normal instances (BENIGN)
df['Label'].value_counts()


# In[34]:


# Creating a dictionary that maps each label to its attack type
attack_map = {'UDP':'UDP', 'MSSQL': 'MSSQL','BENIGN': 'BENIGN', 'NetBIOS' :'NetBIOS','LDAP':'LDAP','Portmap':'Portmap' }

# Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
df['Attack Type'] = df['Label'].map(attack_map)
     


# In[35]:


df['Attack Type'].value_counts()


# In[36]:


df.drop('Label', axis = 1, inplace = True)


# # Encoding 
# Most estimators for classification in scikit-learn convert class labels to integers internally, it is considered good practice to provide class labels as integer arrays to avoid technical glitches (Text book 1: Python Machine Learning, 3rd Ed, Raschka, Sebastian and Mirjalili, Vahid )

# In[37]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Attack Number'] = le.fit_transform(df['Attack Type'])

print(df['Attack Number'].unique())

Note that the fit_transform method is just a shortcut for calling fit and transform separately, and we can use the inverse_transform method to transform
the integer class labels back into their original string representation: 
    textbook 1 [chapter 4]
# In[38]:


# Printing corresponding attack type for each encoded value
encoded_values = df['Attack Number'].unique()
for val in sorted(encoded_values):
    print(f"{val}: {le.inverse_transform([val])[0]}")
     

df.corr(numeric_only=True): This method calculates the correlation matrix of the DataFrame df. 
The numeric_only=True : parameter ensures that only numeric columns are considered for the calculation.
.round(2): This rounds the correlation coefficients to 2 decimal places for a cleaner presentation.corr.style: This accesses the Styler object, which is used for styling the DataFrame.

.background_gradient(cmap='coolwarm', axis=None): This method applies a color gradient to the background of the DataFrame cells.

The cmap='coolwarm' argument specifies the colormap to be used. The coolwarm colormap ranges from cool colors (e.g., blue) to warm colors (e.g., red), which helps in visualizing the strength and direction of the correlation. 

The axis=None argument means that the gradient is applied across the entire DataFrame, rather than along rows or columns.
.format(precision=2): This ensures that the displayed values in the styled DataFrame are formatted to 2 decimal places.
# In[39]:


corr = df.corr(numeric_only = True).round(2)
corr.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)


# In[40]:


fig, ax = plt.subplots(figsize = (24, 24))
sns.heatmap(corr, cmap = 'coolwarm', annot = False, linewidth = 0.5)
plt.title('Correlation Matrix', fontsize = 18)
plt.show()


# In[41]:


fig, ax = plt.subplots(figsize = (80, 80))
sns.heatmap(corr, cmap = 'RdBu', annot = True, vmin = -1, vmax = 1, annot_kws={'fontsize': 11, 'fontweight' : 'bold'})
plt.title('Correlation Matrix', fontsize = 18)
plt.show()

Correlation taken from: https://github.com/noushinpervez/Intrusion-Detection-CIC-IDS2017/blob/main/Intrusion-Detection-CIC-IDS2017.ipynb
# In[42]:


# Positive correlation features for 'Attack Number'
pos_corr_features = corr['Attack Number'][(corr['Attack Number'] > 0) & (corr['Attack Number'] < 1)].index.tolist()

print("Features with positive correlation with 'Attack Number':\n")
for i, feature in enumerate(pos_corr_features, start = 1):
    corr_value = corr.loc[feature, 'Attack Number']
    print('{:<3} {:<24} :{}'.format(f'{i}.', feature, corr_value))


# In[43]:


print(f'Number of considerable important features: {len(pos_corr_features)}')


# In[44]:


# Checking for columns with zero standard deviation (the blank squares in the heatmap)

# Calculate the standard deviation for each numeric column
std = df.std(numeric_only = True)

# Identify columns with zero standard deviation
zero_std_cols = std[std == 0].index.tolist()


zero_std_cols

For plotting the correlation matrix, the 'Attack Type' column is encoded and plotted the heatmap. From the heatmap, it is observed that there are many pairs of highly correlated features. Highly correlated features in the dataset are problematic and lead to overfitting. A positive correlation exists when one variable decreases as the other variable decreases or one variable increases while the other increases. There are 17 features with positive correlations that may help in predicting the target feature.The columns with zero standard deviation have the same value in all rows. These columns don't have any variance. It simply means that there is no meaningful relationship with any other columns which results in NaN correlation cofficient. These columns cannot help differentiate between the classes or groups of data. So, these zero standard deviation columns don't contribute to the correlation matrix and will appear blank in the heatmap. This can be helpful while doing data processing as we may drop the columns if we find out that these columns has no variation.
# # Visualization of Linear Relationships of columns (Continuous Numerical Variables)
The sample method selects sample_size number of rows randomly from the DataFrame df. 

The replace=False parameter ensures that sampling is done without replacement (each row can only be selected once). 

The random_state=0 parameter sets a seed for the random number generator to make the sampling reproducible.
# In[45]:


# Data sampling for data analysis
sample_size = int(0.2 * len(df)) # 20% of the original size
sampled_data = df.sample(n = sample_size, replace = False, random_state = 0)
sampled_data.shape

Calculate Means and Variation:

For each numeric column, the mean for both the original and sampled datasets is calculated using mean().
The variation percentage is calculated as the absolute difference between the new mean and the old mean, divided by the old mean. 
This percentage is added to a list if it exceeds 5%.Print Comparisons:

The means and variation percentages are printed in a formatted manner.
If the variation percentage exceeds 5%, the feature is appended to the high_variations list.
# In[46]:


# To assess if a sample is representative of the population and comparison of descriptive statistics (mean)
numeric_cols = df.select_dtypes(include = [np.number]).columns.tolist()
print('Descriptive Statistics Comparison (mean):\n')
print('{:<32s}{:<22s}{:<22s}{}'.format('Feature', 'Original Dataset', 'Sampled Dataset', 'Variation Percentage'))
print('-' * 96)

high_variations = []
for col in numeric_cols:
    old = df[col].describe()[1]
    new = sampled_data[col].describe()[1]
    if old == 0:
        pct = 0
    else:
        pct = abs((new - old) / old)
    if pct * 100 > 5:
        high_variations.append((col, pct * 100))
    print('{:<32s}{:<22.6f}{:<22.6f}{:<2.2%}'.format(col, old, new, pct))
     


# In[47]:


# Labels and values for features with high variation
labels = [t[0] for t in high_variations]
values = [t[1] for t in high_variations]

#A blue color palette is selected with the number of colors equal to the number of labels.
colors = sns.color_palette('Blues', n_colors=len(labels))


fig, ax = plt.subplots(figsize = (15, 10))
ax.bar(labels, values, color = colors)

for i in range(len(labels)):
    ax.text(i, values[i], str(round(values[i], 2)), ha = 'center', va = 'bottom', fontsize = 10)

plt.xticks(rotation = 90)
ax.set_title('Variation percenatge of the features of the sample which\n mean value variates higher than 5% of the actual mean')
ax.set_ylabel('Percentage (%)')
ax.set_yticks(np.arange(0, 101, 10)) #Set y-axis ticks from 0 to 100 with a step of 10.
plt.show()


# In[48]:


# Printing the unique value count
indent = '{:<3} {:<30}: {}'
print('Unique value count for: ')

#The for loop iterates through the list of column names in sampled_data, excluding the last column ([:-1]).
for i, feature in enumerate(list(sampled_data.columns)[:-1], start = 1):
    print(indent.format(f'{i}.', feature, sampled_data[feature].nunique()))


# In[49]:


'''Generating a set of visualizations for columns that have more than one unique value but less than 50 unique values.
For categorical columns, a bar plot is generated showing the count of each unique value.
For numerical columns, a histogram is generated.'''

# Calculate number of unique values for each column
unique_values = sampled_data.nunique()

# Select columns with more than 1 and less than 50 unique values
selected_cols = sampled_data[[col for col in sampled_data if 1 < unique_values[col] < 50]]
rows, cols = selected_cols.shape
col_names = list(selected_cols)

#Calculates the number of rows needed for subplots based on the number of selected columns (cols).
num_of_rows = (cols + 3) // 4

color_palette = sns.color_palette('Blues', n_colors = 3)
plt.figure(figsize = (6 * 4, 8 * num_of_rows))

for i in range(cols):
    plt.subplot(num_of_rows, 4, i + 1)
    col_data = selected_cols.iloc[:, i]
    if col_data.dtype.name == 'object':
        col_data.value_counts().plot(kind = 'bar', color = color_palette[2])
    else:
        col_data.hist(color = color_palette[0])

    plt.ylabel('Count')
    plt.xticks(rotation = 90)
    plt.title(col_names[i])

plt.tight_layout()
plt.show()


# In[50]:


# Correlation matrix for sampled data
corr_matrix = sampled_data.corr(numeric_only = True).round(2)
corr_matrix.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)

# Plotting the pairs of strongly positive correlated features in the sampled_data that have a correlation coefficient of 0.85 or 
higherColumns List: cols holds the column names excluding the last two.

Finding High Correlation Pairs:

1. Iterate over all unique pairs of columns.
2. Compute the correlation coefficient for each pair.
3. Skip pairs with NaN or coefficients below the threshold (corr_th).
4. Append pairs with coefficients meeting the threshold to high_corr_pairs.

Subplots Layout Calculation:
Determine the number of rows (rows) and columns (cols) needed to accommodate all plots. Ensure any remaining plots fit by adjusting the row count.

Plotting:

1. Create subplots with the specified dimensions.
2. For each subplot, plot the data points for the corresponding pair of columns.
3. Use different colors for near-perfect correlations (val > 0.99) and others.
3. Set labels and titles for each subplot.
4. Remove any unused subplots to avoid empty plots.

Displaying the Plot:

Adjust layout for better spacing and show the plot.
# In[51]:


# List of numeric columns
numeric_cols = sampled_data.select_dtypes(include=[np.number]).columns.tolist()

#An empty list high_corr_pairs is initialized to store pairs of columns with high correlation, 
#corr_th is set to 0.85 as the correlation threshold.
high_corr_pairs = []
corr_th = 0.85

# Find pairs of columns with a correlation coefficient of 0.85 or higher
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        val = sampled_data[numeric_cols[i]].corr(sampled_data[numeric_cols[j]])
        # If the correlation coefficient is NaN or below the threshold, skip to the next pair
        if np.isnan(val) or val < corr_th:
            continue
        high_corr_pairs.append((val, numeric_cols[i], numeric_cols[j]))

# Calculate the number of rows and columns for the subplots
size = len(high_corr_pairs)
cols = 4  # Number of columns for the subplots
rows = size // cols + (1 if size % cols else 0)

fig, axs = plt.subplots(rows, cols, figsize=(24, int(size * 1.7)))

for i in range(rows):
    for j in range(cols):
        index = i * cols + j
        if index >= size:
            fig.delaxes(axs[i, j])
            continue
        val, x, y = high_corr_pairs[index]
        if val > 0.99:
            axs[i, j].scatter(sampled_data[x], sampled_data[y], color='green', alpha=0.1)
        else:
            axs[i, j].scatter(sampled_data[x], sampled_data[y], color='blue', alpha=0.1)
        axs[i, j].set_xlabel(x)
        axs[i, j].set_ylabel(y)
        axs[i, j].set_title(f'{x} vs\n{y} ({val:.2f})')

fig.tight_layout()
plt.show()


# In[52]:


sampled_data.drop('Attack Number', axis = 1, inplace = True)
df.drop('Attack Number', axis = 1, inplace = True)


# In[53]:


# Identifying outliers
numeric_data = sampled_data.select_dtypes(include = ['float', 'int'])
q1 = numeric_data.quantile(0.25)
q3 = numeric_data.quantile(0.75)
iqr = q3 - q1
outlier = (numeric_data < (q1 - 1.5 * iqr)) | (numeric_data > (q3 + 1.5 * iqr))
outlier_count = outlier.sum()
outlier_percentage = round(outlier.mean() * 100, 2)
outlier_stats = pd.concat([outlier_count, outlier_percentage], axis = 1)
outlier_stats.columns = ['Outlier Count', 'Outlier Percentage']

print(outlier_stats)


# In[54]:


# Identifying outliers based on attack type
outlier_counts = {}
for i in numeric_data:
    for attack_type in sampled_data['Attack Type'].unique():
        attack_data = sampled_data[i][sampled_data['Attack Type'] == attack_type]
        q1, q3 = np.percentile(attack_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        num_outliers = ((attack_data < lower_bound) | (attack_data > upper_bound)).sum()
        outlier_percent = num_outliers / len(attack_data) * 100
        outlier_counts[(i, attack_type)] = (num_outliers, outlier_percent)

for i in numeric_data:
  print(f'Feature: {i}')
  for attack_type in sampled_data['Attack Type'].unique():
    num_outliers, outlier_percent = outlier_counts[(i, attack_type)]
    print(f'- {attack_type}: {num_outliers} ({outlier_percent:.2f}%)')
  print()


# In[55]:


# Plotting the percentage of outliers that are higher than 20%
fig, ax = plt.subplots(figsize = (24, 10))
for i in numeric_data:
    for attack_type in sampled_data['Attack Type'].unique():
        num_outliers, outlier_percent = outlier_counts[(i, attack_type)]
        if outlier_percent > 20:
            ax.bar(f'{i} - {attack_type}', outlier_percent)

ax.set_xlabel('Feature-Attack Type')
ax.set_ylabel('Percentage of Outliers')
ax.set_title('Outlier Analysis')
ax.set_yticks(np.arange(0, 41, 10))
plt.xticks(rotation = 90)
plt.show()

Visualization of column relationships (Categorical Variables)
All the features in our dataset is numerical. We have one Categorical Variable
# In[56]:


# Different 'Attack Type' in the main dataset excluding 'BENIGN'
attacks = df.loc[df['Attack Type'] != 'BENIGN']

plt.figure(figsize = (10, 6))
ax = sns.countplot(x = 'Attack Type', data = attacks, palette = 'pastel', order = attacks['Attack Type'].value_counts().index)
plt.title('Types of attacks')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.xticks(rotation = 90)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height() + 1000), ha = 'center')

plt.show()


# In[57]:


attack_counts = attacks['Attack Type'].value_counts()
threshold = 0.005
percentages = attack_counts / attack_counts.sum()
small_slices = percentages[percentages < threshold].index.tolist()
attack_counts['Other'] = attack_counts[small_slices].sum()
attack_counts.drop(small_slices, inplace = True)

sns.set_palette('pastel')
plt.figure(figsize = (8, 8))
plt.pie(attack_counts.values, labels = attack_counts.index, autopct = '%1.1f%%', textprops={'fontsize': 6})
plt.title('Distribution of Attack Types')
plt.legend(attack_counts.index, loc = 'best')
plt.show()
     


# In[58]:


# Creating a boxplot for each attack type with the columns of sampled dataset
for attack_type in sampled_data['Attack Type'].unique():
    attack_data = sampled_data[sampled_data['Attack Type'] == attack_type]
    plt.figure(figsize=(20, 20))
    sns.boxplot(data = attack_data.drop(columns = ['Attack Type']), orient = 'h')
    plt.title(f'Boxplot of Features for Attack Type: {attack_type}')
    plt.xlabel('Feature Value')
    plt.show()


# In[59]:


df.groupby('Attack Type').first()

Data Preprocessing
Preprocessing
# In[60]:


# For improving performance and reduce memory-related errors
old_memory_usage = df.memory_usage().sum() / 1024 ** 2
print(f'Initial memory usage: {old_memory_usage:.2f} MB')
for col in df.columns:
    col_type = df[col].dtype
    if col_type != object:
        c_min = df[col].min()
        c_max = df[col].max()
        # Downcasting float64 to float32
        if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)

        # Downcasting int64 to int32
        elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)

new_memory_usage = df.memory_usage().sum() / 1024 ** 2
print(f"Final memory usage: {new_memory_usage:.2f} MB")


# In[61]:


# Calculating percentage reduction in memory usage
print(f'Reduced memory usage: {1 - (new_memory_usage / old_memory_usage):.2%}')


# In[62]:


df.info()


# In[63]:


df.describe().transpose()


# In[64]:


# Dropping columns with only one unique value
num_unique = df.nunique()
one_variable = num_unique[num_unique == 1]
not_one_variable = num_unique[num_unique > 1].index

dropped_cols = one_variable.index
df = df[not_one_variable]

print('Dropped columns:')
dropped_cols


# In[65]:


df.shape


# In[66]:


# Columns after removing non variant columns
df.columns


# # Applying PCA to reduce dimensions
Preprocessing: This part of the code ensures the features are preprocessed correctly by handling missing values, scaling numeric features, and encoding categorical features.

Standardizing the features: This ensures that the features have zero mean and unit variance, which is important for PCA.

Incremental PCA: This technique is useful for large datasets that cannot fit into memory. It processes the data in batches.
Ensure that your DataFrame df is loaded and preprocessed correctly.
The number of components for PCA is set to half the number of original features (size = len(features.columns) // 2).
The batch_size for IncrementalPCA is set to 500, but you can adjust it based on your dataset size and memory constraints.
# In[67]:


pip install dask


# In[68]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import IncrementalPCA

# Assuming df is your DataFrame with features and 'Attack Type'
# Separate features and target
features = df.drop('Attack Type', axis=1)
attacks = df['Attack Type']

# Step 1: Handle non-numeric values (replace '-' with NaN)
features = features.replace('-', np.nan)

# Step 2: Identify numeric columns properly
numeric_columns = features.select_dtypes(include=[np.number]).columns

# Step 3: Convert to float, handling errors='coerce' to handle NaNs
features[numeric_columns] = features[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Step 4: Impute missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features[numeric_columns])

# Verify there are no NaN values after imputation
assert not np.isnan(features_imputed).any(), "There are still NaN values after imputation."

# Step 5: Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_imputed)

# Apply IncrementalPCA
size = len(numeric_columns) // 2
ipca = IncrementalPCA(n_components=size, batch_size=500)
for batch in np.array_split(scaled_features, len(scaled_features) // 500):
    ipca.partial_fit(batch)

print(f'Information retained: {sum(ipca.explained_variance_ratio_):.2%}')

# If needed, transform the data using the fitted IPCA
transformed_features = ipca.transform(scaled_features)

# Optionally, convert transformed features back to a DataFrame and combine with the target variable
transformed_features_df = pd.DataFrame(transformed_features, columns=[f'PC{i+1}' for i in range(size)])
final_df = pd.concat([transformed_features_df, attacks.reset_index(drop=True)], axis=1)

# Display the final DataFrame
print(final_df.head())


# In[69]:


transformed_features = ipca.transform(scaled_features)
new_data = pd.DataFrame(transformed_features, columns = [f'PC{i+1}' for i in range(size)])
new_data['Attack Type'] = attacks.values
     


# In[70]:


new_data


# # Machine Learning Models

# In[71]:


# For cross validation
from sklearn.model_selection import cross_val_score

normal_traffic contains only 'BENIGN' records.
intrusions contains all non-'BENIGN' records.
sample_size is set to the smaller of the two datasets' lengths to ensure balance.

ids_data is formed by concatenating the sampled normal_traffic and intrusions.
'Attack Type' is converted to binary labels: 0 for 'BENIGN' and 1 for attacks.

# In[72]:


# Assuming new_data is your DataFrame with principal components and 'Attack Type'

# Create balanced dataset for binary classification
normal_traffic = new_data[new_data['Attack Type'] == 'BENIGN']
intrusions = new_data[new_data['Attack Type'] != 'BENIGN']

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


X_bc contains all the columns except 'Attack Type'.
y_bc contains the 'Attack Type' column, which is the target variable for classification.

train_test_split is used to split X_bc and y_bc into training and testing sets.
test_size=0.25 :  25% of the data will be used for testing, and 75% for training.
random_state=0 ensures that the split is reproducible.
# In[73]:


# Splitting the data into features (X) and target (y)
from sklearn.model_selection import train_test_split

X_bc = bc_data.drop('Attack Type', axis = 1)
y_bc = bc_data['Attack Type']

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size = 0.25, random_state = 0)

# Displaying the shapes of the resulting datasets
print(f'X_train_bc shape: {X_train_bc.shape}')
print(f'X_test_bc shape: {X_test_bc.shape}')
print(f'y_train_bc shape: {y_train_bc.shape}')
print(f'y_test_bc shape: {y_test_bc.shape}')
     


# # Binary Classification

# In[74]:


##Logistic Regression (Binary Classification)


# In[75]:


from sklearn.linear_model import LogisticRegression

lr1 = LogisticRegression(max_iter = 10000, C = 0.1, random_state = 0, solver = 'saga')
lr1.fit(X_train_bc, y_train_bc)

cv_lr1 = cross_val_score(lr1, X_train_bc, y_train_bc, cv = 5)
print('Logistic regression Model 1')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_lr1)))
print(f'\nMean cross-validation score: {cv_lr1.mean():.2f}')


# In[76]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on the test set
y_pred_bc = lr1.predict(X_test_bc)

# Calculate accuracy
test_accuracy = accuracy_score(y_test_bc, y_pred_bc)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test_bc, y_pred_bc))

# Print classification report
print('Classification Report:')
print(classification_report(y_test_bc, y_pred_bc))


# In[77]:


print('Logistic Regression Model 1 coefficients:')
print(*lr1.coef_, sep = ', ')
print('\nLogistic Regression Model 1 intercept:', *lr1.intercept_)


# In[78]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Predict on the test set
y_pred_bc = lr1.predict(X_test_bc)
y_pred_proba_bc = lr1.predict_proba(X_test_bc)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test_bc, y_pred_bc)
precision = precision_score(y_test_bc, y_pred_bc)
recall = recall_score(y_test_bc, y_pred_bc)
f1 = f1_score(y_test_bc, y_pred_bc)
auc = roc_auc_score(y_test_bc, y_pred_proba_bc)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'AUC: {auc:.2f}')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_bc, y_pred_proba_bc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()


# In[79]:


lr2 = LogisticRegression(max_iter = 15000, solver = 'sag', C = 100, random_state = 0)
lr2.fit(X_train_bc, y_train_bc)

cv_lr2 = cross_val_score(lr2, X_train_bc, y_train_bc, cv = 5)
print('Logistic regression Model 2')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_lr2)))
print(f'\nMean cross-validation score: {cv_lr2.mean():.2f}')

# Predictions
y_pred_bc = lr2.predict(X_test_bc)
y_pred_prob_bc = lr2.predict_proba(X_test_bc)[:, 1]


# # Comparison between Model 1 & 2
# Both lr1 and lr2 show perfect or near-perfect cross-validation scores. This consistency suggests the models are both performing very well. However, the same caution applies:
# 
# Overfitting: Both models achieving perfect scores may indicate potential overfitting.
# Validation on Unseen Data: Testing on a separate validation or test set not used in cross-validation is crucial.

# In[80]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predictions on the test set
y_pred_bc = lr2.predict(X_test_bc)

# Compute metrics
accuracy = accuracy_score(y_test_bc, y_pred_bc)
precision = precision_score(y_test_bc, y_pred_bc)
recall = recall_score(y_test_bc, y_pred_bc)
f1 = f1_score(y_test_bc, y_pred_bc)
auc = roc_auc_score(y_test_bc, y_pred_bc)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'AUC: {auc:.2f}')


# In[81]:


print('Logistic Regression Model 2 coefficients:')
print(*lr2.coef_, sep = ', ')
print('\nLogistic Regression Model 2 intercept:', *lr2.intercept_)


# In[82]:


from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_bc, y_pred_prob_bc)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test_bc, y_pred_prob_bc)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

##Support Vector Machine (Binary Classification)
# In[83]:


from sklearn.svm import SVC

svm1 = SVC(kernel = 'poly', C = 1, random_state = 0, probability = True)
svm1.fit(X_train_bc, y_train_bc)

cv_svm1 = cross_val_score(svm1, X_train_bc, y_train_bc, cv = 5)
print('Support Vector Machine Model 1')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_svm1)))
print(f'\nMean cross-validation score: {cv_svm1.mean():.2f}')
     


# In[84]:


from sklearn.metrics import roc_curve

# Predict probabilities for SVM
y_pred_prob_svm1 = svm1.predict_proba(X_test_bc)[:, 1]

# Compute ROC curve
fpr_svm1, tpr_svm1, _ = roc_curve(y_test_bc, y_pred_prob_svm1)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_svm1, tpr_svm1, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM Model 1')
plt.legend()
plt.show()


# In[85]:


svm2 = SVC(kernel = 'rbf', C = 1, gamma = 0.1, random_state = 0, probability = True)
svm2.fit(X_train_bc, y_train_bc)

cv_svm2 = cross_val_score(svm2, X_train_bc, y_train_bc, cv = 5)
print('Support Vector Machine Model 2')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_svm2)))
print(f'\nMean cross-validation score: {cv_svm2.mean():.2f}')


# In[86]:


# Predict probabilities for SVM
y_pred_prob_svm2 = svm2.predict_proba(X_test_bc)[:, 1]

# Compute ROC curve
fpr_svm2, tpr_svm2, _ = roc_curve(y_test_bc, y_pred_prob_svm2)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_svm2, tpr_svm2, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM Model 2')
plt.legend()
plt.show()


# In[87]:


print('SVM Model 1 intercept:', *svm1.intercept_)
print('SVM Model 2 intercept:', *svm2.intercept_)


# # Creating a Balanced Dataset for Multi-class Classification
# 

# In[88]:


new_data['Attack Type'].value_counts()


# In[89]:


class_counts = new_data['Attack Type'].value_counts()
selected_classes = class_counts[class_counts > 1500]
class_names = selected_classes.index
selected = new_data[new_data['Attack Type'].isin(class_names)]

dfs = []
for name in class_names:
    df = selected[selected['Attack Type'] == name]
    if len(df) > 5000:
        df = df.sample(n=5000, replace=False, random_state=0)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['Attack Type'].value_counts()


# In[90]:


import sklearn
import imblearn


from imblearn.over_sampling import SMOTE
import pandas as pd

# Assuming X and y are already defined
X = df.drop('Attack Type', axis=1)
y = df['Attack Type']

# Apply SMOTE to upsample the minority class
smote = SMOTE(sampling_strategy='auto', random_state=0)
X_upsampled, y_upsampled = smote.fit_resample(X, y)

# Create a new dataframe with the upsampled data
blnc_data = pd.DataFrame(X_upsampled)
blnc_data['Attack Type'] = y_upsampled
blnc_data = blnc_data.sample(frac=1, random_state=0)  # Shuffle the dataframe

# Check the class distribution to verify balance
print(blnc_data['Attack Type'].value_counts())


# In[91]:


#!pip uninstall scikit-learn --yes


# In[92]:


#!pip uninstall imblearn --yes


# In[93]:


get_ipython().system('pip install imblearn')


# In[94]:


get_ipython().system('pip install scikit-learn==1.2.2')


# In[102]:


features = blnc_data.drop('Attack Type', axis = 1)
labels = blnc_data['Attack Type']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
     


# # Training model

# In[96]:


import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Logistic Regression
lr = LogisticRegression(max_iter=10000, C=0.1, random_state=0, solver='saga')

# Measure training time for Logistic Regression
start_train_lr = time.time()
lr.fit(X_train, y_train)
end_train_lr = time.time()
train_time_lr = end_train_lr - start_train_lr

# Cross Validation
cv_lr = cross_val_score(lr, X_train, y_train, cv=5)
print('Logistic Regression')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_lr)))
print(f'\nMean cross-validation score: {cv_lr.mean():.2f}')

# Measure testing time for Logistic Regression
start_test_lr = time.time()
y_pred_lr = lr.predict(X_test)
y_pred_prob_lr = lr.predict_proba(X_test)
end_test_lr = time.time()
test_time_lr = end_test_lr - start_test_lr

# Evaluate Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
auc_lr = roc_auc_score(y_test, y_pred_prob_lr, multi_class='ovo')

print('Logistic Regression Model')
print(f'Training Time: {train_time_lr:.2f} seconds')
print(f'Testing Time: {test_time_lr:.2f} seconds')
print(f'Accuracy: {accuracy_lr:.2f}')
print(f'Precision: {precision_lr:.2f}')
print(f'Recall: {recall_lr:.2f}')
print(f'F1 Score: {f1_lr:.2f}')
print(f'AUC: {auc_lr:.2f}')

# Compute confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=lr.classes_)

# Create Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=lr.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


# SVM with RBF kernel
svm = SVC(kernel='rbf', C=1, gamma=0.1, random_state=0, probability=True)

# Measure training time for SVM
start_train_svm = time.time()
svm.fit(X_train, y_train)
end_train_svm = time.time()
train_time_svm = end_train_svm - start_train_svm

# Cross Validation
cv_svm = cross_val_score(svm, X_train, y_train, cv=5)
print('\nSVM with RBF kernel')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_svm)))
print(f'\nMean cross-validation score: {cv_svm.mean():.2f}')

# Measure testing time for SVM
start_test_svm = time.time()
y_pred_svm = svm.predict(X_test)
y_pred_prob_svm = svm.predict_proba(X_test)
end_test_svm = time.time()
test_time_svm = end_test_svm - start_test_svm

# Evaluate SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
auc_svm = roc_auc_score(y_test, y_pred_prob_svm, multi_class='ovo')

print('\nSVM Model')
print(f'Training Time: {train_time_svm:.2f} seconds')
print(f'Testing Time: {test_time_svm:.2f} seconds')
print(f'Accuracy: {accuracy_svm:.2f}')
print(f'Precision: {precision_svm:.2f}')
print(f'Recall: {recall_svm:.2f}')
print(f'F1 Score: {f1_svm:.2f}')
print(f'AUC: {auc_svm:.2f}')

# Compute confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm.classes_)

# Create Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=svm.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for SVM with RBF kernel')
plt.show()


# # Random Forest Classifier

# In[97]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


# Random Forest Model 1
rf1 = RandomForestClassifier(n_estimators=10, max_depth=6, max_features=None, random_state=0)
rf1.fit(X_train, y_train)

cv_rf1 = cross_val_score(rf1, X_train, y_train, cv=5)
print('Random Forest Model 1')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_rf1)))
print(f'\nMean cross-validation score: {cv_rf1.mean():.2f}')

y_pred_rf1 = rf1.predict(X_test)
y_pred_prob_rf1 = rf1.predict_proba(X_test)

accuracy_rf1 = accuracy_score(y_test, y_pred_rf1)
precision_rf1 = precision_score(y_test, y_pred_rf1, average='weighted')
recall_rf1 = recall_score(y_test, y_pred_rf1, average='weighted')
f1_rf1 = f1_score(y_test, y_pred_rf1, average='weighted')
auc_rf1 = roc_auc_score(y_test, y_pred_prob_rf1, multi_class='ovo', average='weighted')

print('Random Forest Model 1')
print(f'Accuracy: {accuracy_rf1:.2f}')
print(f'Precision: {precision_rf1:.2f}')
print(f'Recall: {recall_rf1:.2f}')
print(f'F1 Score: {f1_rf1:.2f}')
print(f'AUC: {auc_rf1:.2f}')

# Compute confusion matrix
cm_rf1 = confusion_matrix(y_test, y_pred_rf1, labels=rf1.classes_)

# Create Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf1, display_labels=rf1.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for Random Forest Model 1')
plt.show()

# Random Forest Model 2
rf2 = RandomForestClassifier(n_estimators=15, max_depth=8, max_features=20, random_state=0)
rf2.fit(X_train, y_train)

cv_rf2 = cross_val_score(rf2, X_train, y_train, cv=5)
print('Random Forest Model 2')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_rf2)))
print(f'\nMean cross-validation score: {cv_rf2.mean():.2f}')

y_pred_rf2 = rf2.predict(X_test)
y_pred_prob_rf2 = rf2.predict_proba(X_test)

accuracy_rf2 = accuracy_score(y_test, y_pred_rf2)
precision_rf2 = precision_score(y_test, y_pred_rf2, average='weighted')
recall_rf2 = recall_score(y_test, y_pred_rf2, average='weighted')
f1_rf2 = f1_score(y_test, y_pred_rf2, average='weighted')
auc_rf2 = roc_auc_score(y_test, y_pred_prob_rf2, multi_class='ovo', average='weighted')

print('\n \nRandom Forest Model 2')
print(f'Accuracy: {accuracy_rf2:.2f}')
print(f'Precision: {precision_rf2:.2f}')
print(f'Recall: {recall_rf2:.2f}')
print(f'F1 Score: {f1_rf2:.2f}')
print(f'AUC: {auc_rf2:.2f}')


# Compute confusion matrix
cm_rf2 = confusion_matrix(y_test, y_pred_rf2, labels=rf2.classes_)

# Create Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf2, display_labels=rf2.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for Random Forest Model 2')
plt.show()


# In[104]:


import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Random Forest Model 1
rf1 = RandomForestClassifier(n_estimators=100, max_depth=6, max_features=None, random_state=0)

# Measure training time for Random Forest Model 1
start_train_rf1 = time.time()
rf1.fit(X_train, y_train)
end_train_rf1 = time.time()
train_time_rf1 = end_train_rf1 - start_train_rf1

# Cross-validation
cv_rf1 = cross_val_score(rf1, X_train, y_train, cv=5)
print('Random Forest Model')
print(f'\nCross-validation scores:', ', '.join(map(str, cv_rf1)))
print(f'\nMean cross-validation score: {cv_rf1.mean():.2f}')

# Measure testing time for Random Forest Model 1
start_test_rf1 = time.time()
y_pred_rf1 = rf1.predict(X_test)
y_pred_prob_rf1 = rf1.predict_proba(X_test)
end_test_rf1 = time.time()
test_time_rf1 = end_test_rf1 - start_test_rf1

# Evaluate Random Forest Model 1
accuracy_rf1 = accuracy_score(y_test, y_pred_rf1)
precision_rf1 = precision_score(y_test, y_pred_rf1, average='weighted')
recall_rf1 = recall_score(y_test, y_pred_rf1, average='weighted')
f1_rf1 = f1_score(y_test, y_pred_rf1, average='weighted')
auc_rf1 = roc_auc_score(y_test, y_pred_prob_rf1, multi_class='ovo', average='weighted')

print('Random Forest Model 1')
print(f'Training Time: {train_time_rf1:.2f} seconds')
print(f'Testing Time: {test_time_rf1:.2f} seconds')
print(f'Accuracy: {accuracy_rf1:.2f}')
print(f'Precision: {precision_rf1:.2f}')
print(f'Recall: {recall_rf1:.2f}')
print(f'F1 Score: {f1_rf1:.2f}')
print(f'AUC: {auc_rf1:.2f}')

# Compute confusion matrix
cm_rf1 = confusion_matrix(y_test, y_pred_rf1, labels=rf1.classes_)

# Create Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf1, display_labels=rf1.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for Random Forest Model ')
plt.show()


# Random Forest Model 2
rf2 = RandomForestClassifier(n_estimators=15, max_depth=8, max_features=20, random_state=0)

# Measure training time for Random Forest Model 2
start_train_rf2 = time.time()
rf2.fit(X_train, y_train)
end_train_rf2 = time.time()
train_time_rf2 = end_train_rf2 - start_train_rf2



# # Bar Chat

# In[99]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define models and their corresponding metrics
models = ['Logistic Regression', 'SVM (RBF)', 'Random Forest Model 1', 'Random Forest Model 2']
cv_scores = [cv_lr.mean(), cv_svm.mean(), cv_rf1.mean(), cv_rf2.mean()]
accuracy_scores = [accuracy_lr, accuracy_svm, accuracy_rf1, accuracy_rf2]
precision_scores = [precision_lr, precision_svm, precision_rf1, precision_rf2]
recall_scores = [recall_lr, recall_svm, recall_rf1, recall_rf2]
f1_scores = [f1_lr, f1_svm, f1_rf1, f1_rf2]

# Plotting cross-validation scores
plt.figure(figsize=(12, 6))
plt.bar(models, cv_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Cross-validation Mean Score')
plt.title('Cross-validation Mean Scores for Different Models')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting accuracy, precision, recall, and F1 score
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy_scores, precision_scores, recall_scores, f1_scores]

plt.figure(figsize=(14, 10))

for i, metric in enumerate(metrics, start=1):
    plt.subplot(2, 2, i)
    plt.bar(models, scores[i-1], color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
    plt.xlabel('Models')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison for Different Models')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()


# In[100]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# Define classes based on your specific dataset
classes = ['BENIGN','LDAP','MSSQL','NetBIOS','Portmap','UDP']  # Replace with actual class labels

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
plot_confusion_matrix(cm_lr, classes, title='Confusion Matrix - Logistic Regression')

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plot_confusion_matrix(cm_svm, classes, title='Confusion Matrix - SVM with RBF Kernel')

# Confusion Matrix for Random Forest Model 1
cm_rf1 = confusion_matrix(y_test, y_pred_rf1)
plot_confusion_matrix(cm_rf1, classes, title='Confusion Matrix - Random Forest Model 1')

# Confusion Matrix for Random Forest Model 2
cm_rf2 = confusion_matrix(y_test, y_pred_rf2)
plot_confusion_matrix(cm_rf2, classes, title='Confusion Matrix - Random Forest Model 2')


# # KNN Classifier
# 

# In[101]:


import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Initialize K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=5)

# Measure training time for K-Nearest Neighbors
start_train_knn = time.time()
knn.fit(X_train, y_train)
end_train_knn = time.time()
train_time_knn = end_train_knn - start_train_knn

# Cross Validation for KNN
cv_knn = cross_val_score(knn, X_train, y_train, cv=5)
print('\nK-Nearest Neighbors')
print(f'\nCross-validation scores: {", ".join(map(str, cv_knn))}')
print(f'\nMean cross-validation score: {cv_knn.mean():.2f}')

# Measure testing time for K-Nearest Neighbors
start_test_knn = time.time()
y_pred_knn = knn.predict(X_test)
y_pred_prob_knn = knn.predict_proba(X_test)
end_test_knn = time.time()
test_time_knn = end_test_knn - start_test_knn

# Evaluate KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
auc_knn = roc_auc_score(y_test, y_pred_prob_knn, multi_class='ovo')

print('\nK-Nearest Neighbors Model')
print(f'Training Time: {train_time_knn:.2f} seconds')
print(f'Testing Time: {test_time_knn:.2f} seconds')
print(f'Accuracy: {accuracy_knn:.2f}')
print(f'Precision: {precision_knn:.2f}')
print(f'Recall: {recall_knn:.2f}')
print(f'F1 Score: {f1_knn:.2f}')
print(f'AUC: {auc_knn:.2f}')

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)

# Create Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for K-Nearest Neighbors')
plt.show()

