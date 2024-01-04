#!/usr/bin/env python
# coding: utf-8

# # Task 1 Import Libraries and combine the train and test datasets.
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')
# 
# train_df['source'] = 'train'
# test_df['source'] = 'test'
# combined_df = pd.concat([train_df, test_df], ignore_index=True)
# 
# 
# combined_df.to_csv('combined_data.csv', index=False)
# 

# In[21]:


dataset_path = 'combined_data.csv'  # Replace with the actual path to your dataset
df = pd.read_csv(dataset_path)


# # Analyse the total combine data and get the insights of the data before start any oeprations a 1-9.

# In[20]:


print(df.head())


# In[10]:


num_rows, num_columns = df.shape

# Print the results
print(f'Total number of rows: {num_rows}')
print(f'Total number of columns: {num_columns}')


# In[7]:


print(df.info())


# In[8]:


print("\nSummary statistics of the dataset:")
print(df.describe())


# In[9]:


print("\nMissing values in the dataset:")
print(df.isnull().sum())


# In[12]:


# Choose the column for which you want to get unique values
selected_column = 'Arrival Delay in Minutes'  # Replace with the actual column name

# Get all unique values from the selected column
unique_values = df[selected_column].unique()

# Print the unique values
print(f'Unique values in the column "{selected_column}":')
print(unique_values)


# In[13]:


# Count the number of NaN values in each column
nan_counts = df.isnull().sum()

# Print the results
print("Number of NaN values in each column:")
print(nan_counts)


# # 1a Four univariate plots (Histogram plot - Kernal Density Plot - Box Plot - Violin Plot)

# In[15]:


selected_column = 'Age'

# Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(df[selected_column], bins=30, color='skyblue', edgecolor='black')
plt.title(f'Histogram - {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Frequency')
plt.show()


# # Summary 1a: The majority travellers were of the young age groups (25-30) and middle age group( 45-55).

# plt.figure(figsize=(8, 6))
# sns.histplot(df[selected_column], bins=30, color='skyblue', kde=True)
# plt.title(f'Histogram - {selected_column}')
# plt.xlabel(selected_column)
# plt.ylabel('Frequency')
# plt.show()

# In[17]:


plt.figure(figsize=(8, 6))
sns.kdeplot(df[selected_column], color='orange', shade=True)
plt.title(f'Kernel Density Plot - {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Density')
plt.show()


# # # Summary 1a:Moreover there were very few retired traveller passengers of age > 80.

# In[18]:


# Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x=df[selected_column], color='lightgreen')
plt.title(f'Box Plot - {selected_column}')
plt.xlabel(selected_column)
plt.show()


# In[19]:


plt.figure(figsize=(8, 6))
sns.violinplot(x=df[selected_column], color='lightcoral')
plt.title(f'Violin Plot - {selected_column}')
plt.xlabel(selected_column)
plt.show()


# # Summary 1a: The Center Point for the travellers class was at 40. 

# # 2a  Three bivariate plots (Scatter Plot- line Plot - Joint Plot)

# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[27]:


x_column = 'Age'
y_column = 'Type of Travel'

# Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(df[x_column], df[y_column], color='blue', alpha=0.5)
plt.title(f'Scatter Plot - {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()


# # Summary 2a: The business travellers count was more Compare to total number of personal travellers.

# In[28]:


# Seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_column, y=y_column, data=df, color='blue', alpha=0.5)
plt.title(f'Scatter Plot - {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()


# # Summary 2a: The business travellers age group vary from (9-80) while it was between (9-72) for personal travellers.

# In[29]:


# Seaborn
plt.figure(figsize=(8, 6))
sns.lineplot(x=x_column, y=y_column, data=df, color='green')
plt.title(f'Line Plot - {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()


# # Summary 2a: The business travellers contains Outlies while there was no any outliers in for personal travellers.

# In[30]:


# Seaborn
plt.figure(figsize=(8, 6))
sns.jointplot(x=x_column, y=y_column, data=df, kind='scatter', color='purple')
plt.suptitle(f'Joint Plot - {x_column} vs {y_column}', y=1.02)
plt.show()


# # Summary 2a: The maximum age group for personal travellers was of 72.

# # 3a  A descriptive statistics

# In[32]:


# Display descriptive statistics for the entire DataFrame
print("Descriptive Statistics:")
print(df.describe())

# Display additional statistics for non-numeric columns
print("\nDescriptive Statistics for Non-Numeric Columns:")
print(df.describe(include='all'))


# # Summary 3a :  The comparaive mean values among the features like Flight Distance -1190.316392 , in flight wifi service- 2.728696, departure -arrival time convinient- 3.057599 and ease of online booking -2.756876. while the non numerival data like Arrival Delay in Minutes contains total 393 Outliers.

# # 4A Heatmap to show the correlation among the features

# In[33]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # Summary 4a : From the above heatmap we can clearly visualise some of the important feature relations.

# # There is strong relations between the pair of features like (Inflight wifi service - Ease of online booking ) , (Inflight Entertainment - cleanliness) and  (Arrival - Departure delay minutes)

# # 5a  A missing values plot/summary that shows which feature(s) has missing values

# In[34]:


# Count the number of NaN values in each column
nan_counts = df.isnull().sum()

# Print the results
print("Number of NaN values in each column:")
print(nan_counts)


# # Summary 5a : we can see that the arrival delay in minutes has missing values.

# In[35]:


selected_column = 'Arrival Delay in Minutes'

# Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(df[selected_column], bins=30, color='skyblue', edgecolor='black')
plt.title(f'Histogram - {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Frequency')
plt.show()


# # Summary 5a : we can Visualise that the arrival delay in minutes has missing values through histogram.

# # 6a   Atleast two plots to prove/disprove if the "Flight Distance" feature has any outliers.

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


dataset_path = 'combined_data.csv'  # Replace with the actual path to your dataset
df = pd.read_csv(dataset_path)
selected_column = 'Flight Distance'

# Create a box plot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x=df[selected_column], color='lightgreen')
plt.title(f'Box Plot - {selected_column}')
plt.xlabel(selected_column)
plt.show()


# # Summary 6a : In the box plot, potential outliers are usually represented as individual points beyond the whiskers.Points outside the whiskers are considered potential outliers in the above plot for the feature- flight distance.

# # Statistical methods to programmatically identify outliers. One common method is the IQR (Interquartile Range) method.

# In[9]:


# Calculate the IQR for the selected column
Q1 = df[selected_column].quantile(0.25)
Q3 = df[selected_column].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]

# Display the outliers
print(f'Outliers in {selected_column}:')
print(outliers)


# # 7A Atleast one plot/summary to prove/disprove if the "Arrival or Departure Delay in Minutes" has extreme delays.

# In[38]:


# Choose two columns for comparison
x_column = 'Departure Delay in Minutes'
y_column = 'Arrival Delay in Minutes'

# Matplotlib Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df[x_column], df[y_column], color='blue', alpha=0.5)
plt.title(f'Scatter Plot - {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()

# Seaborn Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_column, y=y_column, data=df, color='blue', alpha=0.5)
plt.title(f'Scatter Plot - {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()


# In[39]:


# Seaborn Line Plot
plt.figure(figsize=(8, 6))
sns.lineplot(x=x_column, y=y_column, data=df, color='green')
plt.title(f'Line Plot - {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()


# In[41]:


selected_column = 'Departure Delay in Minutes'

# Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(df[selected_column], bins=30, color='skyblue', edgecolor='black')
plt.title(f'Histogram - {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Frequency')
plt.show()


# In[42]:


selected_column = 'Arrival Delay in Minutes'

# Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(df[selected_column], bins=30, color='skyblue', edgecolor='black')
plt.title(f'Histogram - {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Frequency')
plt.show()


# # Summary 7a: I take the help of "Matplotlib Scatter Plot" and "Seaborn Scatter Plot"  "Seaborn line plot " and "Histogram plot " to compare extreme delays in Arrival or Departure Delay in Minutes.But as per my opinion both features have same values. there is not any extreme delays.

# # 8A Using plot/summary, prove/disprove that the majority of passengers voted that the seats were comfortable. Use the rating of "Departure/Arrival Time Convenient" feature.

# In[43]:


selected_column = 'Departure/Arrival time convenient'

# Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(df[selected_column], bins=30, color='skyblue', edgecolor='black')
plt.title(f'Histogram - {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Frequency')
plt.show()


# # summary 8a: By analysing the above graph I strongly believe that majority passengers agree in the seats were comfortable.The reason can be see by getting the maximum comfort vote at level -4.

# # 9A Suggest data cleaning steps based on your findings, or you could say the data is clean, then you need to explain why. 

# In[44]:


df[df.isnull().any(axis=1)].head()


# In[10]:


df['Arrival Delay in Minutes'].fillna(0, inplace=True)

# Save the modified DataFrame to a new CSV file
df.to_csv('your_modified_dataset1.csv', index=False)


# In[47]:


dataset_path = 'your_modified_dataset1.csv'  # Replace with the actual path to your dataset
df1 = pd.read_csv(dataset_path)


# In[48]:


df1.shape


# # Summary 9a: among 129880 * 26 , The feature Arrival Delay in Minutes contains Nan values-393. But this feature is important for my research . So inplace of drop I have replace those values by "0". 

# # b Use the train.csv file only to build and train a decision tree classifier to predict customer satisfaction. This feature has two different values which are "neutral or dissatisfied" and "satisfied". 

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[51]:


# Load the dataset into a DataFrame
dataset_path = 'train.csv'
df = pd.read_csv(dataset_path)


# In[52]:


# Assuming 'target_column' is the column you want to predict
target_column = 'satisfaction'

# Extract features (X) and target variable (y)
X = df.drop(target_column, axis=1)  # Features
y = df[target_column]  # Target variable

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)


# # c Use the test.csv file to test your classifier.

# In[53]:


# Load the dataset into a DataFrame
dataset_path = 'test.csv'
df = pd.read_csv(dataset_path)


# In[54]:


# Assuming 'target_column' is the column you want to predict
target_column = 'satisfaction'

# Extract features (X) and target variable (y)
X = df.drop(target_column, axis=1)  # Features
y = df[target_column]  # Target variable

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)


# # d The Result of precision, recall, f1-score as well support is mentioned in b and C seperately

# # Summary E : The features  like Satisfaction with- neutral or dissatisfied       0.95  ,    0.95  ,    0.95  ,   11713, and Arrival Delay in Minutes  were played very crucial role in the classifier b.
