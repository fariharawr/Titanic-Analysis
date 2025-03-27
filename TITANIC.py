#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv("train.csv")
df.head()


# In[4]:


df.tail()
df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


#correlation
corr=df.corr()
corr


# In[8]:


plt.subplots(figsize=(15,10))
sns.heatmap(corr,annot=True)


# In[9]:


df.Survived.value_counts()


# In[10]:


#eliminate ticket, name, body(alot are null, unuseable),boat


# In[11]:


df.Sex.value_counts()


# In[12]:


df.Embarked.value_counts()


# In[13]:


#CHECK NULL
df.isnull().any()


# In[14]:


df.isnull().sum()


# In[15]:


df['Age']


# In[16]:


#fill null age with mean 
mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)


# In[17]:


most_common_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(most_common_embarked, inplace=True)


# In[18]:


df.drop(['Cabin','Name','Ticket'],axis=1, inplace=True)


# In[19]:


#VISUALISATION


# In[20]:


# Visualize the distribution of the 'Survived' column (0 = Not Survived, 1 = Survived)
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


# In[21]:


#Visualize the distribution of the 'Age' column
sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[22]:


#Visualize the distribution of the 'Fare' column and detect outliers we will handle outliers in the next step
sns.boxplot(data=df, x='Fare')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.show()


# In[23]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# # detect and handle outliers with z score

# In[24]:


# z>3 data could be outlier


# In[25]:


z_scores = np.abs(stats.zscore(df['Age']))
max_threshold=3

outliers = df['Age'][z_scores > max_threshold]

# Print and visualize the outliers
print("Outliers detected using Z-Score:")
print(outliers)


# In[26]:


column_name = 'Fare'

# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = df[column_name].quantile(0.25)
Q3 = df[column_name].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1
# Calculate the IQR
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter rows with values outside the IQR bounds
df_cleaned = df[(df[column_name] > lower_bound) & (df[column_name] <upper_bound)]

# Display the original and cleaned DataFrame sizes
print(f"Original DataFrame size: {df.shape}")
print(f"Cleaned DataFrame size: {df_cleaned.shape}")
df_cleaned


# In[27]:


df=df_cleaned

x=df.drop('Survived', axis=1)
y=df['Survived']
x.head()


# In[28]:


#encode 
# LabelEncoder; first category is 0, the second is 1, and so on.
en = LabelEncoder()
x['Sex'] = en.fit_transform(x['Sex'])


# In[29]:


x.head()


# In[30]:


x = pd.get_dummies(x,columns=['Embarked'])
#binary variables indicate presence of a particular category.
#mecah jawabannya yg ada (s dan q : embarked_s and embarked_q)
x.head()


# In[31]:


#feature scaling
scale = StandardScaler()
x[['Age', 'Fare']] = scale.fit_transform(x[['Age', 'Fare']])
x.head()


# In[32]:


#train and test; generalize model generalize new data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[33]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(x_train,y_train)
# print the intercept : predicted  baseline probability of the event 
# when input is 0
print(lm.intercept_)
# best if intercept that is close to zero. 
# indicates model is well-calibrated and is not overfitting or underfitting the training data.


# In[34]:


predictions = lm.predict(x_test)
accuracy = np.mean(predictions == y_test)
print('Accuracy:', accuracy)


# In[35]:


#abis interpreting coefficient mre mse
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[36]:


# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# Check the rest of  classifiers
name = SVC()
name.fit(x_train, y_train)
Y_pred = name.predict(x_test)
acc_svc = round(name.score(x_train, y_train) * 100, 2)
acc_svc


# In[37]:


# from sklearn.linear_model import LogisticRegression
#Support Vector Machine
from sklearn.svm import SVC, LinearSVC
name = SVC()
name.fit(x_train, y_train)
Y_pred = name.predict(x_test)
acc_svc = round(name.score(x_train, y_train) * 100, 2)
acc_svc


# In[38]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# Check the rest of  classifiers
name = KNeighborsClassifier()
name.fit(x_train, y_train)
Y_pred = name.predict(x_test)
acc_svc = round(name.score(x_train, y_train) * 100, 2)
acc_svc


# In[39]:


Y_pred


# In[40]:


y_train


# Compare the results of machine learning techniques, what model predict better for titanic survival ?
# 
# Change the target of prediction from survived to Age, and compare the results of machine learning techniques, what model predict better for titanic survival?

# Part I: Look at the heatmap to find the most important features that are correlated with each other. Then, think of two hypotheses about how these features might be related to survival. Finally, use descriptive statistics to test your hypotheses. 
# 
# SibSp(n of pasenger;s sibling) and Parch(n of spuse/child in the ship) : 0.41 
# #2 the more siblings they have, 
#     CF:the more likely they will travel with more of their fam members
#     DS: scatter plot doesnt show any relation 
# #3 the more siblings, the more likely they will survive 
#     CF: help from other siblings n so on.
#     DS: mean of SibSp and Parch for non-survivor is higher than survivor> WRONG, the corr shows the more sibs, more likely to not survive
# 
# Survived and Fare: 0.26
# #1 The higher the tix the more likely they will survive
#     CF: higher quality if price is higher, safer, less casualties
#     DS: higher fare mean for survivor TRUE?
# 
# 
# Part II: How does the mean age of survivors differ from the mean age of non-survivors in a random sample of people (let us say 50 people)?
# bcs the two groups are not the same people, as age might vary in both, it is 

# In[41]:


# Create a scatter plot
plt.scatter(df['SibSp'], df['Parch'])
# Label
plt.xlabel('SibSp')
plt.ylabel('Parch')
plt.show()
#trend is not easy to define


# In[42]:


# Calculate the mean and standard deviation of SibSp and Parch for passengers who survived
#low sd : clustered around mean
survivors = df[df['Survived'] == 1]
survivor_sibsp_mean = survivors['SibSp'].mean()
survivor_sibsp_std = survivors['SibSp'].std()
survivor_parch_mean = survivors['Parch'].mean()
survivor_parch_std = survivors['Parch'].std()

# Calculate the mean and standard deviation of SibSp and Parch for passengers who did not survive
non_survivors = df[df['Survived'] == 0]
non_survivor_sibsp_mean = non_survivors['SibSp'].mean()
non_survivor_sibsp_std = non_survivors['SibSp'].std()
non_survivor_parch_mean = non_survivors['Parch'].mean()
non_survivor_parch_std = non_survivors['Parch'].std()

# Print the results
print('Mean SibSp for survivors:', survivor_sibsp_mean)
print('Standard deviation of SibSp for survivors:', survivor_sibsp_std)
print('Mean SibSp for non-survivors:', non_survivor_sibsp_mean)
print('Standard deviation of SibSp for non-survivors:', non_survivor_sibsp_std)
print('Mean Parch for survivors:', survivor_parch_mean)
print('Standard deviation of Parch for survivors:', survivor_parch_std)
print('Mean Parch for non-survivors:', non_survivor_parch_mean)
print('Standard deviation of Parch for non-survivors:', non_survivor_parch_std)


# In[43]:


# Calculate the mean and standard deviation of Fare and Survived for passengers who survived
#low sd : clustered around mean
survivors = df[df['Survived'] == 1]
survivor_fare_mean = survivors['Fare'].mean()
survivor_fare_std = survivors['Fare'].std()

non_survivors = df[df['Survived'] == 0]
non_survivor_fare_mean = non_survivors['Fare'].mean()
non_survivor_fare_std = non_survivors['Fare'].std()

# Print the results
print('Mean Fare for survivors:', survivor_fare_mean)
print('Standard deviation of Fare for survivors:', survivor_fare_std)
print('Mean fare for non-survivors:', non_survivor_fare_mean)
print('Standard deviation of fare for non-survivors:', non_survivor_fare_std)


# In[44]:


# Create a scatter plot
plt.scatter(df['Fare'], df['Survived']==1)


# In[49]:


import pandas as pd
df=pd.read_csv(r"train.csv")
mean_age = df['Age'].mean()
print(df.isnull().sum())


# In[50]:


df['Age'].fillna(mean_age, inplace=True)
most_common_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(most_common_embarked, inplace=True)
print(df.isnull().sum())
print(df.notnull().sum())
# 687/ 891, ~ 77 missing data, too much! So we drop Cabin
df.drop(['Cabin'],axis=1, inplace=True)
print(df.isnull().sum())


# In[51]:


import scipy.stats as stats
import numpy as np
z_scores = np.abs(stats.zscore(df['Age']))
max_threshold=3

outliers = df['Age'][z_scores > max_threshold]

# Print and visualize the outliers
print("Outliers detected using Z-Score:")
print(outliers)


# In[52]:


column_name = 'Age'

# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = df[column_name].quantile(0.25)
Q3 = df[column_name].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter rows with values outside the IQR bounds
# df_cleaned = df[(df[column_name] > lower_bound) & (df[column_name] <upper_bound)]

# Replace outliers with median
df.loc[(df[column_name] < lower_bound) | (df[column_name] > upper_bound), column_name] = df[column_name].median()

# Display the original and cleaned DataFrame sizes
print(f"Original DataFrame size: {df.shape}")
print(f"Cleaned DataFrame size: {df.shape}")


# In[53]:


# Import the stats module from scipy library
import scipy.stats as stats
import numpy as np

# Extract the age values of survivors and non-survivors from the dataset
survived_age = df.loc[df['Survived'] == 1, 'Fare']
died_age = df.loc[df['Survived'] == 0, 'Fare']

# State the null and alternative hypotheses
# H0: μ1 = μ2 (where μ1 is the mean age of survivors and μ2 is the mean age of non-survivors)
# H1: μ1 ≠ μ2

# Use the stats.ttest_ind function to calculate the t-statistic and p-value
t_stat, p_value = stats.ttest_ind(survived_age, died_age)

# Print the results
print(f'The t-statistic is {t_stat:.2f} and the p-value is {p_value:.2f}')

# Interpret the results
if p_value < 0.05:
    print('We reject the null hypothesis and conclude that there is a statistically significant difference in the mean age of survivors and non-survivors.')
else:
    print('We conclude that there is no statistically significant difference in the mean age of survivors and non-survivors.')

# Calculate the effect size : a common measure of effect size for comparing the means of two independent groups.
effect_size = t_stat / np.sqrt(len(survived_age) + len(died_age) - 2)

# Print the effect size
print(f'The effect size is {effect_size:.2f}')


# In[61]:


# Import pandas and numpy libraries
import pandas as pd
import numpy as np

# Select a random sample of 50 people from the dataset
sample = df.sample(n=50, random_state=42)


# Extract the Fare values of survivors and non-survivors from the dataset
survived_Fare= sample.loc[df['Survived'] == 1, 'Fare']
died_Fare = sample.loc[df['Survived'] == 0, 'Fare']

# Use the stats.ttest_ind function to calculate the t-statistic and p-value
t_stat, p_value = stats.ttest_ind(survived_Fare, died_Fare)


# Define the hypotheses and alternative hypotheses
hypothesis = "there is no difference in the mean Fare of survivors and non-survivors."
alternative_hypothesis = "there is a difference in the mean Fare of survivors and non-survivors."


# Print the results
print(f'The t-statistic is {t_stat:.2f} and the p-value is {p_value:.2f}')

# Print the p-value
print(f"p-value = {p_value:.4f}")

# Compare the p-value with the significance level and make a decision
if p_value <= 0.05:
    print(f"Since p-value {p_value:.4f} is less than or equal to alpha 0.05, we reject the hypothesis and conclude that {alternative_hypothesis}.")
else:
    print(f"Since p-value {p_value:.4f} is greater than alpha 0.05, we fail to reject the hypothesis.")

# Report and interpret the results
print(f"Our conclusion is that based on a random sample of 50 people from the Titanic dataset, {alternative_hypothesis} at a significance level of 0.5.")


# In[ ]:




