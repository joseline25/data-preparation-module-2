import pandas as pd
import numpy as np
import random

# Counter is a dict subclass for counting hashable objects
from collections import Counter

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings in the notebook
import warnings
warnings.filterwarnings("ignore")

# to display up to 500 rows in the output of the jupyter notebook cell
pd.set_option('display.max_rows', 500)

# The objective is simple: use machine learning to create a model that predicts
# which passengers survived the Titanic shipwreck.

# let's start with data preparation

# import training data as pandas dataframe
# the data is in csv - comma separated file. Hence we use the function 'read_csv'
# train = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/titanic_data.csv')

train = pd.read_csv('../module_2/datasets/titanic_data.csv')
# the below code will print the number of rows and columns
print(train.shape)  # (891, 12)
print(train.columns)

"""
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
"""

# the test dataset
test = pd.read_csv('../module_2/datasets/titanic_test.csv')

# print the numbers of rows n columns of the test dataset
# the target has been removed
print(test.shape)  # (418, 11)
print(test.columns)

"""
Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
"""

# the target is  Survived

# print the columns types

print(train.dtypes)

"""
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: object
"""

print(train.iloc[0])  # la première entrée du dataset classée comme suit

"""
PassengerId                          1
Survived                             0
Pclass                               3
Name           Braund, Mr. Owen Harris
Sex                               male
Age                               22.0
SibSp                                1
Parch                                0
Ticket                       A/5 21171
Fare                              7.25
Cabin                              NaN
Embarked                             S
Name: 0, dtype: object
"""

print(train.iloc[[0]])  # la première entrée du dataset classée comme suit

"""
   PassengerId  Survived  Pclass                     Name   Sex   Age  SibSp  Parch     Ticket  Fare Cabin Embarked
0            1         0       3  Braund, Mr. Owen Harris  male  22.0      1      0  A/5 21171  7.25   NaN        S
"""

# extract the 21th row (entry) of the dataset

print(train.iloc[[20]])

"""
    PassengerId  Survived  Pclass                  Name   Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked
20           21         0       2  Fynney, Mr. Joseph J  male  35.0      0      0  239865  26.0   NaN        S
"""


# 1 - detect columns with missing values

print(train.isnull().sum())

"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
"""

# display only the columns with missing values
columns_missing_values = (train.isnull().sum())
print(columns_missing_values[columns_missing_values > 0])

"""
Age         177
Cabin       687
Embarked      2
dtype: int64
"""

# display only the columns with missing values with percentage!!

print(columns_missing_values[columns_missing_values > 0]/train.shape[0] * 100)

"""
Age         19.865320
Cabin       77.104377
Embarked     0.224467
dtype: float64
"""

# 2 - get the statistics of numerical columns

print(train.describe())

"""
       PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
"""


# We do not care about PassengerId column since it is an index column

# let's transform Survived and PcClass into categorical columns 
train['Survived'] = train['Survived'].astype(str)
train['Pclass'] = train['Pclass'].astype(str)


# 3 - number of unique values in a column: Survived for example

print(train['Survived'].value_counts()) 

"""
Survived
0    549
1    342
"""

# the frequency

print(train['Survived'].value_counts(normalize=True)*100)

"""
Survived
0    61.616162
1    38.383838
Name: proportion, dtype: float64
"""

# Drop irrelevant columns - Ticket and Name (may be passenger ID too - 
# if not set it as index)
del train['Name']
del train['Ticket']
del train['PassengerId']

# 5 -  Missing value treatment

# missing values - too many missing values - dropping entire column
del train['Cabin']

# missing values in numeric column many not be NaN or blank. It could be zero 
# as well

# for example, the Fare cannot be zero in out dataset. So the missing value in
# Fare columns are extracted by filtering for zero

# filter for Fare = 0 and display the shape of the dataframe - 15 rows
print(train[train['Fare'] == 0].shape) # (15, 8)

# There are only few rows with missing values in Fare - Listwise or dropping entire rows
train = train[train['Fare'] != 0]
# shape of the training data after dropping rows with missing Fare
print(train.shape) # (876, 8)

# Importing SimpleImputer from sklearn - 
# this will be used to impute data in the cells with missing values

from sklearn.impute import SimpleImputer

# missing values - numeric - impute with mean in column age
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer = mean_imputer.fit(train[['Age']])
train['Age'] = mean_imputer.transform(train[['Age']]).ravel()

# missing values - categorical - impute with mode (most frequent)
mode_imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
mode_imputer = mode_imputer.fit(train[['Embarked']])
train['Embarked'] = mode_imputer.transform(train[['Embarked']]).ravel()

# alternative method - missing values - categorical - impute with mode (constant)
const_imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value = 'NA')
const_imputer = const_imputer.fit(train[['Embarked']])
train['Embarked'] = const_imputer.transform(train[['Embarked']]).ravel()

# Export the data to csv file and manually check in Excel how the values have been imputed
# Do this for intuituively understanding how the Imputer works
# You can replace the above strategy with median to check how the results differ
train.to_csv('train_imputed.csv')


# 6 - Determine outlier using the Standard deviation method for Age column

# calcuate the mean of age
age_mean = train['Age'].mean()
print(age_mean) # 29.645219236209336

# calculate the standard deviation
age_std = train['Age'].std()
print(age_std) # 13.077539516985095

# Lower limit threshold is Mean - 3* SD
ll = age_mean - (3 * age_std)
print(ll) # -9.587399314745944

# Higher limit threshold is Mean + 3* SD
hh = age_mean + (3 * age_std)
print(hh) # 68.87783778716462

# filter the rows where Age is an outlier. i.e. Age less than 
# the ll or greater than hh

# the usage of & between means it is an and condition
# the usage of | between means it is an or condition
filt_outliers_train = train[(train['Age'] < ll) | (train['Age'] > hh)]
print(filt_outliers_train.head())

"""
    Survived Pclass   Sex   Age  SibSp  Parch     Fare Embarked
96         0      1  male  71.0      0      0  34.6542        C
116        0      3  male  70.5      0      0   7.7500        Q
493        0      1  male  71.0      0      0  49.5042        C
630        1      1  male  80.0      0      0  30.0000        S
672        0      2  male  70.0      0      0  10.5000        S
"""

# the distribution of Age

sns.distplot(train['Age'])
plt.show()
# it is almost a normal distribution

# continue with the management of outliers
# get the shape of the outliers dataset

print(filt_outliers_train.shape) # (7, 8) ie 7 outliers

# Handle the outliers with IQR (Inter Quartile Range)

# IQR method for outlier Age
# Calculate Q1, Q2 and IQR
q1 = train['Age'].quantile(0.25)    # first quartile            
q3 = train['Age'].quantile(0.75)    # second quartile
iqr = q3 - q1
whisker_width = 1.5
# Apply filter with respect to IQR, including optional whiskers
outlier_age_train = train[(train['Age'] < q1 - whisker_width*iqr) | (train['Age'] > q3 + whisker_width*iqr)]
print(outlier_age_train.shape) # (66, 8)
#print(outlier_age_train)

"""
    Survived Pclass     Sex    Age  SibSp  Parch      Fare Embarked
7          0      3    male   2.00      3      1   21.0750        S
11         1      1  female  58.00      0      0   26.5500        S
15         1      2  female  55.00      0      0   16.0000        S
16         0      3    male   2.00      4      1   29.1250        Q
33         0      2    male  66.00      0      0   10.5000        S
54         0      1    male  65.00      0      1   61.9792        C
78         1      2    male   0.83      0      2   29.0000        S
94         0      3    male  59.00      0      0    7.2500        S
96         0      1    male  71.00      0      0   34.6542        C
116        0      3    male  70.50      0      0    7.7500        Q
119        0      3  female   2.00      4      2   31.2750        S
152        0      3    male  55.50      0      0    8.0500        S
164        0      3    male   1.00      4      1   39.6875        S
170        0      1    male  61.00      0      0   33.5000        S
172        1      3  female   1.00      1      1   11.1333        S
174        0      1    male  56.00      0      0   30.6958        C
183        1      2    male   1.00      2      1   39.0000        S
195        1      1  female  58.00      0      0  146.5208        C
205        0      3  female   2.00      0      1   10.4625        S
232        0      2    male  59.00      0      0   13.5000        S
252        0      1    male  62.00      0      0   26.5500        S
268        1      1  female  58.00      0      1  153.4625        S
275        1      1  female  63.00      1      0   77.9583        S
280        0      3    male  65.00      0      0    7.7500        Q
297        0      1  female   2.00      1      2  151.5500        S
305        1      1    male   0.92      1      2  151.5500        S
326        0      3    male  61.00      0      0    6.2375        S
340        1      2    male   2.00      1      1   26.0000        S
366        1      1  female  60.00      1      0   75.2500        C
381        1      3  female   1.00      0      2   15.7417        C
386        0      3    male   1.00      5      2   46.9000        S
438        0      1    male  64.00      1      4  263.0000        S
456        0      1    male  65.00      0      0   26.5500        S
467        0      1    male  56.00      0      0   26.5500        S
469        1      3  female   0.75      2      1   19.2583        C
479        1      3  female   2.00      0      1   12.2875        S
483        1      3  female  63.00      0      0    9.5875        S
487        0      1    male  58.00      0      0   29.7000        C
492        0      1    male  55.00      0      0   30.5000        S
493        0      1    male  71.00      0      0   49.5042        C
530        1      2  female   2.00      1      1   26.0000        S
545        0      1    male  64.00      0      0   26.0000        S
555        0      1    male  62.00      0      0   26.5500        S
570        1      2    male  62.00      0      0   10.5000        S
587        1      1    male  60.00      1      1   79.2000        C
625        0      1    male  61.00      0      0   32.3208        S
626        0      2    male  57.00      0      0   12.3500        Q
630        1      1    male  80.00      0      0   30.0000        S
642        0      3  female   2.00      3      2   27.9000        S
644        1      3  female   0.75      2      1   19.2583        C
647        1      1    male  56.00      0      0   35.5000        C
659        0      1    male  58.00      0      2  113.2750        C
672        0      2    male  70.00      0      0   10.5000        S
684        0      2    male  60.00      1      1   39.0000        S
694        0      1    male  60.00      0      0   26.5500        S
745        0      1    male  70.00      1      1   71.0000        S
755        1      2    male   0.67      1      1   14.5000        S
772        0      2  female  57.00      0      0   10.5000        S
788        1      3    male   1.00      1      2   20.5750        S
803        1      3    male   0.42      0      1    8.5167        C
824        0      3    male   2.00      4      1   39.6875        S
827        1      2    male   1.00      0      2   37.0042        C
829        1      1  female  62.00      0      0   80.0000        S
831        1      2    male   0.83      1      1   18.7500        S
851        0      3    male  74.00      0      0    7.7750        S
879        1      1  female  56.00      0      1   83.1583        C

"""

# boxplot with 1.5 whiskers
sns.boxplot(y='Age', data = train, whis=1.5)
plt.show()

# IQR method for outlier fare
# Calculate Q1, Q2 and IQR
q1 = train['Fare'].quantile(0.25)                 
q3 = train['Fare'].quantile(0.75)
iqr = q3 - q1
whisker_width = 1.5
lower_whisker = q1 - whisker_width*iqr
upper_whisker = q3 + whisker_width*iqr
# Apply filter with respect to IQR, including optional whiskers
outlier_fare_train = train[(train['Fare'] < q1 - whisker_width*iqr) | (train['Fare'] > q3 + whisker_width*iqr)]
outlier_fare_train.shape

# boxplot with 1.5 whiskers
sns.boxplot(y='Fare', data = train, whis=1.5)
plt.show()


# multivariate outlier - fare and class in comparison

# sometimes the outlier can also occur when you compare one column with another
# in our titanic example - we can check if the fares are directly proportional to the class
# Are the first class people paying high fare and third class people paying low fare? are there any overlap between these fares?

# below code take Pclass in X axis and Fare in Y axis to display the distribution of Fare by Pclass
sns.boxplot(x='Pclass', y='Fare', data = train, whis=1.5)
plt.show()

# Outlier treatment with the method (Top, Bottom/Zero coding)

# Top coding - ceiling the uppper limit of the column with the outer whisker value
train.loc[train.Fare>upper_whisker,'Fare'] = upper_whisker

# Bottom / Zero coding - ceiling the lower limit of the column with lower whisker or zero
# It is called Bottom coding when you ceil the lowest value to lower whisker
# It is called zero coding when you ceil the lowest value to zero
# Zero coding should be used for variables which cannot take neagtive values - example, Age cannot be negative

train.loc[train.Fare<0,'Fare'] = 0

# display of minimum and maximum after outlier treatment
print(train['Fare'].min()) # 4.0125
print(train['Fare'].max()) # 66.3


# Another method for outlier treatment is Binning
# Group the values into certain bins -> e.g Age 0 to 10 in a bin called '0 - 10', etc

# Equal width binning -> width = (max value — min value) / N
age_range = train.Age.max() - train.Age.min()
min_value = int(np.floor(train.Age.min()))
max_value = int(np.ceil(train.Age.max()))
 
# let's round the bin width
# N = number of bins (which is 10 in the below code)
# change the value 10 in the below code to see how the grouping differs
inter_value = int(np.round(age_range/10))
 
val = (min_value, max_value, inter_value)
print(val)