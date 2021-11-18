# Phase 2 Project


## Project Overview

This project analyzes house sales data in a northwestern county using regression model.


## Business Problem

After buiding the regression model, the features that are closely related to house price will be identified. 

Therefore, some suggestions could be given to both the buyers and sellers.

* For the buyer, they will know the price of the house based on the characteristics of the house, and also, what's the investment value for the house.

* For the seller, they may know whether they can do something to sell the house with a better price.


## Data

This project uses the King County House Sales dataset, which can be found in  `kc_house_data.csv` in the data folder in this repo. 
The description of the column names can be found in `column_names.md` in the same folder.




## Methods
### Importing necessary libraries
```python
# Warning off
import warnings
warnings.filterwarnings('ignore')

# import pandas and numpy
import pandas as pd
import numpy as np

# import data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline

# import linear regression related modules
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```


###  Loading data to check the potential features
```python
#loading kc_house_data.csv data
df = pd.read_csv('./data/kc_house_data.csv')
df.head() # checking the head for information
```

```python
# Describe the dataset using 5-point statistics
df.describe()
# What data is available to us?
df.info()
```
#We have potentially 19 predictors excluding the id and the target,i.e.,the price
#We have a total of 21597 rows, while some rows have null values in some predictors
#Several predictors' data type need to be changed

### Data Preparation
#### Deal with data types: sqft_basement & date
```python
#sqft_basement: Numerical Data Stored as Strings need to be reformat to float
print(df.sqft_basement.unique())
df.sqft_basement.value_counts()
#there is '?' in the sqft_basement, need to be replaced as nan before reformat to float
df.sqft_basement = df.sqft_basement.map(lambda x: float(x.replace('?', 'nan')))
df.sqft_basement.unique()
```

```python
# For the sold date, since day is not important for the regression model,
# I only extract year and month for the sold date, and add two columns as year_sold and month_sold
df['year_sold'] = pd.DatetimeIndex(df['date']).year
df['month_sold'] = pd.DatetimeIndex(df['date']).month

# Based on the yr_built and month_sold, I create another column as age_sold of the house
df['age_sold'] = df['year_sold'] - df['yr_built'] + 1
df.head()
```

#### Deal with null values
```python
# Get the percentage value of null data for each column
df.isnull().sum()*100/df.shape[0]
```

```python
# There are some null data in waterfront, view, yr_renovated, sqft_basement
# 1) since the percentage of null data in view is low, I just drop these rows
# 2) For waterfront and yr_renovated, the percentage of null data is high,I will assign another value there

# waterfront is a categorical variable
df.waterfront.value_counts()
# replace nan as a value: 
# Originally I used 2.0 as a third category, 
# but late I found the price for this missing data is similar as for waterfront == 0
# Therefore, I fill the null as 0
df.waterfront = df.waterfront.fillna(0)
df.waterfront.value_counts()

# yr_renovated has 17011/17755~96% without renovation, 
#  and only 4% with renovation based on the non-null data
df.yr_renovated.value_counts()
# take a look of histogram
fig, axs = plt.subplots(2,figsize=(12,8))
df['yr_renovated'].hist(ax = axs[0]);
axs[0].set_title('All non-null data')
axs[0].set_xlabel('Year')
# with renovation
df[df.yr_renovated > 0].yr_renovated.hist(ax = axs[1])
# dfwrenov['yr_renovated'].hist(ax = axs[1]);
axs[1].set_title('Renovation data')
axs[0].set_xlabel('Year')
# Based on renovated data, I create a caterogrial variable as is_renovated

ds_renovated = df['yr_renovated']
ds_renovated[ds_renovated >0] = 1
# replace nan as a value: 
# Originally I used 2.0 as a third category, 
# but late I found the price for this missing data is similar as for is_renovated == 0
# Therefore, I fill the null as 0
ds_renovated = ds_renovated.fillna(0)
ds_renovated
df['is_renovated'] = ds_renovated
del ds_renovated
df.is_renovated.value_counts()
# assign as -1 to make sure these rows are not dropped in the following operation
df.yr_renovated = df.yr_renovated.fillna(-1)
# for view and sqft_basement, I just drop those rows with null value, since they are only a few
df.dropna(inplace=True)
print(df.info())
print(df.shape)
df.is_renovated.value_counts()
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 21082 entries, 0 to 21596
Data columns (total 25 columns):
    Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   id             21082 non-null  int64  
 1   date           21082 non-null  object 
 2   price          21082 non-null  float64
 3   bedrooms       21082 non-null  int64  
 4   bathrooms      21082 non-null  float64
 5   sqft_living    21082 non-null  int64  
 6   sqft_lot       21082 non-null  int64  
 7   floors         21082 non-null  float64
 8   waterfront     21082 non-null  float64
 9   view           21082 non-null  float64
 10  condition      21082 non-null  int64  
 11  grade          21082 non-null  int64  
 12  sqft_above     21082 non-null  int64  
 13  sqft_basement  21082 non-null  float64
 14  yr_built       21082 non-null  int64  
 15  yr_renovated   21082 non-null  float64
 16  zipcode        21082 non-null  int64  
 17  lat            21082 non-null  float64
 18  long           21082 non-null  float64
 19  sqft_living15  21082 non-null  int64  
 20  sqft_lot15     21082 non-null  int64  
 21  year_sold      21082 non-null  int64  
 22  month_sold     21082 non-null  int64  
 23  age_sold       21082 non-null  int64  
 24  is_renovated   21082 non-null  float64
dtypes: float64(10), int64(14), object(1)
memory usage: 4.2+ MB
None
(21082, 25)

0.0    20360
1.0      722
Name: is_renovated, dtype: int64

#### Deal with outliers if existed in some columns
```python
# For some selected columns, have a boxplot to examine the outliers
x_cols = ['price','bedrooms','bathrooms','sqft_living','sqft_lot',
          'floors','sqft_above','sqft_basement','sqft_living15','sqft_lot15']
fig, axs = plt.subplots(2,5, figsize = (15,6))
for colii in range(len(x_cols)):
    sns.boxplot(df[x_cols[colii]],ax = axs[colii//5, colii%5])
plt.tight_layout()
```
![figure of outliers](Figs/outliers.png)

```python
# process of outliers:
# It seems all these items except floors have outliers
# I will drop the outliers as there are sufficient data

x_cols = ['price','bedrooms','bathrooms','sqft_living','sqft_lot',
          'sqft_above','sqft_basement','sqft_living15','sqft_lot15']

for colname in x_cols:
    Q1 = df[colname].quantile(0.25)
    Q3 = df[colname].quantile(0.75)
    IQR = Q3 - Q1
    lenori = len(df[colname])
    df = df[(df[colname] >= Q1 - 1.5*IQR) & (df[colname] <= Q3 + 1.5*IQR)]
    lennew = len(df[colname])
    print(f'Number of rows based on {colname} : {lenori} -> {lennew}')
```
Number of rows based on price : 21082 -> 19951
Number of rows based on bedrooms : 19951 -> 19502
Number of rows based on bathrooms : 19502 -> 19437
Number of rows based on sqft_living : 19437 -> 19181
Number of rows based on sqft_lot : 19181 -> 17163
Number of rows based on sqft_above : 17163 -> 16697
Number of rows based on sqft_basement : 16697 -> 16401
Number of rows based on sqft_living15 : 16401 -> 16195
Number of rows based on sqft_lot15 : 16195 -> 15831

```python
# boxplot for the remaining data
fig, axs = plt.subplots(2,5, figsize = (15,6))
for colii in range(len(x_cols)):
    sns.boxplot(df[x_cols[colii]],ax = axs[colii//5, colii%5])
plt.tight_layout()
# now looks all good
```
![figure of outliers](Figs/outliers_refined.png)

```python
# visualization of the final data
# with its histogram:
df.hist(figsize = (20,18));
print(df.waterfront.value_counts())
print(df.condition.value_counts())
print(df.is_renovated.value_counts())
# looks good
df.bathrooms.value_counts()
```
![figure of outliers](Figs/histogram_allfeatures.png)

#### Deal with zipcode
```python
# df.zipcode.value_counts()
# I only keep the first four digits since if I only keep the first three digits,it will only have two zipcodes
df['zipcode4'] = df.zipcode//10 * 10
df.zipcode4.value_counts()
```
98110    2051
98050    1831
98030    1767
98100    1594
98000    1488
98020    1243
98120     931
98130     689
98040     629
98140     588
98070     524
98190     465
98170     407
98160     369
98150     366
98010     337
98090     229
98060     209
98180     114
Name: zipcode4, dtype: int64

```python
# visualization of categorical variables 
plt.figure(figsize=(20, 12))
plt.subplot(3,3,1)
sns.boxplot(x = 'waterfront', y = 'price', data = df)
plt.subplot(3,3,2)
sns.boxplot(x = 'is_renovated', y = 'price', data = df)
plt.subplot(3,3,3)
sns.boxplot(x = 'bedrooms', y = 'price', data = df)
plt.subplot(3,3,4)
sns.boxplot(x = 'condition', y = 'price', data = df)
plt.subplot(3,3,5)
sns.boxplot(x = 'floors', y = 'price', data = df)
plt.subplot(3,3,6)
sns.boxplot(x = 'view', y = 'price', data = df)
plt.subplot(3,1,3)
sns.boxplot(x = 'zipcode4', y = 'price', data = df)
plt.show()
```
![figure of categorial variables](Figs/boxplot_categorialvar.png)

```python
# For the zip codes, I implemented the one hot encoding .get_dummies() method
zipcode4_dums = pd.get_dummies(df['zipcode4'])
df = pd.concat([df,zipcode4_dums],axis = 1)
df.head()
```

```python
# drop one zipcode to redact redundant information, I will drop 98110, since it has the most numeber of 1
df.drop([98110], axis = 1, inplace = True)
df.head()
```

### Modeling
After finishing the data preparation, now I start to build the regression model

```python
# drop id, date, yr_renovated,lat, long, zipcode from df
# the id has not related to the house price
# the date has been transformed into sold_year and sold_month
# the yr_renovated has been transformed into is_renovated
# the lat and long indicate similar information as zipcode
# zipcode has been transformed into zipcode4 and dummy variables
# I keep zipcode4 try to take a look which one is better based on either zipcode4 or zipcode_dummies
drop_vars = ['id','date','yr_renovated','lat','long','zipcode']
df.drop(drop_vars, axis = 1, inplace = True)
df.head()
```
```python
# transform column names as string
df.columns = df.columns.astype(str)
df.head()
```

```python
# Rescaling the features except the 'dummy' variable
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = list(['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'is_renovated',
       'sqft_living15', 'sqft_lot15', 'year_sold','age_sold',
       'month_sold'])
df_scl = df
df_scl[num_vars] = scaler.fit_transform(df[num_vars])
df_scl.head()
```

```python
#Check the distribution of the target: price, to see whether it follows a normal distribution 
sns.distplot(df_scl['price'] , fit=stats.norm);

# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(df_scl['price'])
print( '\n mean = {:.2f} and std dev = {:.2f}\n'.format(mu, sigma))

#NPlotting the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Distribution of Price')

#Also the QQ plot
fig = plt.figure()
res = stats.probplot(df_scl['price'], plot=plt)
plt.show()
```
mean = 446934.39 and std dev = 188664.37

![figure of target](Figs/target_ori.png)
![figure of target](Figs/target_ori_prob.png)

```python
# it looks the target is skewed right, therefore, I used a log transformation to make it more normal
#Using the log1p function applies log(1+x) to all elements of the column
df_scl['price_log1p'] = np.log1p(df_scl['price'])

#Check the new distribution after log transformation 
sns.distplot(df_scl['price_log1p'] , fit=stats.norm);

# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(df_scl['price_log1p'])
print( '\n mean = {:.2f} and std dev = {:.2f}\n'.format(mu, sigma))

#NPlotting the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Distribution of Log(1+price)')

#Also the QQ plot
fig = plt.figure()
res = stats.probplot(df_scl['price_log1p'], plot=plt)
plt.show()
```
mean = 446934.39 and std dev = 188664.37

![figure of target](Figs/target_log1p.png)
![figure of target](Figs/target_log1p+prob.png)

```python
# Splitting the Data into Training and Testing
df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 100)
print(len(df_train), len(df_test))
df_train.head()
df.columns
```
12664 3167
Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'sqft_living15', 'sqft_lot15', 'year_sold',
       'month_sold', 'age_sold', 'is_renovated', 'zipcode4', '98000', '98010',
       '98020', '98030', '98040', '98050', '98060', '98070', '98090', '98100',
       '98120', '98130', '98140', '98150', '98160', '98170', '98180', '98190',
       'price_log1p'],
      dtype='object')

Checking for Multicollinearity
```python
# Check the correlation coefficients to see which variables are highly correlated
num_vars.append('price')
num_vars.append('price_log1p')
corr = df_train[num_vars].corr()
corr
# Using heatmap to visualzation of the correlation coefficients
sns.set(rc = {'figure.figsize':(18,10)})
sns.heatmap(corr, center=0, annot=True);
```
![figure of target](Figs/cc_featurestargets.png)
Price seems to be correlated to sqrt_living and grade the most. 
And some sqft-related and rooms-related variables have high correlation, e.g., sqft_living vs.bedrooms and bathrooms

```
# Fitting the actual model with all available features, zipcode with dummy values
import statsmodels.api as sm

# price vs price_log1p
outcome1 = 'price'
outcome2 = 'price_log1p'
x_cols = list(df_train.columns)
x_cols.remove(outcome1)
x_cols.remove(outcome2)
x_cols.remove('zipcode4')
model1 = sm.OLS(df_train[outcome1],sm.add_constant(df_train[x_cols])).fit()
print(model1.summary())
model2 = sm.OLS(df_train[outcome2],sm.add_constant(df_train[x_cols])).fit()
print(model2.summary())

# zipcode_dummy vs zipcode4
outcome1 = 'price'
outcome2 = 'price_log1p'
# x_cols = list(df.columns)
# x_cols.remove(outcome1)
# x_cols.remove(outcome2)
# # remove zipcdoe_dummies
# for colname in x_cols:
#     if colname.startswith('98') == 1:
#         x_cols.remove(colname)
# print(x_cols)
x_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
          'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
          'sqft_basement', 'yr_built', 'sqft_living15', 'sqft_lot15',
          'year_sold', 'month_sold', 'age_sold', 'is_renovated', 'zipcode4']
model3 = sm.OLS(df_train[outcome1],sm.add_constant(df_train[x_cols])).fit()
print(model3.summary())
model4 = sm.OLS(df_train[outcome2],sm.add_constant(df_train[x_cols])).fit()
print(model4.summary())
```

OLS Regression Results                            

Dep. Variable:                  price   R-squared:                       0.591
Model:                            OLS   Adj. R-squared:                  0.590
Method:                 Least Squares   F-statistic:                     537.5
Date:                Wed, 10 Nov 2021   Prob (F-statistic):               0.00
Time:                        22:30:17   Log-Likelihood:            -1.6610e+05
No. Observations:               12664   AIC:                         3.323e+05
Df Residuals:                   12629   BIC:                         3.325e+05
Df Model:                          34                                         
Covariance Type:            nonrobust                                         

                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const          8918.9424   5829.693      1.530      0.126   -2508.141    2.03e+04
bedrooms      -3.627e+04   5446.755     -6.658      0.000   -4.69e+04   -2.56e+04
bathrooms      6.207e+04   9784.511      6.344      0.000    4.29e+04    8.12e+04
sqft_living    1.229e+05   5404.290     22.743      0.000    1.12e+05    1.34e+05
sqft_lot       -3.88e+04   1.23e+04     -3.158      0.002   -6.29e+04   -1.47e+04
floors         2.008e+04   8240.182      2.437      0.015    3925.964    3.62e+04
waterfront     2.348e+05   3.91e+04      6.000      0.000    1.58e+05    3.12e+05
view           8.478e+04   8627.140      9.827      0.000    6.79e+04    1.02e+05
condition      1.101e+05   7424.973     14.833      0.000    9.56e+04    1.25e+05
grade          6.001e+05   1.32e+04     45.418      0.000    5.74e+05    6.26e+05
sqft_above      1.28e+05   6481.235     19.745      0.000    1.15e+05    1.41e+05
sqft_basement  5.719e+04   4595.457     12.445      0.000    4.82e+04    6.62e+04
yr_built      -9.813e+04   4773.521    -20.557      0.000   -1.07e+05   -8.88e+04
sqft_living15  1.742e+05   8772.987     19.860      0.000    1.57e+05    1.91e+05
sqft_lot15    -2.931e+04   1.18e+04     -2.490      0.013   -5.24e+04   -6239.494
year_sold      1.848e+04   3706.867      4.985      0.000    1.12e+04    2.57e+04
month_sold      911.8656   6079.895      0.150      0.881    -1.1e+04    1.28e+04
age_sold       1.063e+05   4313.402     24.640      0.000    9.78e+04    1.15e+05
is_renovated   3.711e+04   6683.349      5.553      0.000     2.4e+04    5.02e+04
98000         -8.828e+04   5341.045    -16.528      0.000   -9.87e+04   -7.78e+04
98010         -1.187e+05   8633.648    -13.753      0.000   -1.36e+05   -1.02e+05
98020         -1.451e+05   5477.183    -26.496      0.000   -1.56e+05   -1.34e+05
98030          -1.23e+05   5176.919    -23.757      0.000   -1.33e+05   -1.13e+05
98040         -1.343e+05   6923.576    -19.395      0.000   -1.48e+05   -1.21e+05
98050         -1.128e+05   5160.955    -21.850      0.000   -1.23e+05   -1.03e+05
98060         -1.049e+05    1.1e+04     -9.576      0.000   -1.26e+05   -8.34e+04
98070             -7e+04   7290.713     -9.601      0.000   -8.43e+04   -5.57e+04
98090         -2.665e+05   9807.083    -27.176      0.000   -2.86e+05   -2.47e+05
98100         -1.986e+04   4530.968     -4.383      0.000   -2.87e+04    -1.1e+04
98120         -5.206e+04   5354.947     -9.722      0.000   -6.26e+04   -4.16e+04
98130         -7.896e+04   6086.060    -12.974      0.000   -9.09e+04    -6.7e+04
98140         -9.468e+04   6427.806    -14.730      0.000   -1.07e+05   -8.21e+04
98150         -9.598e+04   7976.368    -12.033      0.000   -1.12e+05   -8.03e+04
98160         -1.583e+05   8056.014    -19.654      0.000   -1.74e+05   -1.43e+05
98170         -1.406e+05   7516.628    -18.710      0.000   -1.55e+05   -1.26e+05
98180         -1.907e+05   1.33e+04    -14.294      0.000   -2.17e+05   -1.65e+05
98190         -7.533e+04   6979.028    -10.794      0.000    -8.9e+04   -6.16e+04

Omnibus:                      803.165   Durbin-Watson:                   2.020
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1303.535
Skew:                           0.509   Prob(JB):                    8.73e-284
Kurtosis:                       4.197   Cond. No.                     1.18e+16


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 3.53e-28. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
                            OLS Regression Results                            

Dep. Variable:            price_log1p   R-squared:                       0.592
Model:                            OLS   Adj. R-squared:                  0.591
Method:                 Least Squares   F-statistic:                     538.9
Date:                Wed, 10 Nov 2021   Prob (F-statistic):               0.00
Time:                        22:30:17   Log-Likelihood:                -1419.1
No. Observations:               12664   AIC:                             2908.
Df Residuals:                   12629   BIC:                             3169.
Df Model:                          34                                         

                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             7.9947      0.013    608.916      0.000       7.969       8.020
bedrooms         -0.0617      0.012     -5.027      0.000      -0.086      -0.038
bathrooms         0.1840      0.022      8.351      0.000       0.141       0.227
sqft_living       0.1822      0.012     14.973      0.000       0.158       0.206
sqft_lot         -0.0729      0.028     -2.634      0.008      -0.127      -0.019
floors            0.0356      0.019      1.917      0.055      -0.001       0.072
waterfront        0.5028      0.088      5.704      0.000       0.330       0.676
view              0.1296      0.019      6.672      0.000       0.092       0.168
condition         0.2506      0.017     14.988      0.000       0.218       0.283
grade             1.3042      0.030     43.825      0.000       1.246       1.363
sqft_above        0.2768      0.015     18.962      0.000       0.248       0.305
sqft_basement     0.1498      0.010     14.471      0.000       0.129       0.170
yr_built          3.7655      0.011    350.256      0.000       3.744       3.787
sqft_living15     0.4580      0.020     23.179      0.000       0.419       0.497
sqft_lot15       -0.0914      0.027     -3.450      0.001      -0.143      -0.039
year_sold         0.0073      0.008      0.871      0.384      -0.009       0.024
month_sold        0.0079      0.014      0.573      0.566      -0.019       0.035
age_sold          4.1928      0.010    431.602      0.000       4.174       4.212
is_renovated      0.0674      0.015      4.476      0.000       0.038       0.097
98000            -0.2521      0.012    -20.954      0.000      -0.276      -0.228
98010            -0.2426      0.019    -12.474      0.000      -0.281      -0.204
98020            -0.3534      0.012    -28.645      0.000      -0.378      -0.329
98030            -0.2914      0.012    -24.992      0.000      -0.314      -0.269
98040            -0.3606      0.016    -23.125      0.000      -0.391      -0.330
98050            -0.2545      0.012    -21.899      0.000      -0.277      -0.232
98060            -0.2034      0.025     -8.244      0.000      -0.252      -0.155
98070            -0.1462      0.016     -8.902      0.000      -0.178      -0.114
98090            -0.6454      0.022    -29.222      0.000      -0.689      -0.602
98100            -0.0452      0.010     -4.430      0.000      -0.065      -0.025
98120            -0.0839      0.012     -6.956      0.000      -0.108      -0.060
98130            -0.1440      0.014    -10.504      0.000      -0.171      -0.117
98140            -0.2258      0.014    -15.599      0.000      -0.254      -0.197
98150            -0.1817      0.018    -10.117      0.000      -0.217      -0.147
98160            -0.4296      0.018    -23.676      0.000      -0.465      -0.394
98170            -0.3339      0.017    -19.725      0.000      -0.367      -0.301
98180            -0.4945      0.030    -16.455      0.000      -0.553      -0.436
98190            -0.2300      0.016    -14.632      0.000      -0.261      -0.199

Omnibus:                       74.093   Durbin-Watson:                   2.024
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               98.507
Skew:                          -0.083   Prob(JB):                     4.07e-22
Kurtosis:                       3.399   Cond. No.                     1.18e+16


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 3.53e-28. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
                            OLS Regression Results                            

Dep. Variable:                  price   R-squared:                       0.541
Model:                            OLS   Adj. R-squared:                  0.540
Method:                 Least Squares   F-statistic:                     876.6
Date:                Wed, 10 Nov 2021   Prob (F-statistic):               0.00
Time:                        22:30:17   Log-Likelihood:            -1.6683e+05
No. Observations:               12664   AIC:                         3.337e+05
Df Residuals:                   12646   BIC:                         3.338e+05
Df Model:                          17                                         
Covariance Type:            nonrobust                                         

                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const         -3.021e+06   1.62e+06     -1.859      0.063   -6.21e+06    1.64e+05
bedrooms      -4.499e+04   5746.480     -7.829      0.000   -5.63e+04   -3.37e+04
bathrooms      6.795e+04   1.03e+04      6.570      0.000    4.77e+04    8.82e+04
sqft_living    1.366e+05   1.21e+04     11.300      0.000    1.13e+05     1.6e+05
sqft_lot       -6.64e+04    1.3e+04     -5.121      0.000   -9.18e+04    -4.1e+04
floors         5.415e+04   8562.993      6.323      0.000    3.74e+04    7.09e+04
waterfront     2.128e+05   4.13e+04      5.152      0.000    1.32e+05    2.94e+05
view           8.995e+04   9096.207      9.889      0.000    7.21e+04    1.08e+05
condition      9.337e+04   7781.321     11.999      0.000    7.81e+04    1.09e+05
grade          6.898e+05   1.35e+04     51.145      0.000    6.63e+05    7.16e+05
sqft_above     9.228e+04   1.07e+04      8.656      0.000    7.14e+04    1.13e+05
sqft_basement  7.518e+04   6694.617     11.229      0.000    6.21e+04    8.83e+04
yr_built      -1.648e+06   8.04e+05     -2.048      0.041   -3.22e+06    -7.1e+04
sqft_living15  1.492e+05   9157.177     16.290      0.000    1.31e+05    1.67e+05
sqft_lot15    -9.637e+04    1.2e+04     -8.014      0.000    -1.2e+05   -7.28e+04
year_sold      2.957e+04   8094.952      3.653      0.000    1.37e+04    4.54e+04
month_sold      752.5777   6434.145      0.117      0.907   -1.19e+04    1.34e+04
age_sold      -1.361e+06   8.13e+05     -1.674      0.094   -2.95e+06    2.33e+05
is_renovated   2.327e+04   7054.714      3.299      0.001    9442.772    3.71e+04
zipcode4         45.6118     24.760      1.842      0.065      -2.922      94.146

Omnibus:                      712.654   Durbin-Watson:                   2.022
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1081.332
Skew:                           0.486   Prob(JB):                    1.55e-235
Kurtosis:                       4.052   Cond. No.                     1.53e+21


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 5.2e-29. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
                            OLS Regression Results                            

Dep. Variable:            price_log1p   R-squared:                       0.528
Model:                            OLS   Adj. R-squared:                  0.528
Method:                 Least Squares   F-statistic:                     832.8
Date:                Wed, 10 Nov 2021   Prob (F-statistic):               0.00
Time:                        22:30:17   Log-Likelihood:                -2338.8
No. Observations:               12664   AIC:                             4714.
Df Residuals:                   12646   BIC:                             4848.
Df Model:                          17                                         
Covariance Type:            nonrobust                                         

                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const            -4.7111      3.712     -1.269      0.204     -11.988       2.566
bedrooms         -0.0872      0.013     -6.643      0.000      -0.113      -0.061
bathrooms         0.1967      0.024      8.322      0.000       0.150       0.243
sqft_living       0.2555      0.028      9.247      0.000       0.201       0.310
sqft_lot         -0.1348      0.030     -4.552      0.000      -0.193      -0.077
floors            0.1216      0.020      6.215      0.000       0.083       0.160
waterfront        0.4419      0.094      4.683      0.000       0.257       0.627
view              0.1401      0.021      6.743      0.000       0.099       0.181
condition         0.2039      0.018     11.466      0.000       0.169       0.239
grade             1.5123      0.031     49.074      0.000       1.452       1.573
sqft_above        0.1681      0.024      6.900      0.000       0.120       0.216
sqft_basement     0.1796      0.015     11.742      0.000       0.150       0.210
yr_built         -2.6464      1.838     -1.440      0.150      -6.249       0.956
sqft_living15     0.3993      0.021     19.084      0.000       0.358       0.440
sqft_lot15       -0.2680      0.027     -9.752      0.000      -0.322      -0.214
year_sold         0.0573      0.018      3.100      0.002       0.021       0.094
month_sold        0.0076      0.015      0.520      0.603      -0.021       0.036
age_sold         -2.0464      1.858     -1.101      0.271      -5.688       1.596
is_renovated      0.0343      0.016      2.126      0.034       0.003       0.066
zipcode4          0.0002   5.66e-05      3.402      0.001    8.16e-05       0.000

Omnibus:                       74.474   Durbin-Watson:                   2.022
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               76.983
Skew:                          -0.173   Prob(JB):                     1.92e-17
Kurtosis:                       3.160   Cond. No.                     1.53e+21


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 5.2e-29. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
```
# Based on the R-squared values from four models, 
# I choose to use outcome = 'price_log1p' and zipcode_dummies for regression model
# price_log1p
outcome = 'price_log1p'
x_cols = list(df_train.columns)
x_cols.remove(outcome)
x_cols.remove('price')
x_cols.remove('zipcode4')
model0 = sm.OLS(df_train[outcome],sm.add_constant(df_train[x_cols])).fit()
print(model0.summary()) 
```
 OLS Regression Results                            

Dep. Variable:            price_log1p   R-squared:                       0.592
Model:                            OLS   Adj. R-squared:                  0.591
Method:                 Least Squares   F-statistic:                     538.9
Date:                Wed, 10 Nov 2021   Prob (F-statistic):               0.00
Time:                        22:32:51   Log-Likelihood:                -1419.1
No. Observations:               12664   AIC:                             2908.
Df Residuals:                   12629   BIC:                             3169.
Df Model:                          34                                         
Covariance Type:            nonrobust                                         

                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             7.9947      0.013    608.916      0.000       7.969       8.020
bedrooms         -0.0617      0.012     -5.027      0.000      -0.086      -0.038
bathrooms         0.1840      0.022      8.351      0.000       0.141       0.227
sqft_living       0.1822      0.012     14.973      0.000       0.158       0.206
sqft_lot         -0.0729      0.028     -2.634      0.008      -0.127      -0.019
floors            0.0356      0.019      1.917      0.055      -0.001       0.072
waterfront        0.5028      0.088      5.704      0.000       0.330       0.676
view              0.1296      0.019      6.672      0.000       0.092       0.168
condition         0.2506      0.017     14.988      0.000       0.218       0.283
grade             1.3042      0.030     43.825      0.000       1.246       1.363
sqft_above        0.2768      0.015     18.962      0.000       0.248       0.305
sqft_basement     0.1498      0.010     14.471      0.000       0.129       0.170
yr_built          3.7655      0.011    350.256      0.000       3.744       3.787
sqft_living15     0.4580      0.020     23.179      0.000       0.419       0.497
sqft_lot15       -0.0914      0.027     -3.450      0.001      -0.143      -0.039
year_sold         0.0073      0.008      0.871      0.384      -0.009       0.024
month_sold        0.0079      0.014      0.573      0.566      -0.019       0.035
age_sold          4.1928      0.010    431.602      0.000       4.174       4.212
is_renovated      0.0674      0.015      4.476      0.000       0.038       0.097
98000            -0.2521      0.012    -20.954      0.000      -0.276      -0.228
98010            -0.2426      0.019    -12.474      0.000      -0.281      -0.204
98020            -0.3534      0.012    -28.645      0.000      -0.378      -0.329
98030            -0.2914      0.012    -24.992      0.000      -0.314      -0.269
98040            -0.3606      0.016    -23.125      0.000      -0.391      -0.330
98050            -0.2545      0.012    -21.899      0.000      -0.277      -0.232
98060            -0.2034      0.025     -8.244      0.000      -0.252      -0.155
98070            -0.1462      0.016     -8.902      0.000      -0.178      -0.114
98090            -0.6454      0.022    -29.222      0.000      -0.689      -0.602
98100            -0.0452      0.010     -4.430      0.000      -0.065      -0.025
98120            -0.0839      0.012     -6.956      0.000      -0.108      -0.060
98130            -0.1440      0.014    -10.504      0.000      -0.171      -0.117
98140            -0.2258      0.014    -15.599      0.000      -0.254      -0.197
98150            -0.1817      0.018    -10.117      0.000      -0.217      -0.147
98160            -0.4296      0.018    -23.676      0.000      -0.465      -0.394
98170            -0.3339      0.017    -19.725      0.000      -0.367      -0.301
98180            -0.4945      0.030    -16.455      0.000      -0.553      -0.436
98190            -0.2300      0.016    -14.632      0.000      -0.261      -0.199

Omnibus:                       74.093   Durbin-Watson:                   2.024
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               98.507
Skew:                          -0.083   Prob(JB):                     4.07e-22
Kurtosis:                       3.399   Cond. No.                     1.18e+16


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 3.53e-28. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.

``` python
# Remove the insignificant Features and rerun the model
summary = model0.summary()
p_table = summary.tables[1]
p_table = pd.DataFrame(p_table.data)
p_table.columns = p_table.iloc[0]
p_table = p_table.drop(0)
p_table = p_table.set_index(p_table.columns[0])
p_table['P>|t|'] = p_table['P>|t|'].astype(float)
x_cols = list(p_table[p_table['P>|t|'] < 0.05].index)
x_cols.remove('const')
print(len(p_table), len(x_cols))

p_table.head()
x_cols
```
37 33
['bedrooms',
 'bathrooms',
 'sqft_living',
 'sqft_lot',
 'waterfront',
 'view',
 'condition',
 'grade',
 'sqft_above',
 'sqft_basement',
 'yr_built',
 'sqft_living15',
 'sqft_lot15',
 'age_sold',
 'is_renovated',
 '98000',
 '98010',
 '98020',
 '98030',
 '98040',
 '98050',
 '98060',
 '98070',
 '98090',
 '98100',
 '98120',
 '98130',
 '98140',
 '98150',
 '98160',
 '98170',
 '98180',
 '98190']

``` python
model1 = sm.OLS(df_train[outcome],sm.add_constant(df_train[x_cols])).fit()
model1.summary()
```
OLS Regression Results
Dep. Variable:	price_log1p	R-squared:	0.592
Model:	OLS	Adj. R-squared:	0.591
Method:	Least Squares	F-statistic:	572.4
Date:	Wed, 10 Nov 2021	Prob (F-statistic):	0.00
Time:	22:35:24	Log-Likelihood:	-1421.1
No. Observations:	12664	AIC:	2908.
Df Residuals:	12631	BIC:	3154.
Df Model:	32		
Covariance Type:	nonrobust		
coef	std err	t	P>|t|	[0.025	0.975]
const	7.6065	0.595	12.782	0.000	6.440	8.773
bedrooms	-0.0618	0.012	-5.040	0.000	-0.086	-0.038
bathrooms	0.1932	0.022	8.987	0.000	0.151	0.235
sqft_living	0.1890	0.013	14.937	0.000	0.164	0.214
sqft_lot	-0.0790	0.027	-2.873	0.004	-0.133	-0.025
waterfront	0.5071	0.088	5.755	0.000	0.334	0.680
view	0.1305	0.019	6.720	0.000	0.092	0.169
condition	0.2499	0.017	14.953	0.000	0.217	0.283
grade	1.3083	0.030	44.092	0.000	1.250	1.366
sqft_above	0.2838	0.014	20.090	0.000	0.256	0.311
sqft_basement	0.1409	0.010	14.689	0.000	0.122	0.160
yr_built	4.1670	0.593	7.031	0.000	3.005	5.329
sqft_living15	0.4555	0.020	23.098	0.000	0.417	0.494
sqft_lot15	-0.0985	0.026	-3.749	0.000	-0.150	-0.047
age_sold	4.5906	0.597	7.683	0.000	3.419	5.762
is_renovated	0.0686	0.015	4.563	0.000	0.039	0.098
98000	-0.2544	0.012	-21.264	0.000	-0.278	-0.231
98010	-0.2435	0.019	-12.526	0.000	-0.282	-0.205
98020	-0.3556	0.012	-28.952	0.000	-0.380	-0.332
98030	-0.2938	0.012	-25.349	0.000	-0.316	-0.271
98040	-0.3632	0.016	-23.386	0.000	-0.394	-0.333
98050	-0.2577	0.012	-22.386	0.000	-0.280	-0.235
98060	-0.2063	0.025	-8.374	0.000	-0.255	-0.158
98070	-0.1481	0.016	-9.039	0.000	-0.180	-0.116
98090	-0.6476	0.022	-29.368	0.000	-0.691	-0.604
98100	-0.0439	0.010	-4.313	0.000	-0.064	-0.024
98120	-0.0839	0.012	-6.960	0.000	-0.108	-0.060
98130	-0.1440	0.014	-10.506	0.000	-0.171	-0.117
98140	-0.2264	0.014	-15.638	0.000	-0.255	-0.198
98150	-0.1828	0.018	-10.183	0.000	-0.218	-0.148
98160	-0.4300	0.018	-23.700	0.000	-0.466	-0.394
98170	-0.3353	0.017	-19.826	0.000	-0.368	-0.302
98180	-0.4956	0.030	-16.493	0.000	-0.555	-0.437
98190	-0.2316	0.016	-14.755	0.000	-0.262	-0.201
Omnibus:	75.118	Durbin-Watson:	2.023
Prob(Omnibus):	0.000	Jarque-Bera (JB):	99.613
Skew:	-0.086	Prob(JB):	2.34e-22
Kurtosis:	3.399	Cond. No.	2.25e+15


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 8.65e-27. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.

```python
#Investigate the multicollinearity
X = df_train[x_cols]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(x_cols, vif))
```
[('bedrooms', 1.7051669898683603),
 ('bathrooms', 2.8211105995428625),
 ('sqft_living', 75655116.4799782),
 ('sqft_lot', 4.483001686353898),
 ('waterfront', 1.0559519647486637),
 ('view', 1.1433103281270622),
 ('condition', 1.2761073751068845),
 ('grade', 2.3904373958950544),
 ('sqft_above', 56563748.80295928),
 ('sqft_basement', 6893463.418751773),
 ('yr_built', 4096.808779982019),
 ('sqft_living15', 2.489894206656194),
 ('sqft_lot15', 4.841415540307321),
 ('age_sold', 4092.553260335055),
 ('is_renovated', 1.1139946455921124),
 ('98000', 2.129621291213978),
 ('98010', 1.3341294814620723),
 ('98020', 1.8527786749644335),
 ('98030', 2.2902252338654927),
 ('98040', 1.5948356012290426),
 ('98050', 2.330397065221319),
 ('98060', 1.1998058564341272),
 ('98070', 1.5528075836332702),
 ('98090', 1.2130635491953714),
 ('98100', 1.6287626356712774),
 ('98120', 1.3934950202309393),
 ('98130', 1.3290630486825934),
 ('98140', 1.2668086634739153),
 ('98150', 1.276311483898106),
 ('98160', 1.2950530249665344),
 ('98170', 1.2551762325620677),
 ('98180', 1.09816473696312),
 ('98190', 1.2358175350874316)]
 
```python
# remove the features with vif >=5
vif_scores = list(zip(x_cols, vif))
x_cols = [x for x,vif in vif_scores if vif < 5]
print(len(vif_scores), len(x_cols))
x_cols
```
33 28
['bedrooms',
 'bathrooms',
 'sqft_lot',
 'waterfront',
 'view',
 'condition',
 'grade',
 'sqft_living15',
 'sqft_lot15',
 'is_renovated',
 '98000',
 '98010',
 '98020',
 '98030',
 '98040',
 '98050',
 '98060',
 '98070',
 '98090',
 '98100',
 '98120',
 '98130',
 '98140',
 '98150',
 '98160',
 '98170',
 '98180',
 '98190']
 
```python
# Refit model with subset features
model2 = sm.OLS(df_train[outcome],sm.add_constant(df_train[x_cols])).fit()
model2.summary()
```
OLS Regression Results
Dep. Variable:	price_log1p	R-squared:	0.551
Model:	OLS	Adj. R-squared:	0.550
Method:	Least Squares	F-statistic:	553.9
Date:	Wed, 10 Nov 2021	Prob (F-statistic):	0.00
Time:	22:36:47	Log-Likelihood:	-2024.0
No. Observations:	12664	AIC:	4106.
Df Residuals:	12635	BIC:	4322.
Df Model:	28		
Covariance Type:	nonrobust		
coef	std err	t	P>|t|	[0.025	0.975]
const	11.9547	0.017	697.245	0.000	11.921	11.988
bedrooms	0.0799	0.012	6.835	0.000	0.057	0.103
bathrooms	0.1433	0.019	7.611	0.000	0.106	0.180
sqft_lot	0.0597	0.029	2.092	0.036	0.004	0.116
waterfront	0.4484	0.092	4.854	0.000	0.267	0.629
view	0.1815	0.020	8.961	0.000	0.142	0.221
condition	0.4160	0.016	25.574	0.000	0.384	0.448
grade	1.2627	0.028	44.727	0.000	1.207	1.318
sqft_living15	0.6769	0.019	36.115	0.000	0.640	0.714
sqft_lot15	-0.0874	0.027	-3.184	0.001	-0.141	-0.034
is_renovated	0.1932	0.015	12.795	0.000	0.164	0.223
98000	-0.3756	0.012	-31.898	0.000	-0.399	-0.353
98010	-0.3779	0.020	-19.248	0.000	-0.416	-0.339
98020	-0.4796	0.012	-39.731	0.000	-0.503	-0.456
98030	-0.4288	0.011	-38.662	0.000	-0.451	-0.407
98040	-0.5181	0.015	-34.126	0.000	-0.548	-0.488
98050	-0.3843	0.011	-34.989	0.000	-0.406	-0.363
98060	-0.3270	0.025	-13.071	0.000	-0.376	-0.278
98070	-0.2736	0.016	-16.599	0.000	-0.306	-0.241
98090	-0.7971	0.022	-35.624	0.000	-0.841	-0.753
98100	-0.0615	0.011	-5.769	0.000	-0.082	-0.041
98120	-0.1174	0.013	-9.323	0.000	-0.142	-0.093
98130	-0.2092	0.014	-14.753	0.000	-0.237	-0.181
98140	-0.2716	0.015	-18.028	0.000	-0.301	-0.242
98150	-0.2805	0.019	-15.159	0.000	-0.317	-0.244
98160	-0.5098	0.019	-27.215	0.000	-0.547	-0.473
98170	-0.4020	0.018	-22.872	0.000	-0.436	-0.368
98180	-0.5953	0.031	-19.028	0.000	-0.657	-0.534
98190	-0.2947	0.016	-18.069	0.000	-0.327	-0.263
Omnibus:	33.073	Durbin-Watson:	2.020
Prob(Omnibus):	0.000	Jarque-Bera (JB):	41.312
Skew:	-0.032	Prob(JB):	1.07e-09
Kurtosis:	3.272	Cond. No.	58.9


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

```python
# checking normality
fig = sm.graphics.qqplot(model2.resid, dist=stats.norm, line='45', fit=True)
```
![figure of target](Figs/model_normality.png)

```python
#Check Homoscedasticity Assumption
plt.figure(figsize = (8,6))
plt.scatter(model2.predict(sm.add_constant(df_train[x_cols])), model2.resid)
plt.plot(model2.predict(sm.add_constant(df_train[x_cols])), [0 for i in range(len(df_train[x_cols]))]
```
![figure of target](Figs/model_homoscedasticity.png)

model2 seems pretty good in terms of normality, Homoscedasticity, and R-squared values

### Evaluation
```python
from sklearn.linear_model import LinearRegression

final_model = LinearRegression()
# Fit the model on X_train_final and y_train
final_model.fit(df_train[x_cols], df_train[outcome])

# Score the model on X_test_final and y_test
# (use the built-in .score method)
print( "Test score:  ", final_model.score(df_test[x_cols], df_test[outcome]))
print( "Train score: ", final_model.score(df_train[x_cols], df_train[outcome]))
```
Test score:   0.5432593071160836
Train score:  0.55107388384023

```python
# use cross validation to evaluate the model
from sklearn.model_selection import cross_validate, ShuffleSplit
splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
baseline_scores = cross_validate(
    estimator = final_model,
    X= df[x_cols],
    y= df[outcome],
    return_train_score=True,
    cv=splitter
)
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())
```
```python
Train score:      0.5487979017274688
Validation score: 0.5509113316623484
```
Train and validation scores are similar

```python
# check mean squared error, root mean squared error, and Mean absolute error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import math
print("MSE:       ", mean_squared_error(df_test[outcome], final_model.predict(df_test[x_cols])))
print("RMSE:      ", math.sqrt(mean_squared_error(df_test[outcome], final_model.predict(df_test[x_cols]))))
print("MAE:       ", mean_absolute_error(df_test[outcome], final_model.predict(df_test[x_cols])))
print("R-Squared: ", r2_score(df_test[outcome], final_model.predict(df_test[x_cols])))            
```
MSE:        0.08275149712742216
RMSE:       0.28766559948562176
MAE:        0.22861402157360175
R-Squared:  0.5432593071160836

```python
# visualization of real and predicted values for each value
preds = final_model.predict(df_test[x_cols])
fig, axs = plt.subplots(1,2, figsize =(16,8))
perfect_line = np.arange(min(preds.min(),df_test[outcome].min())*0.99, max(preds.max(),df_test[outcome].max())*1.01)
axs[0].plot(perfect_line,perfect_line, linestyle="--", color="black", label="Perfect Fit")
axs[0].scatter(df_test[outcome], preds, alpha=0.5)
axs[0].set_xlabel("Actual Price")
axs[0].set_ylabel("Predicted Price")
axs[0].legend();
axs[0].set_xlim([min(preds.min(),df_test[outcome].min())*0.99, max(preds.max(),df_test[outcome].max())*1.01])
axs[0].set_ylim([min(preds.min(),df_test[outcome].min())*0.99, max(preds.max(),df_test[outcome].max())*1.01])
axs[1].scatter(df_test[outcome], np.divide((df_test[outcome] - preds),df_test[outcome]) * 100, alpha=0.5)
axs[1].set_xlabel("Actual Price")
axs[1].set_ylabel("(Predicted Price - actual price)/actual price * 100")
```
![figure of target](Figs/prediction.png)

From above values and plots, the fitted regression model can predict house price very well

## Summary
```python
# the beta coefficients for different predictors
print(pd.Series(final_model.coef_, index=x_cols, name="Coefficients"))
print()
print("Intercept:", final_model.intercept_)
```
bedrooms         0.079858
bathrooms        0.143295
sqft_lot         0.059662
waterfront       0.448369
view             0.181528
condition        0.416001
grade            1.262687
sqft_living15    0.676947
sqft_lot15      -0.087438
is_renovated     0.193224
98000           -0.375587
98010           -0.377925
98020           -0.479622
98030           -0.428799
98040           -0.518139
98050           -0.384339
98060           -0.326992
98070           -0.273552
98090           -0.797135
98100           -0.061472
98120           -0.117371
98130           -0.209158
98140           -0.271636
98150           -0.280506
98160           -0.509839
98170           -0.402032
98180           -0.595276
98190           -0.294679
Name: Coefficients, dtype: float64

Intercept: 11.954698075947055


### From coefficients described above, I observed:
1) The grade and sqft_living15 have the strongest relationship with the house price
2) It is interesting to see the sqft_lot15 has the negative relationship with the house price
### To address the business question:
1) For buyer, they will know the house price is higer for a house with high grade and sqrt_living15
2) For seller, if they want to sell their house with a higher price, they could add waterfront, improve the grade/condition.
