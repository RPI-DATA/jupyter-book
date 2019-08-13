---
interact_link: content/C:\Users\sjgar\Documents\GitHub\jupyter-book\content\notebooks/linearregression.ipynb
kernel_name: python3
has_widgets: false
title: 'Linear Regression'
prev_page:
  url: /notebooks/pytorch
  title: 'PyTorch'
next_page:
  url: /notebooks/pca
  title: 'Principal Component Analysis'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


<img src="https://raw.githubusercontent.com/RPI-DATA/website/master/static/images/rpilogo.png" alt="RPI LOGO" style="width:400px">

<h1 style="text-align:center">Linear Regression</h1>

<a href="https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/linearregression.ipynb" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"> </a>



This project will work on how to predict the prices of homes based on the properties of the house. I will determine which house affected the final sale price and how effectively we can predict the sale price.

This notebook uses the following pedagogical patterns:
* [**4.2** Shift-enter for the Win](https://jupyter4edu.github.io/jupyter-edu-book/catalogue.html#shift-enter-for-the-win)
* [**4.8** Top-down Sequence](https://jupyter4edu.github.io/jupyter-edu-book/catalogue.html#top-down-sequence)



## Learning Objectives
---
By the end of this notebook, the reader should be able to perform Linear Regression techniques in Python. This includes:

1. Importing and formatting data
2. Training the LinearRegression model from the `sklearn.linear_model` library
3. Work with qualitative and quantitative data, and effectively deal with instances of categorical data
4. Analyze and determine proper handling of redundant and/or inconsistent data features
5. Create a heatmap visual with `matplot.lib` library



## Read the Data
---
   
For this exercise we will use the Ames Housing dataset to delineate our training and testing data. The Ames dataset is a collection of different characteristics and observations on houses in Ames, Iowa that is commonly used to create and test algorithms for predicting the prices of homes in that area. It contains 82 columns, or features, of data that describe the residences, such as:
- _Lot Area_: Lot size in square feet
- _Overall Qual_: Rates the overall material and finish of the house
- _Overall Cond_: Rates the overall condition of the house
- _Year Built_: Original construction date
- _Low Qual Fin SF_: Low quality finished square feet (all floors)
- _Full Bath_: Full bathrooms above grade
- _Fireplaces_: Number of fireplaces

and so on.

We will import this dataset using the `pandas` library, which is an open-source data analytics tool for Python that allows the use of `dataframe` objects and clean file parsing. Libraries in general are simply collections of functions in Python that users can import instead of having to write out the code themselves. This is done with the following lines of code:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import pandas as pd   # The 'as' renames the library to whatever follows, in this case 'pd' to simplify the syntax
data = pd.read_csv("data/AmesHousing.txt", delimiter = '\t')

```
</div>

</div>



Here the `read_csv` function looks at the fie `AmesHousing` and turns it into a dataframe `data`, which we can now use in Python. The delimiter `\t` is simply telling the function to distinguish any characteristics separated by a tab as a new value.

Next we will split the data so that the first 1,460 entries will be our training set, which we will use to build our linear regression model, while the remaining entries will represent the test set, which we will use to test its accuracy.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train = data[0:1460]
test = data[1460:]
target = 'SalePrice'
print(train.info())

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 82 columns):
Order              1460 non-null int64
PID                1460 non-null int64
MS SubClass        1460 non-null int64
MS Zoning          1460 non-null object
Lot Frontage       1211 non-null float64
Lot Area           1460 non-null int64
Street             1460 non-null object
Alley              109 non-null object
Lot Shape          1460 non-null object
Land Contour       1460 non-null object
Utilities          1460 non-null object
Lot Config         1460 non-null object
Land Slope         1460 non-null object
Neighborhood       1460 non-null object
Condition 1        1460 non-null object
Condition 2        1460 non-null object
Bldg Type          1460 non-null object
House Style        1460 non-null object
Overall Qual       1460 non-null int64
Overall Cond       1460 non-null int64
Year Built         1460 non-null int64
Year Remod/Add     1460 non-null int64
Roof Style         1460 non-null object
Roof Matl          1460 non-null object
Exterior 1st       1460 non-null object
Exterior 2nd       1460 non-null object
Mas Vnr Type       1449 non-null object
Mas Vnr Area       1449 non-null float64
Exter Qual         1460 non-null object
Exter Cond         1460 non-null object
Foundation         1460 non-null object
Bsmt Qual          1420 non-null object
Bsmt Cond          1420 non-null object
Bsmt Exposure      1419 non-null object
BsmtFin Type 1     1420 non-null object
BsmtFin SF 1       1459 non-null float64
BsmtFin Type 2     1419 non-null object
BsmtFin SF 2       1459 non-null float64
Bsmt Unf SF        1459 non-null float64
Total Bsmt SF      1459 non-null float64
Heating            1460 non-null object
Heating QC         1460 non-null object
Central Air        1460 non-null object
Electrical         1460 non-null object
1st Flr SF         1460 non-null int64
2nd Flr SF         1460 non-null int64
Low Qual Fin SF    1460 non-null int64
Gr Liv Area        1460 non-null int64
Bsmt Full Bath     1459 non-null float64
Bsmt Half Bath     1459 non-null float64
Full Bath          1460 non-null int64
Half Bath          1460 non-null int64
Bedroom AbvGr      1460 non-null int64
Kitchen AbvGr      1460 non-null int64
Kitchen Qual       1460 non-null object
TotRms AbvGrd      1460 non-null int64
Functional         1460 non-null object
Fireplaces         1460 non-null int64
Fireplace Qu       743 non-null object
Garage Type        1386 non-null object
Garage Yr Blt      1385 non-null float64
Garage Finish      1385 non-null object
Garage Cars        1460 non-null float64
Garage Area        1460 non-null float64
Garage Qual        1385 non-null object
Garage Cond        1385 non-null object
Paved Drive        1460 non-null object
Wood Deck SF       1460 non-null int64
Open Porch SF      1460 non-null int64
Enclosed Porch     1460 non-null int64
3Ssn Porch         1460 non-null int64
Screen Porch       1460 non-null int64
Pool Area          1460 non-null int64
Pool QC            1 non-null object
Fence              297 non-null object
Misc Feature       60 non-null object
Misc Val           1460 non-null int64
Mo Sold            1460 non-null int64
Yr Sold            1460 non-null int64
Sale Type          1460 non-null object
Sale Condition     1460 non-null object
SalePrice          1460 non-null int64
dtypes: float64(11), int64(28), object(43)
memory usage: 935.4+ KB
None
```
</div>
</div>
</div>



## Model the Data
--- 
###  Simple Regression
In this case, we will use a **simple linear regression** to evaluate the relationship between two variables: living area ("Gr Liv Area") and price ("SalePrice"). Assuming a linear relationship, the equation for the two variables will of course look like this:

$$
y = mx + b
$$

where $x$ is the independent variable ("Gr Liv Area"), $y$ is the dependent ("SalePrice"), $m$ is the slope, and $b$ is the intercept. Because these variables are data and thus cannot be changed, a linear regression looks to control $m$ and $b$ instead. Essentially, a linear regression simply changes these two terms to fit multiple lines onto the data, reviews the error between the line and the data points, and then returns the line that generates the least amount of error.

In Python, the function that performs the regression is `linearRegression.fit()`, which can be found in the scikit learning library. Simply include the two variables as arguments and it will find the line that best fits for you. We can also use the `mean_squared_error` function to get the total variance of the linear function, which can be found in the `numpy` library.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import numpy as np
from sklearn.linear_model import LinearRegression   # You can use this method to directly call for a function in a library

lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
from sklearn.metrics import mean_squared_error
train_predictions = lr.predict(train[['Gr Liv Area']])
test_predictions = lr.predict(test[['Gr Liv Area']])

train_mse = mean_squared_error(train_predictions, train['SalePrice'])
test_mse = mean_squared_error(test_predictions, test['SalePrice'])

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(lr.coef_)
print(train_rmse)
print(test_rmse)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[116.86624683]
56034.362001412796
57088.25161263909
```
</div>
</div>
</div>



In this case, we use the `lr.coef_()` to get the coefficient of the linear function which is 116.87. More than that, the standard error for the train data is 56034 and test data is 57088. Now, let's make the result more visible by plotting.

The following is the linear regression line made from data in "train":



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import matplotlib.pyplot as plt
plt.scatter(train[['Gr Liv Area']], train[['SalePrice']],  color='black')
plt.xlabel('Gr Liv Area in Train', fontsize = '18')
plt.ylabel('train_predictions' ,fontsize = '18')
trainPlot =plt.plot(train[['Gr Liv Area']], train_predictions, color='blue', linewidth=3)
trainPlot

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[<matplotlib.lines.Line2D at 0x1a19271d68>]
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/linearregression_10_1.png)

</div>
</div>
</div>



Now let's juxtapose the model with the test dataset to see if it can accurately predict the value:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import matplotlib.pyplot as plt
plt.scatter(test[['Gr Liv Area']], test[['SalePrice']],  color='black')
plt.xlabel('Gr Liv Area in Test', fontsize = '18')
plt.ylabel('test_predictions' ,fontsize = '18')
testPlot = plt.plot(test[['Gr Liv Area']], test_predictions, color='blue', linewidth=3)
testPlot

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[<matplotlib.lines.Line2D at 0x1a192b3ac8>]
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/linearregression_12_1.png)

</div>
</div>
</div>



Since the test data looks like it concentrates around the linear regression model, we can conclude that the model can predict the "Sale Price".



### Multiple Regression
In the real world, a **multiple regression** is a more useful technique since we need to evaluate more than one correlation in most cases. Now, we will still predict the SalePrice, but with one more variable - Overall Condition (Overall Cond). In this case the model will be a **binary linear equation** in the form of:

$$
Y = a_0 + coef_{Cond} * (Overall Cond) + coef_{Area} * (Gr Liv Area)
$$

where $a_0$ stands for the intial value while both "Overall Cond" and "Gr Liv Area" is zero, $coef_{Cond}$ stands for the coefficient of Overall Cond, and $coef_{Area}$ stands for the coefficient of Gr Liv Area.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.metrics import mean_squared_error
cols = ['Overall Cond', 'Gr Liv Area']
lr.fit(train[cols], train['SalePrice'])
train_predictions = lr.predict(train[cols])
test_predictions = lr.predict(test[cols])

train_rmse_2 = np.sqrt(mean_squared_error(train_predictions, train['SalePrice']))
test_rmse_2 = np.sqrt(mean_squared_error(test_predictions, test['SalePrice']))

print(lr.coef_)
print(lr.intercept_)
print(train_rmse_2)
print(test_rmse_2)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[-409.56846611  116.73118339]
7858.691146390513
56032.398015258674
57066.90779448559
```
</div>
</div>
</div>



such that the linear model will be like: 
$$
Y = 7858.7 - 409.6 * (Overall Cond) + 116.7 * (Gr Liv Area)
$$

However, it's hard to make a geometric explanation since the model will be either surface or high-dimension which can't be plotted. 



## Handling Data Types with Missing/Non-numeric Values
---
In the machine learning workflow, once we've selected the model we want to use, selecting the appropriate features for that model is the next important step. In the following code snippets, I will explore how to use correlation between features and the target column, correlation between features, and variance of features to select features.

I will specifically focus on selecting from feature columns that don't have any missing values or don't need to be transformed to be useful (e.g. columns like Year Built and Year Remod/Add). 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
numerical_train = train.select_dtypes(include=['int64', 'float'])
numerical_train = numerical_train.drop(['PID', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Mo Sold', 'Yr Sold'], axis=1)
null_series = numerical_train.isnull().sum()
full_cols_series = null_series[null_series == 0]
print(full_cols_series)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Order              0
MS SubClass        0
Lot Area           0
Overall Qual       0
Overall Cond       0
1st Flr SF         0
2nd Flr SF         0
Low Qual Fin SF    0
Gr Liv Area        0
Full Bath          0
Half Bath          0
Bedroom AbvGr      0
Kitchen AbvGr      0
TotRms AbvGrd      0
Fireplaces         0
Garage Cars        0
Garage Area        0
Wood Deck SF       0
Open Porch SF      0
Enclosed Porch     0
3Ssn Porch         0
Screen Porch       0
Pool Area          0
Misc Val           0
SalePrice          0
dtype: int64
```
</div>
</div>
</div>



### Correlating Feature Columns with Target Columns
I will show the the correlation between feature columns and target columns(SalesPrice) by percentage. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train_subset = train[full_cols_series.index]
corrmat = train_subset.corr()
sorted_corrs = corrmat['SalePrice'].abs().sort_values()
print(sorted_corrs)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Misc Val           0.009903
3Ssn Porch         0.038699
Low Qual Fin SF    0.060352
Order              0.068181
MS SubClass        0.088504
Overall Cond       0.099395
Screen Porch       0.100121
Bedroom AbvGr      0.106941
Kitchen AbvGr      0.130843
Pool Area          0.145474
Enclosed Porch     0.165873
2nd Flr SF         0.202352
Half Bath          0.272870
Lot Area           0.274730
Wood Deck SF       0.319104
Open Porch SF      0.344383
TotRms AbvGrd      0.483701
Fireplaces         0.485683
Full Bath          0.518194
1st Flr SF         0.657119
Garage Area        0.662397
Garage Cars        0.663485
Gr Liv Area        0.698990
Overall Qual       0.804562
SalePrice          1.000000
Name: SalePrice, dtype: float64
```
</div>
</div>
</div>



### Correlation Matrix Heatmap
We now have a decent list of candidate features to use in our model, sorted by how strongly they're correlated with the SalePrice column. For now, I will keep only the features that have a correlation of 0.3 or higher. This cutoff is a bit arbitrary and, in general, it's a good idea to experiment with this cutoff. For example, you can train and test models using the columns selected using different cutoffs and see where your model stops improving.

The next thing we need to look for is for potential collinearity between some of these feature columns. Collinearity is when two feature columns are highly correlated and stand the risk of duplicating information. If we have two features that convey the same information using two different measures or metrics, we need to choose just one or predictive accuracy can suffer.

While we can check for collinearity between two columns using the correlation matrix, we run the risk of information overload. We can instead generate a correlation matrix heatmap using Seaborn to visually compare the correlations and look for problematic pairwise feature correlations. Because we're looking for outlier values in the heatmap, this visual representation is easier.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import seaborn as sns
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,6))
strong_corrs = sorted_corrs[sorted_corrs > 0.3]
corrmat = train_subset[strong_corrs.index].corr()
ax = sns.heatmap(corrmat)
ax

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1a19605668>
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/linearregression_22_1.png)

</div>
</div>
</div>



### Train and Test the Model
Based on the correlation matrix heatmap, we can tell that the following pairs of columns are strongly correlated:

- Gr Liv Area and TotRms AbvGrd
- Garage Area and Garage Cars

We will only use one of these pairs and remove any columns with missing values



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
final_corr_cols = strong_corrs.drop(['Garage Cars', 'TotRms AbvGrd'])
features = final_corr_cols.drop(['SalePrice']).index
target = 'SalePrice'
clean_test = test[final_corr_cols.index].dropna()

lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])

train_mse = mean_squared_error(train_predictions, train[target])
test_mse = mean_squared_error(test_predictions, clean_test[target])

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(train_rmse)
print(test_rmse)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
34173.97629185851
41032.026120197705
```
</div>
</div>
</div>



## Removing Low Variance Features
---
The last technique I will explore is removing features with low variance. When the values in a feature column have low variance, they don't meaningfully contribute to the model's predictive capability. On the extreme end, let's imagine a column with a variance of 0. This would mean that all of the values in that column were exactly the same. This means that the column isn't informative and isn't going to help the model make better predictions.

To make apples to apples comparisions between columns, we need to standardize all of the columns to vary between 0 and 1. Then, we can set a cutoff value for variance and remove features that have less than that variance amount.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
unit_train = train[features]/(train[features].max())
sorted_vars = unit_train.var().sort_values()
print(sorted_vars)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Open Porch SF    0.013938
Gr Liv Area      0.018014
Full Bath        0.018621
1st Flr SF       0.019182
Overall Qual     0.019842
Garage Area      0.020347
Wood Deck SF     0.033064
Fireplaces       0.046589
dtype: float64
```
</div>
</div>
</div>



### Final Model
Let's set a cutoff variance of 0.015, remove the Open Porch SF feature, and train and test a model using the remaining features.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
features = features.drop(['Open Porch SF'])

clean_test = test[final_corr_cols.index].dropna()

lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])

train_mse = mean_squared_error(train_predictions, train[target])
test_mse = mean_squared_error(test_predictions, clean_test[target])

train_rmse_2 = np.sqrt(train_mse)
test_rmse_2 = np.sqrt(test_mse)
print(lr.intercept_)
print(lr.coef_)
print(train_rmse_2)
print(test_rmse_2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
-112764.87061464708
[   37.88152677  7086.98429942 -2221.97281278    43.18536387
    64.88085639    38.71125489 24553.18365123]
34372.696707783965
40591.42702437726
```
</div>
</div>
</div>



The final model will be a 7-dimension linear function which looks like:
$$
Y = -112765 + 37.9 * Wood Deck + 7087 * Fire Places - 2222 * Full Bath + 43 * 1st Fle SF + 65 * garage Area + 39 * Liv area + 24553 * Overall Qual
$$



## Feature Transformation
---
To understand how linear regression works, I have stuck to using features from the training dataset that contained no missing values and were already in a convenient numeric representation. In this section, we'll explore how to transform some of the the remaining features so we can use them in our model. Broadly, the process of processing and creating new features is known as feature engineering.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train = data[0:1460]
test = data[1460:]
train_null_counts = train.isnull().sum()
df_no_mv = train[train_null_counts[train_null_counts==0].index]

```
</div>

</div>



### Categorical Features
You'll notice that some of the columns in the data frame df_no_mv contain string values. To use these features in our model, we need to transform them into numerical representations:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
text_cols = df_no_mv.select_dtypes(include=['object']).columns

for col in text_cols:
    print(col+":", len(train[col].unique()))
for col in text_cols:
    train[col] = train[col].astype('category')
train['Utilities'].cat.codes.value_counts()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
('MS Zoning:', 6)
('Street:', 2)
('Lot Shape:', 4)
('Land Contour:', 4)
('Utilities:', 3)
('Lot Config:', 5)
('Land Slope:', 3)
('Neighborhood:', 26)
('Condition 1:', 9)
('Condition 2:', 6)
('Bldg Type:', 5)
('House Style:', 8)
('Roof Style:', 6)
('Roof Matl:', 5)
('Exterior 1st:', 14)
('Exterior 2nd:', 16)
('Exter Qual:', 4)
('Exter Cond:', 5)
('Foundation:', 6)
('Heating:', 6)
('Heating QC:', 4)
('Central Air:', 2)
('Electrical:', 4)
('Kitchen Qual:', 5)
('Functional:', 7)
('Paved Drive:', 3)
('Sale Type:', 9)
('Sale Condition:', 5)
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0    1457
2       2
1       1
dtype: int64
```


</div>
</div>
</div>



### Dummy Coding
When we convert a column to the categorical data type, pandas assigns a number from 0 to n-1 (where n is the number of unique values in a column) for each value. The drawback with this approach is that one of the assumptions of linear regression is violated here. Linear regression operates under the assumption that the features are linearly correlated with the target column. For a categorical feature, however, there's no actual numerical meaning to the categorical codes that pandas assigned for that colum. An increase in the Utilities column from 1 to 2 has no correlation value with the target column, and the categorical codes are instead used for uniqueness and exclusivity (the category associated with 0 is different than the one associated with 1).

The common solution is to use a technique called _dummy coding_. Dummy coding takes a categorical variable and turns it into a set of dichotomous (0 or 1) variables. For example, if we were to look at feature of 'House Style' we would see that there are three participants, each with a nominal variable:

| Style | Code  |
| ----- | ----- |
| 1 Floor | 1 |
| 1.5 Floors | 2 |
| 2 Floors | 3 |

can be made useful by setting 1 Floor as a baseline and then comparing the other categories to that baseline, like so:

| Style | 1.5 Floors | 2 Floors |
| :---: | :--------: | :------: |
| 1.5 Floors | 1 | 0 |
| 2 Floors | 0 | 1 |
| 1 Floor | 0 | 0 |

The pandas function `.get_dummies` will perform this conversion automatically:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
dummy_cols = pd.DataFrame()
for col in text_cols:
    col_dummies = pd.get_dummies(train[col])
    train = pd.concat([train, col_dummies], axis=1)
    del train[col]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train['years_until_remod'] = train['Year Remod/Add'] - train['Year Built']

```
</div>

</div>



## Missing Values
---
Now I will focus on handling columns with missing values. When values are missing in a column, there are two main approaches we can take:

- Remove rows containing missing values for specific columns
Pro: Rows containing missing values are removed, leaving only clean data for modeling
Con: Entire observations from the training set are removed, which can reduce overall prediction accuracy
- Impute (or replace) missing values using a descriptive statistic from the column
Pro: Missing values are replaced with potentially similar estimates, preserving the rest of the observation in the model.
Con: Depending on the approach, we may be adding noisy data for the model to learn

Given that we only have 1460 training examples (with ~80 potentially useful features), we don't want to remove any of these rows from the dataset. Let's instead focus on imputation techniques.

We'll focus on columns that contain at least 1 missing value but less than 365 missing values (or 25% of the number of rows in the training set). There's no strict threshold, and many people instead use a 50% cutoff (if half the values in a column are missing, it's automatically dropped). Having some domain knowledge can help with determining an acceptable cutoff value.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
df_missing_values = train[train_null_counts[(train_null_counts>0) & (train_null_counts<584)].index]

print(df_missing_values.isnull().sum())
print(df_missing_values.dtypes)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Lot Frontage      249
Mas Vnr Type       11
Mas Vnr Area       11
Bsmt Qual          40
Bsmt Cond          40
Bsmt Exposure      41
BsmtFin Type 1     40
BsmtFin SF 1        1
BsmtFin Type 2     41
BsmtFin SF 2        1
Bsmt Unf SF         1
Total Bsmt SF       1
Bsmt Full Bath      1
Bsmt Half Bath      1
Garage Type        74
Garage Yr Blt      75
Garage Finish      75
Garage Qual        75
Garage Cond        75
dtype: int64
Lot Frontage      float64
Mas Vnr Type       object
Mas Vnr Area      float64
Bsmt Qual          object
Bsmt Cond          object
Bsmt Exposure      object
BsmtFin Type 1     object
BsmtFin SF 1      float64
BsmtFin Type 2     object
BsmtFin SF 2      float64
Bsmt Unf SF       float64
Total Bsmt SF     float64
Bsmt Full Bath    float64
Bsmt Half Bath    float64
Garage Type        object
Garage Yr Blt     float64
Garage Finish      object
Garage Qual        object
Garage Cond        object
dtype: object
```
</div>
</div>
</div>



### Inputing Missing Values
It looks like about half of the columns in `df_missing_values` are string columns (object data type), while about half are float64 columns. For numerical columns with missing values, a common strategy is to compute the mean, median, or mode of each column and replace all missing values in that column with that value. So we will take the float64 columns and fill all of the missing values with the mean using the function `df_missing_values.mean()`:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
float_cols = df_missing_values.select_dtypes(include=['float'])
float_cols = float_cols.fillna(df_missing_values.mean())
print(float_cols.isnull().sum())

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Lot Frontage      0
Mas Vnr Area      0
BsmtFin SF 1      0
BsmtFin SF 2      0
Bsmt Unf SF       0
Total Bsmt SF     0
Bsmt Full Bath    0
Bsmt Half Bath    0
Garage Yr Blt     0
dtype: int64
```
</div>
</div>
</div>



And as you can see from the print statement, now all of the float64 columns are without missing values. 



## Conclusion
---
This Notebook talks about how to do linear regression in machine learning by analyzing the real example of the Ames Housing dataset. In this case, to do the linear regression not only means we need to figure out the correlation among all the variable, but also eliminate the variable with either insignificant influence or missing value. 



<a href="https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/linearregression.ipynb" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"> </a>


