---
title: "Regression - Explained"
layout: post
category: note
tags: [machine-leaning,regression,algorithms]
excerpt: "Machine Learning Algorithms for Regression"
---

# Regression

## Overview
Regression is an analysis that assesses whether one or more predictor variables or independent variables explain the dependent (target) variable. 

## Linear Regression
**Linear regression** is a technique used to analyze a **linear relationship** 
between **input** variables and a single **output** variable. A **linear 
relationship** means that the data points tend to follow a straight line. 
**Simple linear regression** involves only a single input variable. *Figure 1* 
shows a data set with a linear relationship.
 
The idea is to fit a straight line in the n-dimensional space that holds all our observational points. This would constitute forming an equation of the form **y = mx + c**. Because we have multiple variables, we might need to extend this **mx** to be **m<sub>1</sub>x<sub>1</sub>**, **m<sub>2</sub>x<sub>2</sub>** and so on. This extensions results in the following mathematical representation between the independent and dependent variables:

![eq](http://mathurl.com/y8eahwj3.png)

where 

1. **y** = dependent variable/outcome
2. **x<sub>1</sub> to x<sub>n</sub>** are the dependent variables
3. **b<sub>0</sub> to b<sub>n</sub>** are the coefficients of the linear model

A linear regression models bases itself on the following assumptions:

1. Linearity
2. Homoscedasticity or 
3. Multivariate Normality
4. Independence of errors
5. Lack of multicollinearity

If these assumptions do not hold, then linear regression probably isn't the model for your problem. 



## Variable Selection

The dataset will, more often than not, contain columns that do not have any effect on the dependent variable. This becomes problematic because we don't want to add too much noise to the dataset. The variables that do not effect the dependent variable (the outcome) will usually only decrease the performance of our model, and even if they do not, there's always an additional computing complexity that comes along with them. This could influence the costs of building the model, especially when we want to do it iteratively. 

There are five ways for doing feature selection while building a model:

1. **All-in**: If we are aware that all the variables in the dataset are useful and are required for building the model, then we simply use all the variables that are available to us.
2. **Backward Elimination**: The following steps need to be taken in order to conduct a backward elimination feature selection:
	1. 	We set a significance level for a feature to stay in the model
	2. Fit the model with all possible predictors
	3. Consider the predictor with the highest p-value. If P > SL go to **Step 4**, otherwise go to **Step 6**
	4.  Remove the variable
	5.  Fit the model again, and move to **Step 3**
	6.  Finish
3. **Forward Selection**: Although, it may seem like a straightforward procedure, it is quite intricate in practical implementation. The following steps need to be performed in order to make a linear regressor using forward selection:
	1. We set a significance level to enter the model
	2. Fit regression models with each one of those independent variables and then select the one with the lowest p-value
	3. Fit all possible regression models with the one that we selected in previous step and one additional variable.  
	4.  Consider the model with the lowest p-value. If p < SL go to **Step 3** otherwise go to **Step 5**
	5.  Finish and select the second last model 
4. **Bidirectional Elimination**: The algorithm works as follows:
	1. Select a Significance Level to enter and a Significance Level to stay in the model, viz. SLENTER and SLSTAY
	2. Perform the next step of Forward Selection, i.e. all the new variables that are to be added must have p < SLENTER in order to enter the model
	3. Perform all the steps of Backward Elimination, i.e. all the existing variables must have p < SLSTAY in order to stay in the model
	4. No new variables can be added, and no new variables can be removed
		
> The details on variable selection by **Score Comparison** is yet to be found.

The **lower the p-value** is, the **more important** a particular variable is for our model.

> The term **'Step wise regression'** is often used for 2, 3, and 4 but sometimes, it refers only to 4, depending on context. 

**Dummy variables:**
The variables that are created when categorical variables are encoded, are called dummy variables. 

We usually use one-hot encoding to do this, and it might seem like not including one last categorical dummy variable would cause a positive bias towards the rest of the equation but this is not the case. The coefficient for the last dummy variable is included in the b<sub>0</sub> term of the equation.

*Dummy Variable trap*: One can never have all the dummy variables and b<sub>0</sub> in a model at the same time. We need to remove at least one dummy variable for each of the corresponding categorical variables because all of that will be modeled into b<sub>0</sub>. 




## When to Use

Linear regression is a useful technique but isn’t always the right choice for 
your data. Linear regression is a good choice when there is a linear 
relationship between your independent and dependent variables and you are 
trying to predict continuous values.

It is not a good choice when the relationship between independent and 
dependent variables is more complicated or when outputs are discrete values. 


## Assumptions of Linear Regression

- A note about sample size.  In Linear regression the sample size rule of thumb is that the regression analysis requires at least 20 cases per independent variable in the analysis.

 The regression has five key assumptions:

- **Linear relationship**
First, linear regression needs the relationship between the independent and dependent variables to be linear.  It is also important to check for outliers since linear regression is sensitive to outlier effects.  The linearity assumption can best be tested with scatter plots, the following two examples depict two cases, where no and little linearity is present.
![NonLinear](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression01.jpg)
![LessNonLinear](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression02.jpg)

- **Multivariate normality**
Secondly, the linear regression analysis requires all variables to be multivariate normal.  This assumption can best be checked with a histogram or a Q-Q-Plot.  Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test. 
When the data is not normally distributed a non-linear transformation (e.g., log-transformation) might fix this issue.
![histogram](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression03.jpg)
![QQPLot](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression04.jpg)


- **No or little multicollinearity**
Thirdly, linear regression assumes that there is little or no multicollinearity in the data.  Multicollinearity occurs when the independent variables are too highly correlated with each other.
Multicollinearity may be tested with three central criteria:

1) Correlation matrix – when computing the matrix of Pearson’s Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.
2) Tolerance – the tolerance measures the influence of one independent variable on all other independent variables; the tolerance is calculated with an initial linear regression analysis.  Tolerance is defined as T = 1 – R² for these first step regression analysis.  With T < 0.1 there might be multicollinearity in the data and with T < 0.01 there certainly is.
3) Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is defined as VIF = 1/T. With VIF > 5 there is an indication that multicollinearity may be present; with VIF > 10 there is certainly multicollinearity among the variables.

If multicollinearity is found in the data, centering the data (that is deducting the mean of the variable from each score) might help to solve the problem.  However, the simplest way to address the problem is to remove independent variables with high VIF values.

4) Condition Index – the condition index is calculated using a factor analysis on the independent variables.  Values of 10-30 indicate a mediocre multicollinearity in the linear regression variables, values > 30 indicate strong multicollinearity.

If multicollinearity is found in the data centering the data, that is deducting the mean score might help to solve the problem.  Other alternatives to tackle the problems is conducting a factor analysis and rotating the factors to insure independence of the factors in the linear regression analysis.

- **No auto-correlation**
Fourthly, linear regression analysis requires that there is little or no autocorrelation in the data.  Autocorrelation occurs when the residuals are not independent from each other.  In other words when the value of y(x+1) is not independent from the value of y(x).

While a scatterplot allows you to check for autocorrelations, you can test the linear regression model for autocorrelation with the Durbin-Watson test.  Durbin-Watson’s d tests the null hypothesis that the residuals are not linearly auto-correlated.  While d can assume values between 0 and 4, values around 2 indicate no autocorrelation.  As a rule of thumb values of 1.5 < d < 2.5 show that there is no auto-correlation in the data. However, the Durbin-Watson test only analyses linear autocorrelation and only between direct neighbors, which are first order effects.

- **Homoscedasticity**

The last assumption of the linear regression analysis is homoscedasticity.  The scatter plot is good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line).  The following scatter plots show examples of data that are not homoscedastic (i.e., heteroscedastic):

The Goldfeld-Quandt Test can also be used to test for heteroscedasticity.  The test splits the data into two groups and tests to see if the variances of the residuals are similar across the groups.  If homoscedasticity is present, a non-linear correction might fix the problem.


Above, we discussed what is Regression and the assumptions or so called limitations of linear regression. Linear Regression is assumed to be the simplest machine learning algorithm the world has ever seen, and yes! it is! We also discussed how your model can give you poor predictions in real time if you don't obey the assumptions of linear regression. Whatever you are going to predict, whether it is stock value, sales or some revenue, linear regression must be handled with care if you want to get best values from it.

## Multiple Linear Regression
When there are more than one independent variable (x1,x2,x3...xn) and one dependent variable (y), its called Multiple Linear Regression. Most linear Regressions are multiple linear regression itself.


## Polynomial Linear Regression
This regression allows us to regress over dependent variable(s) that has a polynomial relationship with the independent variables generally represented.


## Non Linear Regression

As we saw, linear regression says, the data should be linear in nature, there must be a linear relationship. But, wait! the real world data is always non-linear. Yes, so, what should we do, should we try to bring non-linearity into the regression model, or check out the residuals and fitted values, or keep applying transformations and working harder and harder to get the best predictive model using linear regression. Yes or No? Now, the question is.. is there any other way to deal with this, so that we can get a better predictive model without getting into these assumptions of linear regression.

Yes! there is a solution, in fact a bunch of solutions.

There are many different analytic procedures for fitting regressive models of nonlinear nature (e.g., **Generalized Linear/Nonlinear Models (GLZ), Generalized Additive Models (GAM), etc.), or more better models called tree based regressive models, boosted tree based based, support vector machine based regression model etc**.

Most of us know about Decision Trees and Random Forest, it is very common, in case of classification or regression and it is also true that they often perform far better than other regression models with minimum efforts. So, now we will be talking about tree based models such as Decision Trees and ensemble tree based like Random forests. Tree based model have proven themselves to be both reliable and effective, and are now part of any modern predictive modeler’s toolkit.

The bottom line is: You can spend 3 hours playing with the data, generating features and interaction variables for linear regression and get a 77% r-squared; and I can “from sklearn.ensemble import RandomForestRegressor” and in 3 minutes get an 82% r-squared. I am not creating a hype for these tree model, but its the truth.

Let me explain it using some examples for clear intuition with an example. Linear regression is a linear model, which means it works really nicely when the data has a linear shape. But, when the data has a non-linear shape, then a linear model cannot capture the non-linear features. So in this case, you can use the decision trees, which do a better job at capturing the non-linearity in the data by dividing the space into smaller sub-spaces depending on the rules that exist.

Now, the question is when do you use linear regression vs Trees? Let’s suppose you are trying to predict income. The predictor variables that are available are education, age, and city. Now in a linear regression model, you have an equation with these three attributes. Fine. You’d expect higher degrees of education, higher “age” and larger cities to be associated with higher income. But what about a PhD who is 40 years old and living in Scranton Pennsylvania? Is he likely to earn more than a BS holder who is 35 and living in Upper West SIde NYC? Maybe not. Maybe education totally loses its predictive power in a city like Scranton? Maybe age is a very ineffective, weak variable in a city like NYC?

This is where decision trees are handy. The tree can split by city and you get to use a different set of variables for each city. Maybe Age will be a strong second-level split variable in Scranton, but it might not feature at all in the NYC branch of the tree. Education may be a stronger variable in NYC.

Decision Trees, be it Random Forest or Gradient Bossted Machine based Regression, handle messier data and messier relationships better than regression models. ANd there is seldom a dataset in the real world where relationships are not messy. No wonder you will seldom see a linear regression model outperforming RF or GBM. So, this is the main idea behind tree **(Decision Tree Regression) and ensemble based models (Random Forest Regression/Gradient Boosting Regression/ Extreme Boosting Regression/Adaboost Regression)**. 

**Support Vector Regression** is regression equivalent of classification with Support Vector Machine . Much like Support Vector Machines in general, the idea of SVR, is to find a plane(linear or otherwise) in a plane that allows us to make accurate predictions for future data. The regression is done in a way that all the currently available datapoints fit in an error width given by ![epsilon_small](http://mathurl.com/ybr3ffkc.png). This allows us to find a plane which fits the data best and then this can be used to make future predictions on more data. 



## Performance Criteria

### Mean Squared Error
The root mean squared error in a linear regression problem is given by the equation ![mse](http://mathurl.com/y9brzcnn.png) which is the sum of squared differences between the actual value ![actualValue](http://mathurl.com/kt496dt.png) and the predicted value ![yhat](http://mathurl.com/yc3fp4p7.png) for each of the rows in the dataset (index iterated over `i`).
###  R-Squared
R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. R-sqaure is given by the following formula ![rsq](http://mathurl.com/ybf2xwp2.png) 
The idea is to minimize ![ssres](http://mathurl.com/ycattv9v.png) so as to keep the value of ![rsq](http://mathurl.com/3kwwdyh.png) as close as possible to 1. The calculation essentially gives us a numeric value as to how good is our regression line, as compared to the average value.
### Adjusted R-Squared
The value of ![rsq](http://mathurl.com/3kwwdyh.png) is considered to be better as it gets closer to 1, but there's a catch to this statement. The ![rsq](http://mathurl.com/3kwwdyh.png) value can be artifically inflated by simply adding more variables. This is a problem because the complexity of the model would increase due to this and would result in overfitting. The formulae for Adjusted R-squared is mathematically given as:
![](http://mathurl.com/yclkhq5z.png) 
where **p** is the number of regressors and **n** is the sample size. Adjusted R-squared has a penalizing factor that reduces it's value when a non-significant variable is added to the model.
> **p-value** based backward elimination can be useful in removing non-significant variables that we might have added in our model initially.
![Metrices](https://4.bp.blogspot.com/-wG7IbjTfE6k/XGUvqm7TCVI/AAAAAAAAAZU/vpH1kuKTIooKTcVlnm1EVRCXLVZM9cPNgCLcBGAs/s1600/formula-MAE-MSE-RMSE-RSquared.JPG)


### References
1. https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
2. https://machinelearningmastery.com/linear-regression-for-machine-learning/
3. https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
#. https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
#. https://medium.com/analytics-vidhya/linear-regression-in-python-from-scratch-24db98184276
#. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#. https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
