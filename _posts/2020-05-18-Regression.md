---
title: "Regression"
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
2. Homoscedasticity
3. Multivariate normality
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

## Measure of Accuracy

### Mean Squared Error
The root mean squared error in a linear regression problem is given by the equation ![mse](http://mathurl.com/y9brzcnn.png) which is the sum of squared differences between the actual value ![actualValue](http://mathurl.com/kt496dt.png) and the predicted value ![yhat](http://mathurl.com/yc3fp4p7.png) for each of the rows in the dataset (index iterated over `i`).

## Intuition (Univariate Linear Regression)

### Minimizing the error term we have above
We do so by going through the following steps: 

1. We write the equation ![mse](http://mathurl.com/y9brzcnn.png) again but we replace ![yhat](http://mathurl.com/yc3fp4p7.png) with the equation of the line that we are to predict. Let's say ![predictionLine](http://mathurl.com/y94r3wvh.png) where we don't know the values of ![a](http://mathurl.com/25elof5.png) and ![b](http://mathurl.com/25js5ug.png) yet. 

	With this, our updated equation or the error term becomes ![error](http://mathurl.com/yd5kfvsb.png). 

2. We now need to minimize the error term ![E](http://mathurl.com/y82dzd23.png) with respect to ![a](http://mathurl.com/25elof5.png) and ![b](http://mathurl.com/25js5ug.png) both. For this we use calculus method of partial derivatives. 
3. We calculate partial derivative of ![E](http://mathurl.com/y82dzd23.png) w.r.t. ![a](http://mathurl.com/25elof5.png) and that can be written as:
	
	![partialDiffA](http://mathurl.com/yb8wutve.png) ![parDifA](http://mathurl.com/y8sa4nwz.png) `-1`
	
	Now we calculate partial derivate of the same equation w.r.t. ![b](http://mathurl.com/25js5ug.png) and that can be written and simplified as:
	
	![parDifB](http://mathurl.com/yb8v6uar.png) ![parDifB2](http://mathurl.com/y98a7qgf.png) `-2`
	
4. In order to minimize we equate these equations to zero and solve the equations:
	
	By solving `Eq 1`, we get 
	
	![eq1zero](http://mathurl.com/y7r55sjx.png) `-3`
	
	By solving `Eq 2`, we get 
	
	![eq2zero](http://mathurl.com/yd2uztuy.png) `-4`
	
5. Now we have two equations `3` and `4`. We can use these to solve for ![a](http://mathurl.com/25elof5.png) and ![b](http://mathurl.com/25js5ug.png), upon doing so we get the following values:
	
	![valA](http://mathurl.com/y8leyvd3.png) `-5`
	
	![valB](http://mathurl.com/ycq57l2z.png) `-6`
	
6. Now that we have  these equations, we can divide boh tops and bottoms by N, so that all our summation terms can be turned into means. For instance ![](http://mathurl.com/y8lfwpmw.png). We can divide the equations `5` and `6` with ![n2](http://mathurl.com/ycsnzgo2.png) to get the following results:

	![ares](http://mathurl.com/yae3fw4d.png), 
	
	![bres](http://mathurl.com/ybnzy6jd.png)


## Intuition (Multivariate Linear Regression)

### Base equation

A multivariate Linear Regression can be represented as 

![multilinreg](http://mathurl.com/ybupzufq.png)

where ![yhat](http://mathurl.com/oz8dctm.png) is the list of predictions, ![wt](http://mathurl.com/yckejlne.png) is the vector of weights for each variable, ![x](http://mathurl.com/4dpgym.png) is the set of parameters 
Note that we could have more than one input variable. In this case, we call it 
**multiple linear regression**. Adding extra input variables just means that 
we’ll need to find more weights. For this exercise, we will only consider a 
simple linear regression.

## When to Use

Linear regression is a useful technique but isn’t always the right choice for 
your data. Linear regression is a good choice when there is a linear 
relationship between your independent and dependent variables and you are 
trying to predict continuous values [*Figure 1*].

It is not a good choice when the relationship between independent and 
dependent variables is more complicated or when outputs are discrete values. 
For example, *Figure 3* shows a data set that does not have a linear 
relationship so linear regression would not be a good choice. 

.. figure:: _img/Not_Linear.png
   
   **Figure3. A sample data set without a linear relationship** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/not_linear_regression.py

It is worth noting that sometimes you can apply transformations to data so 
that it appears to be linear. For example, you could apply a logarithm to 
exponential data to flatten it out. Then you can use linear regression on the 
transformed data. One method of transforming data in :code:`sklearn` is 
documented here_.

.. _here: https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html

*Figure 4* is an example of data that does not look linear but can be 
transformed to have a linear relationship.

.. figure:: _img/Exponential.png
   
   **Figure 4. A sample data set that follows an exponential curve** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/exponential_regression.py

*Figure 5* is the same data after transforming the output variable with a 
logarithm.

.. figure:: _img/Exponential_Transformed.png
   
   **Figure 5. The data set from Figure 4 after applying a logarithm to the 
   output variable** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/exponential_regression_transformed.py


*************
Cost Function
*************
Once we have a prediction, we need some way to tell if it’s reasonable. A 
**cost function** helps us do this. The cost function compares all the 
predictions against their actual values and provides us with a single number 
that we can use to score the prediction function. *Figure 6* shows the cost 
for one such prediction.

.. figure:: _img/Cost.png
   
   **Figure 6. The plot from Figure 2 with the cost of one prediction 
   emphasized** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression_cost.py

Two common terms that appear in cost functions are the **error** and 
**squared error**. The error [*Equation 2*] is how far away from the actual 
value our prediction is.

.. figure:: _img/Error_Function.png
   
   **Equation 2. An example error function**

Squaring this value gives us a useful expression for the general error 
distance as shown in *Equation 3*.

.. figure:: _img/Square_Error_Function.png
   
   **Equation 3. An example squared error function**

We know an error of 2 above the actual value and an error of 2 below the 
actual value should be about as bad as each other. The squared error makes 
this clear because both of these values result in a squared error of 4.

We will use the Mean Squared Error (MSE) function shown in *Equation 4* as our 
cost function. This function finds the average squared error value for all of 
our data points.

.. figure:: _img/MSE_Function.png
   
   **Equation 4. The Mean Squared Error (MSE) function**

Cost functions are important to us because they measure how accurate our model 
is against the target values. Making sure our models are accurate will remain 
a key theme throughout later modules.


*******
Methods
*******
A lower cost function means a lower average error across the data points. In 
other words, lower cost means a more accurate model for the data set. We will 
briefly mention a couple of methods for minimizing the cost function.

Ordinary Least Squares
======================
Ordinary least squares is a common method for minimizing the cost function. In 
this method, we treat the data as one big matrix and use linear algebra to 
estimate the optimal values of the coefficients in our linear equation. 
Luckily, you don't have to worry about doing any linear algebra because the 
Python code handles it for you. This also happens to be the method used for 
this modules code.




## Assumptions of Linear Regression
 The regression has five key assumptions:

- **Linear relationship**
First, linear regression needs the relationship between the independent and dependent variables to be linear.  It is also important to check for outliers since linear regression is sensitive to outlier effects.  The linearity assumption can best be tested with scatter plots, the following two examples depict two cases, where no and little linearity is present.
![NonLinear](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression01.jpg)
![LessNonLinear](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression02.jpg)


- **Multivariate normality**
Secondly, the linear regression analysis requires all variables to be multivariate normal.  This assumption can best be checked with a histogram or a Q-Q-Plot.  Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test.  When the data is not normally distributed a non-linear transformation (e.g., log-transformation) might fix this issue.

![histogram](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression03.jpg)
![QQPLot](https://www.statisticssolutions.com/wp-content/uploads/2010/01/linearregression04.jpg)


- **No or little multicollinearity**
Thirdly, linear regression assumes that there is little or no multicollinearity in the data.  Multicollinearity occurs when the independent variables are too highly correlated with each other.

Multicollinearity may be tested with three central criteria:

1) Correlation matrix – when computing the matrix of Pearson’s Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.

2) Tolerance – the tolerance measures the influence of one independent variable on all other independent variables; the tolerance is calculated with an initial linear regression analysis.  Tolerance is defined as T = 1 – R² for these first step regression analysis.  With T < 0.1 there might be multicollinearity in the data and with T < 0.01 there certainly is.

3) Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is defined as VIF = 1/T. With VIF > 5 there is an indication that multicollinearity may be present; with VIF > 10 there is certainly multicollinearity among the variables.

If multicollinearity is found in the data, centering the data (that is deducting the mean of the variable from each score) might help to solve the problem.  However, the simplest way to address the problem is to remove independent variables with high VIF values.

Fourth, linear regression analysis requires that there is little or no autocorrelation in the data.  Autocorrelation occurs when the residuals are not independent from each other.  For instance, this typically occurs in stock prices, where the price is not independent from the previous price.

4) Condition Index – the condition index is calculated using a factor analysis on the independent variables.  Values of 10-30 indicate a mediocre multicollinearity in the linear regression variables, values > 30 indicate strong multicollinearity.

If multicollinearity is found in the data centering the data, that is deducting the mean score might help to solve the problem.  Other alternatives to tackle the problems is conducting a factor analysis and rotating the factors to insure independence of the factors in the linear regression analysis.



- **No auto-correlation**
Fourthly, linear regression analysis requires that there is little or no autocorrelation in the data.  Autocorrelation occurs when the residuals are not independent from each other.  In other words when the value of y(x+1) is not independent from the value of y(x).

While a scatterplot allows you to check for autocorrelations, you can test the linear regression model for autocorrelation with the Durbin-Watson test.  Durbin-Watson’s d tests the null hypothesis that the residuals are not linearly auto-correlated.  While d can assume values between 0 and 4, values around 2 indicate no autocorrelation.  As a rule of thumb values of 1.5 < d < 2.5 show that there is no auto-correlation in the data. However, the Durbin-Watson test only analyses linear autocorrelation and only between direct neighbors, which are first order effects.



- **Homoscedasticity**
- A note about sample size.  In Linear regression the sample size rule of thumb is that the regression analysis requires at least 20 cases per independent variable in the analysis.

The last assumption of the linear regression analysis is homoscedasticity.  The scatter plot is good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line).  The following scatter plots show examples of data that are not homoscedastic (i.e., heteroscedastic):

The Goldfeld-Quandt Test can also be used to test for heteroscedasticity.  The test splits the data into two groups and tests to see if the variances of the residuals are similar across the groups.  If homoscedasticity is present, a non-linear correction might fix the problem.



















Gradient Descent
================
Gradient descent is an iterative method of guessing the coefficients of our 
linear equation in order to minimize the cost function. The name comes from 
the concept of gradients in calculus. Basically this method will slightly move 
the values of the coefficients and monitor whether the cost decreases or not. 
If the cost keeps increasing over several iterations, we stop because we've 
probably hit the minimum already. The number of iterations and tolerance 
before stopping can both be chosen to fine tune the method.

Below are the relevant lines of Python code from this module modified to use 
gradient descent.

.. code-block:: python

   # Create a linear regression object
   regr = linear_model.SGDRegressor(max_iter=10000, tol=0.001)


****
Code
****
This module's main code is available in the linear_regression_lobf.py_ file.

.. _linear_regression_lobf.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression_lobf.py

All figures in this module were created with simple modifications of the 
linear_regression.py_ code.

.. _linear_regression.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression.py

In the code, we analyze a data set with a linear relationship. We split the 
data into a training set to train our model and a testing set to test its 
accuracy. You may have guessed that the model used is based on linear 
regression. We also display a nice plot of the data with a line of best fit.


**********
Conclusion
**********
In this module, we learned about linear regression. This technique helps us 
model data with linear relationships. Linear relationships are fairly simple 
but still show up in a lot of data sets so this is a good technique to know. 
Learning about linear regression is a good first step towards learning more 
complicated analysis techniques. We will build on a lot of the concepts 
covered here in later modules.


************
References
************

1. https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
2. https://machinelearningmastery.com/linear-regression-for-machine-learning/
3. https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
#. https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
#. https://medium.com/analytics-vidhya/linear-regression-in-python-from-scratch-24db98184276
#. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#. https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html


