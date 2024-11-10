# A machine learning application

Welcome to our *Streamlit* application that trains models.
With this application you can plot your data or train a model using it.

## We offer three **regression** models for prediciting measurments and quantities (numerical values)

### 1. Decision Tree Regressor
Decision trees are tools for decision-making which follow a tree structure that resembles a flowchart. The **decision-tree algorithm** works with both *categorical* and *numerical* output values, since this is the **regressor** it works with *numerical*. Decision tree regression can observe an objects features to train a model that predicts meaniningful numerical outputs, for example, it can be used to predict the sale revenue of a store based on features such as the number of customers and the season. The model will build a tree structure to predict the total revenue *(numerical value)* for future periods.

### 2. Multiple Linear Regression
Multiple Linear Regression (MLR) is a statistical technique used to model the relationship between a dependent variable and two (or more) independent variables. MLR assumes a **linear relationship** between the variables and tries to find the best-fitting line (or plane for multiple variables) which minimizes the error between predicted and actual values. The MLR we provide is best used to predict numerical outcomes, like as predicting house prices based on features like size, number of bedrooms, and location. The model **learns the weights** of each feature and combines them to make **predictions**.

### 3. Random Forest Regressor
The **random (decision) forest** machine-learning algorithm is an *ensemble learning method*. Ensemble learning involves finding a more accurate/stabler prediction by combining prediction from various models. Random forest specifically focuses on combining different **decision tree predictions**. It works well with larger, more complicated datasets. One benefit of this algorithm is that it minimizes overfitting while keeping its predictive accuracy high. Forest tree works for both regression and classification, since this is the **regressor**, we use it for regression, it can be used when working with numerical output variables, the final prediction will **average** all the tree predictions.



## We offer three **classification** models for predicting classes of new instances (categorical values)

### 1. Decision Tree Classifier
Decision trees are tools for decision-making which follow a tree structure that resembles a flowchart. The **decision-tree algorithm** works with both categorical and numerical output values, since this is the **classifier** it works with *categorical*. Decision tree classificaton can observe an objects features to train a model that predicts meaniningful categorical outputs, for example, it can be used to predict whether a customer will buy a product *(categorical "yes" or "no")* based on features such as their age and income. The model will build a tree structure by learning patterns in the data and using it to assign incoming customers to classes of "buyers" and "non-buyers".

### 4. K-Nearest Neighbour
K-Nearest Neighbors (KNN) is a classification algorithm that assigns a class to an instance based on the **majority class** among its *K-nearest neighbors* in the feature space. The idea is that similar instances often are from of the same class. When assigning a class to a new instance, KNN calculates the distance between the instance and all other instances in the training dataset, selects the *K-nearest* instances, and predicts by assigning the **most common class** among those neighbors as the instance's class. KNN works well for classification tasks and it handles situations where the decision boundary between classes is complex and non-linear. Since it is *non-parametric*, it does not assume anything about the distribution of the data, it is also less sensisitve to outliers relative to other algorithms.

### 3. Random Forest Classifier
The **random (decision) forest** machine-learning algorithm is an *ensemble learning method*. Ensemble learning involves finding a more accurate/stabler prediction by combining prediction from various models. Random forest specifically focuses on combining different **decision tree predictions**. It works well with larger, more complicated datasets. One benefit of this algorithm is that it minimizes overfitting while keeping its predictive accuracy high. Forest tree works for both regression and classification, since this is the **classifier**, we use it for classification, it can be used when working with categorical output variables, the final prediction will be the class which the **majority** of trees assigned to the instance.

## Additionally we offer six metrics, three for regression models and three for classification models
Metrics are used to evaluate the models' performance. They can be used to check ow well the model predicts based on the data it has been given. The effectiveness of classification and regression models is not universal, so we provide three ways to assess prediction accuracy and effectiveness for each.

### The three regression model metrics
#### Mean Squared Error
*"On average, how far off are the predicted values from the true values, not all errors are equal, the larger the error, the more we penalize?"*

Mean Squared Error calculates the average of squared differences between the predicted values and the actual values.

MSE = (1/*number of data points*) * Σ (*actual value* - *predicted value*)²

#### Mean Absolute Error
*"How far off are the predicted values from the true values, treating all errors equally, on average?"*

MAE = (1/*number of data points*) * Σ |*actual value* - *predicted value*|

#### R^2 Score
*"How well does the model explain the variability in the actual data? How good is the model at predicting the target variable?"*

R-squared calculates the proportion of variance in the dependent variable that can be predicted from the independent variables. It reprsents how well-suited the model is for the given data.

R² = 1 - Σ (*actual value* - *predicted value*)² / Σ (*actual value* - *mean of actual values*)²

### The three classification model metrics
#### Accuracy
*"Of all the predictions made by the model, how many did it get right?"*

Accuracy calculates the percentage of correctly predicted instances out of all instances. 

Accuracy = (Correct Predictions) / (Total Predictions)

#### Precision
*"Of all the predicted positive instances identified by the model, how many were actually positive?"*

Precision calculates the precentage of correct postivie predictions, so out of the total number of predicted positives, the number of postivies that were truly positive.

Precision = (True Positives) / (True Positives + False Positives)

#### Recall
*"Of all the actual positive instances, how many did the model correctly find?"*

Recall calculates the percentage of correctly identified postive instances, so out of the total number of postive instances identified, the number of true positives the model identified.

Recall = (True Positives) / (True Positives + False Negatives)
