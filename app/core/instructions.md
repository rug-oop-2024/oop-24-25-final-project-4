# Instructions

## Uploading and using datasets
To upload datasets, navigate to the datasets page. Here you can upload and name your datasets. 

To use the datasets, navigate to the modelling page and select your desired dataset.

## Selecting what to do with the data
Click on the bar below the `What do you want to do with your data` to open a drop-down select menu.

If you wish to *plot* your data, select `create a plot` from the drop-down select list.

To *train* a model with your data, select `Train a model` from the drop-down select list.

## How to train a model
The following steps will allow you to train a model using your dataset.

### Choosing predictor and outcome variables
Select the desired **input feature or features** using the `Select input features` drop-down menu.

*This is your predictor or independent variable, it will be the attributes the model uses to make predictions*

Select the desired **target feature**  using the `Select a target feature` drop-down menu.

*This is your outcome or dependent variable, it will be the value the model aims to predict*

**Note that choosing a categorical variable will limit your model choices to classification types and numerical variables will enforce regression type models.**

It will state which task type it detects after the choices are made, either classification or regression.

### Choosing a model
Select the desired model you wish to train using the `Select a model` drop-down menu.

*To see definitions of each model and their purpose, navigate back to the welcome page.*

*If the desired model is not in the list, make sure that the selected target variable is of the correct feature type*

### Dataset split
Drag the circle across the bar to choose how you want your data split between training and testing.

*Defaults at a 80/20 Split: 80% of the data is used for training, and 20% is reserved for testing.*

### Choosing metrics
Select the desired *metric(s)* you wish to test the model's effectiveness with using the `Select the metrics you want to use` drop-down menu

*To see definitions of each metric, how they work and their purpose, navigate back to the welcome page.*

*Up to three per model type are currently available.*

### Train the model
Make sure the pipeline summary is correctly stating your selections and click `Train the model` to train the model with your data.