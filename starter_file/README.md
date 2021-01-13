# Capstone - Azure Machine Learning Engineer
### Clarisse Ribeiro

In this project, we use the knowledge obtained in **Machine Learning Engineer with Microsoft Azure Nanodegree Program** to solve an interesting problem. 

The problem chosen is the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic).  In the famous Titanic shipwreck, some passengers were more likely to survive than others. The dataset from Kaggle platform presents information about 871 passengers and a column that states if they have survived or not. The ultimate goal is to build a model predicts which passengers survived the Titanic shipwreck.

Here we do this in two different ways:
1) using AutoML;
2) Using a customized Logistic Regression model from SKLearn framework whose hyperparameters are tuned using HyperDrive

We then compare the performance of both the models and deploy the best performing model.

## Dataset

### Overview

The dataset chosen for this project is the one from [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic). 

In the famous Titanic shipwreck, some passengers were more likely to survive than others. The dataset presents information about 871 passengers and a column that states if they have survived or not.

Variable | Definition | Key
------------ | ------------- | -------------
Survived | Survival | 0 = No, 1 = Yes
Nclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd
Name | Name | name of the passenger
Age	| Age | in years
Pibsp | # of siblings / spouses aboard the Titanic	| 
Parch	# of parents / children aboard the Titanic	| 
Ticket | Ticket number	|
Fare | Passenger fare | 
Cabin | Cabin number | 
Q | Port of Embarkation	is Q = Queenstown | 0 = No, 1 = Yes
S | Port of Embarkation	is S = Southampton | 0 = No, 1 = Yes
male | Is male. If not, we consider female. | 0 = No, 1 = Yes

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
https://drive.google.com/file/d/1-DRqQ1hwh7izFWY5uOYBsdrsliRKAMXF/view?usp=sharing
