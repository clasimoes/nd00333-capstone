- [Capstone Project - Azure Machine Learning Engineer Nanodegree](#capstone-project---azure-machine-learning-engineer-nanodegree)
    - [Clarisse Ribeiro](#clarisse-ribeiro)
  - [Dataset](#dataset)
    - [Overview](#overview)
    - [Task](#task)
    - [Access](#access)
  - [Automated ML](#automated-ml)
    - [Results](#results)
      - [AutoML Run Details widget](#automl-run-details-widget)
      - [AutoML Best Model Run](#automl-best-model-run)
      - [AutoML Best Model Run Properties](#automl-best-model-run-properties)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Results](#results-1)
      - [HyperDrive Run Details widget](#hyperdrive-run-details-widget)
      - [HyperDrive Best Model Run](#hyperdrive-best-model-run)
      - [HyperDrive Best Model Run Properties](#hyperdrive-best-model-run-properties)
  - [Model Deployment](#model-deployment)
      - [Service of HyperDrive model with "Active" deployment state](#service-of-hyperdrive-model-with-active-deployment-state)
  - [Future improvements](#future-improvements)

# Capstone Project - Azure Machine Learning Engineer Nanodegree


In this project, we use the knowledge obtained in **Machine Learning Engineer with Microsoft Azure Nanodegree Program** to solve an interesting problem. 

The problem chosen is the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic).  In the famous Titanic shipwreck, some passengers were more likely to survive than others. The dataset from Kaggle platform presents information about 871 passengers and a column that states if they have survived or not. The ultimate goal is to build a model that predicts which passengers survived the Titanic shipwreck. The Titanic Dataset is commonly referred to as the "hello world" of data science.

Here we do this in two different ways:
1) Using AutoML;
2) Using a customized model and tuning its hyperparameters with HyperDrive

We then compare the performance of both the models and deploy the best performing model.
The deployment is done using the Azure Python SDK, and creates an endpoint that can be accessed through an HTTP REST API. The step makes it possible for any new data to be evaluated by the best model thorugh the service.

## Dataset

### Overview

The dataset chosen for this project is the one from [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic). 

In the famous Titanic shipwreck, some passengers were more likely to survive than others. The dataset presents information about 871 passengers and a column that states if they have survived or not.

Here we use only the "training" data of the original challenge because this is the data with the "Survived" label, which is necessary for the Supervised Learning algorithms that are used in this project.

Find below the data dictionary:

Variable | Definition | Key
------------ | ------------- | -------------
Survived | Survival | 0 = No, 1 = Yes
Nclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd
Name | Name | name of the passenger (string)
Age	| Age | in years
Pibsp | # of siblings / spouses aboard the Titanic	| 
Parch | # of parents / children aboard the Titanic	| 
Ticket | Ticket number	| string
Fare | Passenger fare | Value (float)
Cabin | Cabin number |  string
Q | Port of Embarkation	is Q = Queenstown | 0 = No, 1 = Yes
S | Port of Embarkation	is S = Southampton | 0 = No, 1 = Yes
male | Is male. If not, we consider the passenger female. | 0 = No, 1 = Yes

The data has been uploaded to [this repository](https://github.com/clasimoes/nd00333-capstone/blob/master/titanic_data/full_capstone.csv) for ease.

### Task
In this project, we aim to create a model with the best possible **Accuracy** to classify if a passenger survives or not the shipwreck.
For this, we use two approaches:

1) **Using AutoML**:
In this approach, we provide the dataset to AutoML and it automatically does the featurization, tries different algorithms, and test the performance of many different models. 

2) **Using HyperDrive**: 
In this case, we test only a single algorithm and create different models by providing different hyperparameters. The chosen algorithm is Logistic Regression using the framework SKLearn.
Unlike AutoML, here we need to manually perform feature scaling, normalization and select interesting columns for our model.

In both cases, the best performing model created during runs can be saved and deployed, and its parameters can be checked both in the Azure ML portal and in the run logs.

The features that are used in this experiment are the ones described in the data dictionary above. However, in the case of the HyperDrive, we manually remove the columns "Name", "Ticket", and "Cabin" which are not supported by the Logistic Regression classifier.

### Access

The data has been uploaded to [this repository](https://github.com/clasimoes/nd00333-capstone/blob/master/titanic_data/full_capstone.csv) for ease.
To access it in our Azure notebooks, we need to download it from an external link into the Azure workspace.

For that, we can use the `Dataset` class, which allows importing tabular data from files on the web.
With that, we become able to create and register a dataset in Azure ML Platform and convert it to a Pandas Dataframe.

```python
    # Create AML Dataset and register it into Workspace
    example_data = 'https://raw.githubusercontent.com/clasimoes/nd00333-capstone/master/titanic_data/full_capstone.csv'
    dataset = Dataset.Tabular.from_delimited_files(example_data)
    #Register Dataset in Workspace
    dataset = dataset.register(...)
```


## Automated ML
For the AutoML run, first we create a compute cluster to run the experiment. In this cluster, we provide 2 to 10 machines with the **"STANDARD_DS12_V2"** configuration.
Because we have 10 nodes in our cluster, we can run up to 9 concurrent iterations in our experiment (1 node is meant to be used by the "parent" experiment).

The constructor of `AutoMLConfig` class takes the following parameters:
* `compute_target`: cluster where the experiment jobs will run;
* `task`: type of ML problem to solve, set as `classification`;
* `experiment_timeout_minutes`: 20;
* `training_data`: the dataset loaded; 
* `label_column_name`: The column that should be predicted, which is the "Survived" one; 
* `path`: the full path to the Azure Machine Learning project folder; 
* `enable_early_stopping`: makes it possible for the AutoML to stop jobs that are not performing well after a minimum number of iterations; 
* `featurization`: indicator that featurization step should be done automatically;
* `debug_log`: The log file to write debug information to; 
* `automl_settings`: other settings passed as a dictionary. 
    * `max_concurrent_iterations`: Represents the maximum number of iterations that would be executed in parallel. Set to 9;
    * `primary_metric`: The metric that Automated Machine Learning will optimize for model selection. We chose to optimize for `Accuracy`.

The `accuracy` metric might be misleading when the training data has very few samples of one of the classes, which is **not** the case in Titanic dataset. The distribution of classes on it is about 40% and 60% of samples in each class. Moreover, accuracy is a straightforward metric and easy to understand. This is the main reason why it has been chosen in this experiment.

Because AutoML is an automated process that might take a long time to use many resources, it is a good idea to enable the early stopping of model training jobs. When the training takes a long time, it can lead to higher costs. The tool is then able to kill jobs that are not performing well, leading to better resource usage

Featurization enables techniques of feature engineering to be applied to the dataset, enhancing the learning process. Examples of featurization steps are imputing missing values, generation of more features, dropping features with no variance, among others. `Data guardrails` is a feature that helps to identify automatically potential issues with the data, like missing values or class imbalance. Setting the `featurization` option to `auto` specifies that, as part of preprocessing, data guardrails and featurization steps are to be done automatically.

### Results
Among 100 experiments included in the AutoML, the best model produced relied on the **Voting Ensemble** algorithm, from the SKLearn framework. This model had an **accuracy** of **83,39%**.

Voting Ensemble uses multiple models as inner estimators and each one has its unique hyperparameters.
I've chose one of the estimators used by the best model to display the parameters used by it here. The classifier is a **Light GBM** algorithm with the following parameters:

* `boosting_type`: 'gbdt'
* `class_weight`: None
* `importance_type`: 'split'
* `learning_rate`: 0.1
* `max_depth`: -1
* `colsample_bytree`: 1.0
* `colsample_bytree`: 1.0
* `min_child_samples`: 20
* `min_child_weight`: 0.001
* `min_split_gain`: 0.0
* `n_estimators`: 100
* `n_jobs`: 1
* `num_leaves`: 31
* `objective`: None
* `random_state`: None
* `reg_alpha`: 0.0
* `reg_lambda`: 0.0
* `silent`: True
* `subsample`: 1.0
* `subsample_for_bin`: 200000
* `subsample_freq`: 0
* `verbose`: -10

Find below the screenshots of the AutoML run details widget, together with the best run details in the Azure ML platform and its properties in the Jupyter notebook.

#### AutoML Run Details widget
![automl_run_details_widget](starter_file/screenshots/automl_run_details.png)

#### AutoML Best Model Run
![automl_run_web_gui](starter_file/screenshots/automl_run.png)

#### AutoML Best Model Run Properties
![automl_run_properties_1](starter_file/screenshots/automl_run_properties_1.png)
![automl_run_properties_2](starter_file/screenshots/automl_run_properties_2.png)

## Hyperparameter Tuning
Here we are using a Logistic Regression model coming from the SKLearn framework to classify if a passenger would survive or not in the Titanic shipwreck.
Although logistic regression assumes a linear relationship between input and output, which is rarely the real case, it is easy to implement, interpret, and very efficient to train and classify unknown records.
So, this algorithm has been chosen because it would allow us to experiment quickly in the Azure ML environment.

Hyperdrive is used to sample different values for two algorithm hyperparameters:
* `C`: Inverse of regularization strength
* `max_iter`: Maximum number of iterations taken for the solvers to converge

My choice here was to sample the values using Random Sampling, in which hyperparameter values are randomly selected from the defined search space. `C` is chosen randomly in uniformly distributed between **0.001** and **1.0**, while `max_iter` is sampled from one of the three values: **1000, 10000, and 100000**.

### Results
Surprisingly, the best Logistic Regression model in the HyperDrive run performed even better than the best one in the AutoML run. This model had an **accuracy** of **85,20%**.

The parameters used by this classifier are the following:
* C = 0.8893892118773127
* Max iterations = 1000

#### HyperDrive Run Details widget
![hyperdrive_run_details](starter_file/screenshots/hyperdrive_run_details.png)

#### HyperDrive Best Model Run
![hyperdrive_best_run_graph](starter_file/screenshots/hyperdrive_best_run_graph.png)

#### HyperDrive Best Model Run Properties
![hyperdrive_best_run_parameters](starter_file/screenshots/hyperdrive_best_run_parameters.png)

## Model Deployment
The model created by the HyperDrive has been deployed in an endpoint that can be accessed using the following REST API:
`http://XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX.southcentralus.azurecontainer.io/score`

The expected input type consists of a JSON with the following format:
```json
"data":
        [
          {
            "PassengerId": integer,
            "Pclass": integer,
            "Age": float,
            "SibSp": integer,
            "Parch": integer, 
            "Fare": float,
            "Q": integer,
            "S": integer,
            "male": integer
          }
        ]
```

To query it using a simple input, anyone can mock the following python code:

```python
import requests
import json

scoring_uri = 'http://01b44a8b-d762-47c0-af37-16bc6cdf52aa.southcentralus.azurecontainer.io/score'

data = {"data":
        [
          {
            "PassengerId": 812,
            "Pclass": 2,
            "Age": 23.0,
            "SibSp": 0,
            "Parch": 0, 
            "Fare": 13.0,
            "Q": 0,
            "S": 1,
            "male": 1
          }
        ]
       }

# Convert to JSON string
input_data = json.dumps(data)
# Set the content type
headers = {'Content-Type': 'application/json'}
# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```

#### Service of HyperDrive model with "Active" deployment state
![hyperdrive_service_active](starter_file/screenshots/hyperdrive_service_active.png)

## Future improvements

There are many ways to improve AutoML and HyperDrive runs in this project.

To improve the AutoML, we could choose the best 3 to 5 algorithms that performed well in this classification task and create another AutoML run forbidding any other algorithm type. We could also take a look at the data that has been wrongly classified by the best model and try to identify a pattern that points to transformations that we can perform in the dataset. That can be done by creating a pipeline with a first step to transform the data and a second one to execute the AutoML.

Moving on to the HyperDrive algorithm, we could have used regularization strength as a reference (it was randomly picked) and created a second HyperDrive run using a different sampling method using values closer to it.
Another strategy would be to test different classifier algorithms in our training script and change their hyperparameters too. We could do that to a finite set of algorithms and hyperparameters and select the best one among all runs. Many other classification algorithms could be tested, like Decision Tree, Random Forest, Support Vector Classification, and so on. Each of these algorithms has different hyperparameters that can be choose using either Random Sampling or other sampling methods. Deep Learning algorithms could also be applied to solve this problem.

Going even further, models performance was measured using the metric `Accuracy` for simplicity, and this could be changed to a more robust metric like `AUC_weighted` for example.
