from typing_extensions import Optional, Dict, Any, List
from langchain.tools import tool


@tool("train_h2o_automl", return_direct=True, response_format="content_and_artifact")
def train_h2o_automl(
    data_raw: List[Dict[str, Any]],
    target: str = "Churn",
    max_runtime_secs: int = 30,
    exclude_algos: List[str] = None,
    balance_classes: bool = True,
    nfolds: int = 5,
    seed: int = 42,
    max_models: int = 20,
    stopping_metric: str = "logloss",
    stopping_tolerance: float = 0.001,
    stopping_rounds: int = 3,
    sort_metric: str = "AUC",
    model_directory: Optional[str] = None,
    log_path: Optional[str] = None,
    enable_mlflow: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: str = "H2O AutoML",
    run_name: str = None,
    **kwargs,
) -> str:
    """
    A tool to train an H2O AutoML model on the provided data.
    Optionally logs results to MLflow if `enable_mlflow=True`.

    Parameters
    ----------
    data_raw : List[Dict[str, Any]]
        Row-wise data (like df.to_dict(orient="records")).
    target : str, default "Churn"
        The target column name.
    max_runtime_secs : int, default 30
    exclude_algos : List[str], optional
        e.g., ["DeepLearning"]. If not provided, defaults to ["DeepLearning"].
    balance_classes : bool, default True
    nfolds : int, default 5
    seed : int, default 42
    max_models : int, default 20
    stopping_metric : str, default "logloss"
    stopping_tolerance : float, default 0.001
    stopping_rounds : int, default 3
    sort_metric : str, default "AUC"
    model_directory : str or None
        Directory path to save the best model. If None, won't save unless log_path is set.
    log_path : str or None
        Fallback path if model_directory is None. If both are None, model won't be saved.
    enable_mlflow : bool, default False
        Whether to enable MLflow logging. If False, skip MLflow entirely.
    mlflow_tracking_uri : str or None
        If provided, sets MLflow tracking URI at runtime.
    mlflow_experiment_name : str
        Name of the MLflow experiment (created if doesn't exist).
    run_name : str, default "h2o_automl_run"
        A custom name for the MLflow run.
    **kwargs : dict
        Additional keyword arguments to pass to H2OAutoML().

    Returns
    -------
    str (JSON)
        {
          "leaderboard": dict of entire leaderboard,
          "best_model_id": str,
          "model_path": str or None,
          "model_results": {
             "model_flavor": "H2O AutoML",
             "model_path": str or None,
             "best_model_id": str,
             "metrics": dict
          },
          "mlflow_run_id": Optional[str]
        }
    """

    import h2o
    from h2o.automl import H2OAutoML
    import pandas as pd
    import json

    # Optional MLflow usage
    if enable_mlflow:
        import mlflow

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        run_context = mlflow.start_run(run_name=run_name)
    else:
        # Dummy context manager to skip MLflow if not enabled
        from contextlib import nullcontext

        run_context = nullcontext()

    exclude_algos = exclude_algos or ["DeepLearning"]  # default if not provided

    # Convert data to DataFrame
    df = pd.DataFrame(data_raw)

    with run_context as run:
        # If using MLflow, track run ID
        run_id = None
        if enable_mlflow and run is not None:
            run_id = run.info.run_id
            import mlflow

            # Log user-specified parameters
            mlflow.log_params(
                {
                    "target": target,
                    "max_runtime_secs": max_runtime_secs,
                    "exclude_algos": str(exclude_algos),
                    "balance_classes": balance_classes,
                    "nfolds": nfolds,
                    "seed": seed,
                    "max_models": max_models,
                    "stopping_metric": stopping_metric,
                    "stopping_tolerance": stopping_tolerance,
                    "stopping_rounds": stopping_rounds,
                    "sort_metric": sort_metric,
                    "model_directory": model_directory,
                    "log_path": log_path,
                    **kwargs,
                }
            )

        # Initialize H2O
        h2o.init()

        # Create H2OFrame
        data_h2o = h2o.H2OFrame(df)

        # Setup AutoML
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            exclude_algos=exclude_algos,
            balance_classes=balance_classes,
            nfolds=nfolds,
            seed=seed,
            max_models=max_models,
            stopping_metric=stopping_metric,
            stopping_tolerance=stopping_tolerance,
            stopping_rounds=stopping_rounds,
            sort_metric=sort_metric,
            **kwargs,
        )

        # Train
        x = [col for col in data_h2o.columns if col != target]
        aml.train(x=x, y=target, training_frame=data_h2o)

        # Save model if we have a directory/log path
        if model_directory is None and log_path is None:
            model_path = None
        else:
            path_to_save = model_directory if model_directory else log_path
            model_path = h2o.save_model(model=aml.leader, path=path_to_save, force=True)

        # Leaderboard (DataFrame -> dict)
        leaderboard_df = aml.leaderboard.as_data_frame()
        leaderboard_dict = leaderboard_df.to_dict()

        # Gather top-model metrics from the first row
        top_row = leaderboard_df.iloc[0].to_dict()  # includes model_id, etc.
        # Optionally remove model_id from metrics
        top_metrics = {k: v for k, v in top_row.items() if k.lower() != "model_id"}

        # Construct model_results
        model_results = {
            "model_flavor": "H2O AutoML",
            "model_path": model_path,
            "best_model_id": aml.leader.model_id,
            "metrics": top_metrics,  # all metrics from the top row
        }

        # If using MLflow, log the top metrics
        if enable_mlflow and run is not None:
            for metric_name, metric_value in top_metrics.items():
                # Only log floats/ints as metrics
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            # Log artifact if we saved the model
            if model_path is not None:
                mlflow.log_artifact(model_path, artifact_path="h2o_best_model")

        # Build the output
        output = {
            "leaderboard": leaderboard_dict,
            "best_model_id": aml.leader.model_id,
            "model_path": model_path,
            "model_results": model_results,
            "mlflow_run_id": run_id,
        }

    return json.dumps(output, indent=2)


H2O_AUTOML_DOCUMENTATION = """
Title: H2O AutoML: Automatic Machine Learning
Source: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

AutoML Interface
The H2O AutoML interface is designed to have as few parameters as possible so that all the user needs to do is point to their dataset, identify the response column and optionally specify a time constraint or limit on the number of total models trained. Below are the parameters that can be set by the user in the R and Python interfaces. See the Web UI via H2O Wave section below for information on how to use the H2O Wave web interface for AutoML.

In both the R and Python API, AutoML uses the same data-related arguments, x, y, training_frame, validation_frame, as the other H2O algorithms. Most of the time, all you'll need to do is specify the data arguments. You can then configure values for max_runtime_secs and/or max_models to set explicit time or number-of-model limits on your run.

Required Parameters
Required Data Parameters
y: This argument is the name (or index) of the response column.

training_frame: Specifies the training set.

Required Stopping Parameters
One of the following stopping strategies (time or number-of-model based) must be specified. When both options are set, then the AutoML run will stop as soon as it hits one of either When both options are set, then the AutoML run will stop as soon as it hits either of these limits.

max_runtime_secs: This argument specifies the maximum time that the AutoML process will run for. The default is 0 (no limit), but dynamically sets to 1 hour if none of max_runtime_secs and max_models are specified by the user.

max_models: Specify the maximum number of models to build in an AutoML run, excluding the Stacked Ensemble models. Defaults to NULL/None. Always set this parameter to ensure AutoML reproducibility: all models are then trained until convergence and none is constrained by a time budget.

Optional Parameters
Optional Data Parameters
x: A list/vector of predictor column names or indexes. This argument only needs to be specified if the user wants to exclude columns from the set of predictors. If all columns (other than the response) should be used in prediction, then this does not need to be set.

validation_frame: This argument is ignored unless nfolds == 0, in which a validation frame can be specified and used for early stopping of individual models and early stopping of the grid searches (unless max_models or max_runtime_secs overrides metric-based early stopping). By default and when nfolds > 1, cross-validation metrics will be used for early stopping and thus validation_frame will be ignored.

leaderboard_frame: This argument allows the user to specify a particular data frame to use to score and rank models on the leaderboard. This frame will not be used for anything besides leaderboard scoring. If a leaderboard frame is not specified by the user, then the leaderboard will use cross-validation metrics instead, or if cross-validation is turned off by setting nfolds = 0, then a leaderboard frame will be generated automatically from the training frame.

blending_frame: Specifies a frame to be used for computing the predictions that serve as the training frame for the Stacked Ensemble models metalearner. If provided, all Stacked Ensembles produced by AutoML will be trained using Blending (a.k.a. Holdout Stacking) instead of the default Stacking method based on cross-validation.

fold_column: Specifies a column with cross-validation fold index assignment per observation. This is used to override the default, randomized, 5-fold cross-validation scheme for individual models in the AutoML run.

weights_column: Specifies a column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative weights are not allowed.

Optional Miscellaneous Parameters
nfolds: Specify a value >= 2 for the number of folds for k-fold cross-validation of the models in the AutoML run or specify “-1” to let AutoML choose if k-fold cross-validation or blending mode should be used. Blending mode will use part of training_frame (if no blending_frame is provided) to train Stacked Ensembles. Use 0 to disable cross-validation; this will also disable Stacked Ensembles (thus decreasing the overall best model performance). This value defaults to “-1”.

balance_classes: Specify whether to oversample the minority classes to balance the class distribution. This option is not enabled by default and can increase the data frame size. This option is only applicable for classification. If the oversampled size of the dataset exceeds the maximum size calculated using the max_after_balance_size parameter, then the majority classes will be undersampled to satisfy the size limit.

class_sampling_factors: Specify the per-class (in lexicographical order) over/under-sampling ratios. By default, these ratios are automatically computed during training to obtain the class balance. Note that this requires balance_classes set to True.

max_after_balance_size: Specify the maximum relative size of the training data after balancing class counts (balance_classes must be enabled). Defaults to 5.0. (The value can be less than 1.0).

max_runtime_secs_per_model: Specify the max amount of time dedicated to the training of each individual model in the AutoML run. Defaults to 0 (disabled). Note that models constrained by a time budget are not guaranteed reproducible.

stopping_metric: Specify the metric to use for early stopping. Defaults to AUTO. The available options are:

- AUTO: This defaults to logloss for classification and deviance for regression.
- deviance (mean residual deviance)
- logloss
- MSE
- RMSE
- MAE
- RMSLE
- AUC (area under the ROC curve)
- AUCPR (area under the Precision-Recall curve)
- lift_top_group
- misclassification
- mean_per_class_error

stopping_tolerance: This option specifies the relative tolerance for the metric-based stopping criterion to stop a grid search and the training of individual models within the AutoML run. This value defaults to 0.001 if the dataset is at least 1 million rows; otherwise it defaults to a bigger value determined by the size of the dataset and the non-NA-rate. In that case, the value is computed as 1/sqrt(nrows * non-NA-rate).

stopping_rounds: This argument is used to stop model training when the stopping metric (e.g. AUC) doesn't improve for this specified number of training rounds, based on a simple moving average. In the context of AutoML, this controls early stopping both within the random grid searches as well as the individual models. Defaults to 3 and must be an non-negative integer. To disable early stopping altogether, set this to 0.

sort_metric: Specifies the metric used to sort the Leaderboard by at the end of an AutoML run. Available options include:

- AUTO: This defaults to AUC for binary classification, mean_per_class_error for multinomial classification, and deviance for regression.
- deviance (mean residual deviance)
- logloss
- MSE
- RMSE
- MAE
- RMSLE
- AUC (area under the ROC curve)
- AUCPR (area under the Precision-Recall curve)
- mean_per_class_error

seed: Integer. Set a seed for reproducibility. AutoML can only guarantee reproducibility under certain conditions. H2O Deep Learning models are not reproducible by default for performance reasons, so if the user requires reproducibility, then exclude_algos must contain "DeepLearning". In addition max_models must be used because max_runtime_secs is resource limited, meaning that if the available compute resources are not the same between runs, AutoML may be able to train more models on one run vs another. Defaults to NULL/None.

project_name: Character string to identify an AutoML project. Defaults to NULL/None, which means a project name will be auto-generated based on the training frame ID. More models can be trained and added to an existing AutoML project by specifying the same project name in multiple calls to the AutoML function (as long as the same training frame is used in subsequent runs).

exclude_algos: A list/vector of character strings naming the algorithms to skip during the model-building phase. An example use is exclude_algos = ["GLM", "DeepLearning", "DRF"] in Python or exclude_algos = c("GLM", "DeepLearning", "DRF") in R. Defaults to None/NULL, which means that all appropriate H2O algorithms will be used if the search stopping criteria allows and if the include_algos option is not specified. This option is mutually exclusive with include_algos. See include_algos below for the list of available options.

include_algos: A list/vector of character strings naming the algorithms to include during the model-building phase. An example use is include_algos = ["GLM", "DeepLearning", "DRF"] in Python or include_algos = c("GLM", "DeepLearning", "DRF") in R. Defaults to None/NULL, which means that all appropriate H2O algorithms will be used if the search stopping criteria allows and if no algorithms are specified in exclude_algos. This option is mutually exclusive with exclude_algos. The available algorithms are:

- DRF (This includes both the Distributed Random Forest (DRF) and Extremely Randomized Trees (XRT) models. Refer to the Extremely Randomized Trees section in the DRF chapter and the histogram_type parameter description for more information.)
- GLM (Generalized Linear Model with regularization)
- XGBoost (XGBoost GBM)
- GBM (H2O GBM)
- DeepLearning (Fully-connected multi-layer artificial neural network)
- StackedEnsemble (Stacked Ensembles, includes an ensemble of all the base models and ensembles using subsets of the base models)

modeling_plan: The list of modeling steps to be used by the AutoML engine. (They may not all get executed, depending on other constraints.)

preprocessing: The list of preprocessing steps to run. Only ["target_encoding"] is currently supported. There is more information about how Target Encoding is automatically applied here. Experimental.

exploitation_ratio: Specify the budget ratio (between 0 and 1) dedicated to the exploitation (vs exploration) phase. By default, the exploitation phase is disabled (exploitation_ratio=0) as this is still experimental; to activate it, it is recommended to try a ratio around 0.1. Note that the current exploitation phase only tries to fine-tune the best XGBoost and the best GBM found during exploration. Experimental.

monotone_constraints: A mapping that represents monotonic constraints. Use +1 to enforce an increasing constraint and -1 to specify a decreasing constraint.

keep_cross_validation_predictions: Specify whether to keep the predictions of the cross-validation predictions. This needs to be set to TRUE if running the same AutoML object for repeated runs because CV predictions are required to build additional Stacked Ensemble models in AutoML. This option defaults to FALSE.

keep_cross_validation_models: Specify whether to keep the cross-validated models. Keeping cross-validation models may consume significantly more memory in the H2O cluster. This option defaults to FALSE.

keep_cross_validation_fold_assignment: Enable this option to preserve the cross-validation fold assignment. Defaults to FALSE.

verbosity: (Optional: Python and R only) The verbosity of the backend messages printed during training. Must be one of "debug", "info", "warn". Defaults to NULL/None (client logging disabled).

export_checkpoints_dir: Specify a directory to which generated models will automatically be exported.

Notes
Validation Options
If the user turns off cross-validation by setting nfolds == 0, then cross-validation metrics will not be available to populate the leaderboard. In this case, we need to make sure there is a holdout frame (i.e. the “leaderboard frame”) to score the models on so that we can generate model performance metrics for the leaderboard. Without cross-validation, we will also require a validation frame to be used for early stopping on the models. Therefore, if either of these frames are not provided by the user, they will be automatically partitioned from the training data. If either frame is missing, 10% of the training data will be used to create a missing frame (if both are missing then a total of 20% of the training data will be used to create a 10% validation and 10% leaderboard frame).

XGBoost Memory Requirements
XGBoost, which is included in H2O as a third party library, requires its own memory outside the H2O (Java) cluster. When running AutoML with XGBoost (it is included by default), be sure you allow H2O no more than 2/3 of the total available RAM. Example: If you have 60G RAM, use h2o.init(max_mem_size = "40G"), leaving 20G for XGBoost.

Scikit-learn Compatibility
H2OAutoML can interact with the h2o.sklearn module. The h2o.sklearn module exposes 2 wrappers for H2OAutoML (H2OAutoMLClassifier and H2OAutoMLRegressor), which expose the standard API familiar to sklearn users: fit, predict, fit_predict, score, get_params, and set_params. It accepts various formats as input data (H2OFrame, numpy array, pandas Dataframe) which allows them to be combined with pure sklearn components in pipelines. For an example using H2OAutoML with the h2o.sklearn module, click here.

Explainability
AutoML objects are fully supported though the H2O Model Explainability interface. A large number of multi-model comparison and single model (AutoML leader) plots can be generated automatically with a single call to h2o.explain(). We invite you to learn more at page linked above.

Code Examples

Training
Here’s an example showing basic usage of the h2o.automl() function in R and the H2OAutoML class in Python. For demonstration purposes only, we explicitly specify the x argument, even though on this dataset, that’s not required. With this dataset, the set of predictors is all columns other than the response. Like other H2O algorithms, the default value of x is “all columns, excluding y”, so that will produce the same result.

``` python
import h2o
from h2o.automl import H2OAutoML

# Start the H2O cluster (locally)
h2o.init()

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/higgs/higgs_train_10k.csv")
test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/higgs/higgs_test_5k.csv")

# Identify predictors and response
x = train.columns
y = "response"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# model_id                                                  auc    logloss    mean_per_class_error      rmse       mse
# ---------------------------------------------------  --------  ---------  ----------------------  --------  --------
# StackedEnsemble_AllModels_AutoML_20181212_105540     0.789801   0.551109                0.333174  0.43211   0.186719
# StackedEnsemble_BestOfFamily_AutoML_20181212_105540  0.788425   0.552145                0.323192  0.432625  0.187165
# XGBoost_1_AutoML_20181212_105540                     0.784651   0.55753                 0.325471  0.434949  0.189181
# XGBoost_grid_1_AutoML_20181212_105540_model_4        0.783523   0.557854                0.318819  0.435249  0.189441
# XGBoost_grid_1_AutoML_20181212_105540_model_3        0.783004   0.559613                0.325081  0.435708  0.189841
# XGBoost_2_AutoML_20181212_105540                     0.78136    0.55888                 0.347074  0.435907  0.190015
# XGBoost_3_AutoML_20181212_105540                     0.780847   0.559589                0.330739  0.43613   0.190209
# GBM_5_AutoML_20181212_105540                         0.780837   0.559903                0.340848  0.436191  0.190263
# GBM_2_AutoML_20181212_105540                         0.780036   0.559806                0.339926  0.436415  0.190458
# GBM_1_AutoML_20181212_105540                         0.779827   0.560857                0.335096  0.436616  0.190633
# GBM_3_AutoML_20181212_105540                         0.778669   0.56179                 0.325538  0.437189  0.191134
# XGBoost_grid_1_AutoML_20181212_105540_model_2        0.774411   0.575017                0.322811  0.4427    0.195984
# GBM_4_AutoML_20181212_105540                         0.771426   0.569712                0.33742   0.44107   0.194543
# GBM_grid_1_AutoML_20181212_105540_model_1            0.769752   0.572583                0.344331  0.442452  0.195764
# GBM_grid_1_AutoML_20181212_105540_model_2            0.754366   0.918567                0.355855  0.496638  0.246649
# DRF_1_AutoML_20181212_105540                         0.742892   0.595883                0.355403  0.452774  0.205004
# XRT_1_AutoML_20181212_105540                         0.742091   0.599346                0.356583  0.453117  0.205315
# DeepLearning_grid_1_AutoML_20181212_105540_model_2   0.741795   0.601497                0.368291  0.454904  0.206937
# XGBoost_grid_1_AutoML_20181212_105540_model_1        0.693554   0.620702                0.40588   0.465791  0.216961
# DeepLearning_1_AutoML_20181212_105540                0.69137    0.637954                0.409351  0.47178   0.222576
# DeepLearning_grid_1_AutoML_20181212_105540_model_1   0.690084   0.661794                0.418469  0.476635  0.227181
# GLM_grid_1_AutoML_20181212_105540_model_1            0.682648   0.63852                 0.397234  0.472683  0.223429
#
# [22 rows x 6 columns]

# The leader model is stored here
aml.leader
```

Prediction
Using the predict() function with AutoML generates predictions on the leader model from the run. The order of the rows in the results is the same as the order in which the data was loaded, even if some rows fail (for example, due to missing values or unseen factor levels).

``` python
# To generate predictions on a test set, you can make predictions
# directly on the `H2OAutoML` object or on the leader model
# object directly
preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)
```

AutoML Output

Leaderboard
The AutoML object includes a “leaderboard” of models that were trained in the process, including the 5-fold cross-validated model performance (by default). The number of folds used in the model evaluation process can be adjusted using the nfolds parameter. If you would like to score the models on a specific dataset, you can specify the leaderboard_frame argument in the AutoML run, and then the leaderboard will show scores on that dataset instead.

The models are ranked by a default metric based on the problem type (the second column of the leaderboard). In binary classification problems, that metric is AUC, and in multiclass classification problems, the metric is mean per-class error. In regression problems, the default sort metric is RMSE. Some additional metrics are also provided, for convenience.

To help users assess the complexity of AutoML models, the h2o.get_leaderboard function has been been expanded by allowing an extra_columns parameter. This parameter allows you to specify which (if any) optional columns should be added to the leaderboard. This defaults to None. Allowed options include:

- training_time_ms: A column providing the training time of each model in milliseconds. (Note that this doesn't include the training of cross validation models.)

- predict_time_per_row_ms: A column providing the average prediction time by the model for a single row.

- ALL: Adds columns for both training_time_ms and predict_time_per_row_ms.

``` python
# Get leaderboard with all possible columns
lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
lb
```

Examine Models
To examine the trained models more closely, you can interact with the models, either by model ID, or a convenience function which can grab the best model of each model type (ranked by the default metric, or a metric of your choosing).

``` python
# Get the best model using the metric
m = aml.leader
# this is equivalent to
m = aml.get_best_model()

# Get the best model using a non-default metric
m = aml.get_best_model(criterion="logloss")

# Get the best XGBoost model using default sort metric
xgb = aml.get_best_model(algorithm="xgboost")

# Get the best XGBoost model, ranked by logloss
xgb = aml.get_best_model(algorithm="xgboost", criterion="logloss")
```

Get a specific model by model ID:

``` python
# Get a specific model by model ID
m = h2o.get_model("StackedEnsemble_BestOfFamily_AutoML_20191213_174603")
```

Once you have retreived the model in R or Python, you can inspect the model parameters as follows:

``` python
# View the parameters for the XGBoost model selected above
xgb.params.keys()

# Inspect individual parameter values
xgb.params['ntrees']
```

AutoML Log
When using Python or R clients, you can also access meta information with the following AutoML object properties:

- event_log: an H2OFrame with selected AutoML backend events generated during training.

- training_info: a dictionary exposing data that could be useful for post-analysis (e.g. various timings). If you want training and prediction times for each model, it's easier to explore that data in the extended leaderboard using the h2o.get_leaderboard() function.

``` python
# Get AutoML event log
log = aml.event_log

# Get training timing info
info = aml.training_info
```

Experimental Features

Preprocessing
As of H2O 3.32.0.1, AutoML now has a preprocessing option with minimal support for automated Target Encoding of high cardinality categorical variables. The only currently supported option is preprocessing = ["target_encoding"]: we automatically tune a Target Encoder model and apply it to columns that meet certain cardinality requirements for the tree-based algorithms (XGBoost, H2O GBM and Random Forest). 

FAQ

1. Which models are trained in the AutoML process?

The current version of AutoML trains and cross-validates the following algorithms: three pre-specified XGBoost GBM (Gradient Boosting Machine) models, a fixed grid of GLMs, a default Random Forest (DRF), five pre-specified H2O GBMs, a near-default Deep Neural Net, an Extremely Randomized Forest (XRT), a random grid of XGBoost GBMs, a random grid of H2O GBMs, and a random grid of Deep Neural Nets. In some cases, there will not be enough time to complete all the algorithms, so some may be missing from the leaderboard. In other cases, the grids will stop early, and if there's time left, the top two random grids will be restarted to train more models. AutoML trains multiple Stacked Ensemble models throughout the process (more info about the ensembles below).

Particular algorithms (or groups of algorithms) can be switched off using the exclude_algos argument. This is useful if you already have some idea of the algorithms that will do well on your dataset, though sometimes this can lead to a loss of performance because having more diversity among the set of models generally increases the performance of the Stacked Ensembles. As a first step you could leave all the algorithms on, and examine their performance characteristics (e.g. prediction speed) to get a sense of what might be practically useful in your specific use-case, and then turn off algorithms that are not interesting or useful to you. We recommend using the H2O Model Explainability interface to explore and further evaluate your AutoML models, which can inform your choice of model (if you have other goals beyond simply maximizing model accuracy).

A list of the hyperparameters searched over for each algorithm in the AutoML process is included in the appendix below. More details about the hyperparameter ranges for the models in addition to the hard-coded models will be added to the appendix at a later date.

AutoML trains several Stacked Ensemble models during the run (unless ensembles are turned off using exclude_algos). We have subdivided the model training in AutoML into “model groups” with different priority levels. After each group is completed, and at the very end of the AutoML process, we train (at most) two additional Stacked Ensembles with the existing models. There are currently two types of Stacked Ensembles: one which includes all the base models (“All Models”), and one comprised only of the best model from each algorithm family (“Best of Family”). The Best of Family ensembles are more optimized for production use since it only contains six (or fewer) base models. It should be relatively fast to use in production (to generate predictions on new data) without much degradation in model performance when compared to the final “All Models” ensemble, for example. This may be useful if you want the model performance boost from ensembling without the added time or complexity of a large ensemble. You can also inspect some of the earlier “All Models” Stacked Ensembles that have fewer models as an alternative to the Best of Family ensembles. The metalearner used in all ensembles is a variant of the default Stacked Ensemble metalearner: a non-negative GLM with regularization (Lasso or Elastic net, chosen by CV) to encourage more sparse ensembles. The metalearner also uses a logit transform (on the base learner CV preds) for classification tasks before training.

For information about how previous versions of AutoML were different than the current one, there's a brief description here.

2. How do I save AutoML runs?

Rather than saving an AutoML object itself, currently, the best thing to do is to save the models you want to keep, individually. A utility for saving all of the models at once, along with a way to save the AutoML object (with leaderboard), will be added in a future release.

3. Can we make use of GPUs with AutoML?

XGBoost models in AutoML can make use of GPUs. Keep in mind that the following requirements must be met:

- NVIDIA GPUs (GPU Cloud, DGX Station, DGX-1, or DGX-2)
- CUDA 8

You can monitor your GPU utilization via the nvidia-smi command. Refer to https://developer.nvidia.com/nvidia-system-management-interface for more information.

4. Why don't I see XGBoost models?

AutoML includes XGBoost GBMs (Gradient Boosting Machines) among its set of algorithms. This feature is currently provided with the following restrictions:

- XGBoost is not currently available on Windows machines. Follow here: https://github.com/h2oai/h2o-3/issues/7139 for updates.

- XGBoost is used only if it is available globally and if it hasn't been explicitly disabled. You can check if XGBoost is available by using the h2o.xgboost.available() in R or h2o.estimators.xgboost.H2OXGBoostEstimator.available() in Python.

5. Why doesn't AutoML use all the time that it's given?

If you're using 3.34.0.1 or later, AutoML should use all the time that it's given using max_runtime_secs. However, if you're using an earlier version, then early stopping was enabled by default and you can stop early. With early stopping, AutoML will stop once there's no longer “enough” incremental improvement. The user can tweak the early stopping paramters to be more or less sensitive. Set stopping_rounds higher if you want to slow down early stopping and let AutoML train more models before it stops.

6. Does AutoML support MOJOs?

AutoML will always produce a model which has a MOJO. Though it depends on the run, you are most likely to get a Stacked Ensemble. While all models are importable, only individual models are exportable.

7. Why doesn't AutoML use all the time that it's given?

If you're using 3.34.0.1 or later, AutoML should use all the time that it's given using max_runtime_secs. However, if you're using an earlier version, then early stopping was enabled by default and you can stop early. With early stopping, AutoML will stop once there's no longer “enough” incremental improvement. The user can tweak the early stopping paramters to be more or less sensitive. Set stopping_rounds higher if you want to slow down early stopping and let AutoML train more models before it stops.

8. What is the history of H2O AutoML?

The H2O AutoML algorithm was first released in H2O 3.12.0.1 on June 6, 2017 by Erin LeDell, and is based on research from her PhD thesis. New features and performance improvements have been made in every major version of H2O since the initial release.

"""

# @tool("get_h2o_leaderboard", return_direct=True)
# def get_h2o_leaderboard(h2o_agent) -> str:
#     """
#     Retrieve the current H2O AutoML leaderboard from the given H2OMLAgent.

#     Parameters
#     ----------
#     h2o_agent : H2OMLAgent
#         An instance of your H2OMLAgent that has run H2O AutoML.

#     Returns
#     -------
#     str
#         A stringified JSON of the H2O AutoML leaderboard.
#         (Use JSON or CSV format as desired for your tooling.)
#     """
#     leaderboard_df = h2o_agent.get_leaderboard()
#     if leaderboard_df is None:
#         return "No leaderboard found. Make sure the agent has been run."
#     return leaderboard_df.to_json(orient="records")


# @tool("get_h2o_best_model_id", return_direct=True)
# def get_h2o_best_model_id(h2o_agent) -> str:
#     """
#     Retrieve the best model ID from the H2OMLAgent.

#     Parameters
#     ----------
#     h2o_agent : H2OMLAgent

#     Returns
#     -------
#     str
#         The best model identifier (e.g., "StackedEnsemble_BestOfFamily_...").
#     """
#     best_id = h2o_agent.get_best_model_id()
#     return best_id if best_id else "No best model found."


# @tool("get_h2o_best_model_path", return_direct=True)
# def get_h2o_best_model_path(h2o_agent) -> str:
#     """
#     Retrieve the file path of the best model saved by the H2OMLAgent.

#     Parameters
#     ----------
#     h2o_agent : H2OMLAgent

#     Returns
#     -------
#     str
#         The path where the best model is saved, or 'None' if not saved.
#     """
#     path = h2o_agent.get_model_path()
#     return path if path else "No model path found."


# @tool("predict_with_h2o_model", return_direct=True)
# def predict_with_h2o_model(
#     h2o_agent,
#     data: List[Dict[str, Any]],
#     model_id_or_path: Optional[str] = None
# ) -> str:
#     """
#     Predict on new data using a model from H2O. You can specify either:
#       - model_id_or_path as a model ID (if it's in the H2O cluster), or
#       - a local file path to a saved H2O model (if you have it).

#     Parameters
#     ----------
#     h2o_agent : H2OMLAgent
#         Instance of H2OMLAgent, to facilitate H2O usage.
#     data : List[Dict[str, Any]]
#         The data to predict on, in a list-of-rows (dictionary) format.
#     model_id_or_path : str, optional
#         Either the H2O model ID or the local path where the model is saved.

#     Returns
#     -------
#     str
#         A stringified JSON of the prediction results.
#     """
#     import h2o
#     import pandas as pd

#     # Convert the data to a pandas DataFrame
#     df = pd.DataFrame(data)
#     # Convert to H2OFrame
#     h2o_frame = h2o.H2OFrame(df)

#     # If model_id_or_path is an H2O model ID in the cluster:
#     try:
#         # Attempt to load it as a model ID from the cluster
#         model = h2o.get_model(model_id_or_path)
#     except Exception:
#         model = None

#     # If that fails, assume it's a local path
#     if model is None and model_id_or_path is not None:
#         model = h2o.load_model(model_id_or_path)

#     if model is None:
#         # As a fallback, default to the best model in the agent if no ID/path was given
#         best_id = h2o_agent.get_best_model_id()
#         if best_id:
#             model = h2o.get_model(best_id)
#         else:
#             return "No valid model_id_or_path found, and agent has no best model."

#     preds = model.predict(h2o_frame).as_data_frame()
#     return preds.to_json(orient="records")
