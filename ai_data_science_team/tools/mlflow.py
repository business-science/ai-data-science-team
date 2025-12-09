from typing_extensions import Optional, Union, List, Annotated
from langgraph.prebuilt import InjectedState
from langchain.tools import tool
import psutil


@tool(response_format="content_and_artifact")
def mlflow_search_experiments(
    filter_string: Optional[str] = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> str:
    """
    Search and list existing MLflow experiments.

    Parameters
    ----------
    filter_string : str, optional
        Filter query string (e.g., "name = 'my_experiment'"), defaults to
        searching for all experiments.

    tracking_uri: str, optional
        Address of local or remote tracking server.
        If not provided, defaults
        to the service set by mlflow.tracking.set_tracking_uri. See Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>_ for more info.
    registry_uri: str, optional
        Address of local or remote model registry
        server. If not provided,
        defaults to the service set by mlflow.tracking.set_registry_uri. If no such service was set, defaults to the tracking uri of the client.

    Returns
    -------
    tuple
        - JSON-serialized list of experiment metadata (ID, name, etc.).
        - DataFrame of experiment metadata.
    """
    print("    * Tool: mlflow_search_experiments")
    from mlflow.tracking import MlflowClient
    import pandas as pd

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    experiments = client.search_experiments(filter_string=filter_string)
    # Convert to a dictionary in a list
    experiments_data = [dict(e) for e in experiments]
    # Convert to a DataFrame
    experiments_df = pd.DataFrame(experiments_data)
    # Convert timestamps to datetime objects
    experiments_df["last_update_time"] = pd.to_datetime(
        experiments_df["last_update_time"], unit="ms"
    )
    experiments_df["creation_time"] = pd.to_datetime(
        experiments_df["creation_time"], unit="ms"
    )

    return (experiments_df.to_dict(), experiments_df.to_dict())


@tool(response_format="content_and_artifact")
def mlflow_search_runs(
    experiment_ids: Optional[Union[List[str], List[int], str, int]] = None,
    filter_string: Optional[str] = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> str:
    """
    Search runs within one or more MLflow experiments, optionally filtering by a filter_string.

    Parameters
    ----------
    experiment_ids : list or str or int, optional
        One or more Experiment IDs.
    filter_string : str, optional
        MLflow filter expression, e.g. "metrics.rmse < 1.0".
    tracking_uri: str, optional
        Address of local or remote tracking server.
        If not provided, defaults
        to the service set by mlflow.tracking.set_tracking_uri. See Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>_ for more info.
    registry_uri: str, optional
        Address of local or remote model registry
        server. If not provided,
        defaults to the service set by mlflow.tracking.set_registry_uri. If no such service was set, defaults to the tracking uri of the client.

    Returns
    -------
    str
        JSON-formatted list of runs that match the query.
    """
    print("    * Tool: mlflow_search_runs")
    from mlflow.tracking import MlflowClient
    import pandas as pd

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    if experiment_ids is None:
        experiment_ids = []
    if isinstance(experiment_ids, (str, int)):
        experiment_ids = [experiment_ids]

    runs = client.search_runs(
        experiment_ids=experiment_ids, filter_string=filter_string
    )

    # If no runs are found, return an empty DataFrame
    if not runs:
        return "No runs found.", pd.DataFrame()

    # Extract relevant information
    data = []
    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "end_time": pd.to_datetime(run.info.end_time, unit="ms"),
            "experiment_id": run.info.experiment_id,
            "user_id": run.info.user_id,
        }

        # Flatten metrics, parameters, and tags
        run_info.update(run.data.metrics)
        run_info.update({f"param_{k}": v for k, v in run.data.params.items()})
        run_info.update({f"tag_{k}": v for k, v in run.data.tags.items()})

        data.append(run_info)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return (df.iloc[:, 0:15].to_dict(), df.to_dict())


@tool(response_format="content")
def mlflow_create_experiment(experiment_name: str) -> str:
    """
    Create a new MLflow experiment by name.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to create.

    Returns
    -------
    str
        The experiment ID or an error message if creation failed.
    """
    print("    * Tool: mlflow_create_experiment")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp_id = client.create_experiment(experiment_name)
    return f"Experiment created with ID: {exp_id}, name: {experiment_name}"


@tool(response_format="content_and_artifact")
def mlflow_predict_from_run_id(
    run_id: str,
    data_raw: Annotated[dict, InjectedState("data_raw")],
    tracking_uri: Optional[str] = None,
) -> tuple:
    """
    Predict using an MLflow model (PyFunc) directly from a given run ID.

    Parameters
    ----------
    run_id : str
        The ID of the MLflow run that logged the model.
    data_raw : dict
        The incoming data as a dictionary.
    tracking_uri : str, optional
        Address of local or remote tracking server.

    Returns
    -------
    tuple
        (user_facing_message, artifact_dict)
    """
    print("    * Tool: mlflow_predict_from_run_id")
    import mlflow
    import mlflow.pyfunc
    import pandas as pd

    # 1. Check if data is loaded
    if not data_raw:
        return (
            "No data provided for prediction. Please use `data_raw` parameter inside of `invoke_agent()` or `ainvoke_agent()`.",
            {},
        )
    df = pd.DataFrame(data_raw)

    # 2. Prepare model URI
    model_uri = f"runs:/{run_id}/model"

    # 3. Load or cache the MLflow model
    model = mlflow.pyfunc.load_model(model_uri)

    # 4. Make predictions
    try:
        preds = model.predict(df)
    except Exception as e:
        return f"Error during inference: {str(e)}", {}

    # 5. Convert predictions to a user-friendly summary + artifact
    if isinstance(preds, pd.DataFrame):
        sample_json = preds.head().to_json(orient="records")
        artifact_dict = preds.to_dict(orient="records")  # entire DF
        message = f"Predictions returned. Sample: {sample_json}"
    elif hasattr(preds, "to_json"):
        # e.g., pd.Series
        sample_json = preds[:5].to_json(orient="records")
        artifact_dict = preds.to_dict()
        message = f"Predictions returned. Sample: {sample_json}"
    elif hasattr(preds, "tolist"):
        # e.g., a NumPy array
        preds_list = preds.tolist()
        artifact_dict = {"predictions": preds_list}
        message = f"Predictions returned. First 5: {preds_list[:5]}"
    else:
        # fallback
        preds_str = str(preds)
        artifact_dict = {"predictions": preds_str}
        message = (
            f"Predictions returned (unrecognized type). Example: {preds_str[:100]}..."
        )

    return (message, artifact_dict)


# MLflow tool to launch gui for mlflow
@tool(response_format="content")
def mlflow_launch_ui(
    port: int = 5000, host: str = "localhost", tracking_uri: Optional[str] = None
) -> str:
    """
    Launch the MLflow UI.

    Parameters
    ----------
    port : int, optional
        The port on which to run the UI.
    host : str, optional
        The host address to bind the UI to.
    tracking_uri : str, optional
        Address of local or remote tracking server.

    Returns
    -------
    str
        Confirmation message.
    """
    print("    * Tool: mlflow_launch_ui")
    import subprocess

    # Try binding to the user-specified port first
    allocated_port = _find_free_port(start_port=port, host=host)

    cmd = ["mlflow", "ui", "--host", host, "--port", str(allocated_port)]
    if tracking_uri:
        cmd.extend(["--backend-store-uri", tracking_uri])

    process = subprocess.Popen(cmd)
    return f"MLflow UI launched at http://{host}:{allocated_port}. (PID: {process.pid})"


def _find_free_port(start_port: int, host: str) -> int:
    """
    Find a free port >= start_port on the specified host.
    If the start_port is free, returns start_port, else tries subsequent ports.
    """
    import socket

    for port_candidate in range(start_port, start_port + 1000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port_candidate))
            except OSError:
                # Port is in use, try the next one
                continue
            # If bind succeeds, it's free
            return port_candidate

    raise OSError(
        f"No available ports found in the range {start_port}-{start_port + 999}"
    )


@tool(response_format="content")
def mlflow_stop_ui(port: int = 5000) -> str:
    """
    Kill any process currently listening on the given MLflow UI port.
    Requires `pip install psutil`.

    Parameters
    ----------
    port : int, optional
        The port on which the UI is running.
    """
    print("    * Tool: mlflow_stop_ui")
    import psutil

    # Attempt to find processes listening on port; on macOS this may require elevated perms.
    try:
        conns = psutil.net_connections(kind="inet")
    except psutil.AccessDenied:
        return (
            "Unable to enumerate network connections (permission denied). "
            "Try running with elevated permissions or stop the MLflow UI manually."
        )
    except Exception as e:
        return f"Failed to inspect network connections: {e}"

    for conn in conns:
        if conn.laddr and conn.laddr.port == port and conn.pid is not None:
            try:
                p = psutil.Process(conn.pid)
                p_name = p.name()
                p.kill()
                return f"Killed process {conn.pid} ({p_name}) listening on port {port}."
            except psutil.NoSuchProcess:
                return "Process was already terminated before we could kill it."
            except psutil.AccessDenied:
                return (
                    f"Process {conn.pid} is listening on port {port} but cannot be killed "
                    "due to insufficient permissions."
                )
            except Exception as e:
                return f"Failed to kill process {conn.pid} on port {port}: {e}"

    return f"No process found listening on port {port}."


@tool(response_format="content_and_artifact")
def mlflow_list_artifacts(
    run_id: str, path: Optional[str] = None, tracking_uri: Optional[str] = None
) -> tuple:
    """
    List artifacts under a given MLflow run.

    Parameters
    ----------
    run_id : str
        The ID of the run whose artifacts to list.
    path : str, optional
        Path within the run's artifact directory to list. Defaults to the root.
    tracking_uri : str, optional
        Custom tracking server URI.

    Returns
    -------
    tuple
        (summary_message, artifact_listing)
    """
    print("    * Tool: mlflow_list_artifacts")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    # If path is None, list the root folder
    artifact_list = client.list_artifacts(run_id, path or "")

    # Convert to a more user-friendly structure
    artifacts_data = []
    for artifact in artifact_list:
        artifacts_data.append(
            {
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size,
            }
        )

    return (f"Found {len(artifacts_data)} artifacts.", artifacts_data)


@tool(response_format="content_and_artifact")
def mlflow_download_artifacts(
    run_id: str,
    path: Optional[str] = None,
    dst_path: Optional[str] = "./downloaded_artifacts",
    tracking_uri: Optional[str] = None,
) -> tuple:
    """
    Download artifacts from MLflow to a local directory.

    Parameters
    ----------
    run_id : str
        The ID of the run whose artifacts to download.
    path : str, optional
        Path within the run's artifact directory to download. Defaults to the root.
    dst_path : str, optional
        Local destination path to store artifacts.
    tracking_uri : str, optional
        MLflow tracking server URI.

    Returns
    -------
    tuple
        (summary_message, artifact_dict)
    """
    print("    * Tool: mlflow_download_artifacts")
    from mlflow.tracking import MlflowClient
    import os

    client = MlflowClient(tracking_uri=tracking_uri)
    local_path = client.download_artifacts(run_id, path or "", dst_path)

    # Build a recursive listing of what was downloaded
    downloaded_files = []
    for root, dirs, files in os.walk(local_path):
        for f in files:
            downloaded_files.append(os.path.join(root, f))

    message = (
        f"Artifacts for run_id='{run_id}' have been downloaded to: {local_path}. "
        f"Total files: {len(downloaded_files)}."
    )

    return (message, {"downloaded_files": downloaded_files})


@tool(response_format="content_and_artifact")
def mlflow_list_registered_models(
    max_results: int = 100,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
) -> tuple:
    """
    List all registered models in MLflow's model registry.

    Parameters
    ----------
    max_results : int, optional
        Maximum number of models to return.
    tracking_uri : str, optional
    registry_uri : str, optional

    Returns
    -------
    tuple
        (summary_message, model_list)
    """
    print("    * Tool: mlflow_list_registered_models")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    # The list_registered_models() call can be paginated; for simplicity, we just pass max_results
    models = client.list_registered_models(max_results=max_results)

    models_data = []
    for m in models:
        models_data.append(
            {
                "name": m.name,
                "latest_versions": [
                    {
                        "version": v.version,
                        "run_id": v.run_id,
                        "current_stage": v.current_stage,
                    }
                    for v in m.latest_versions
                ],
            }
        )

    return (f"Found {len(models_data)} registered models.", models_data)


@tool(response_format="content_and_artifact")
def mlflow_search_registered_models(
    filter_string: Optional[str] = None,
    order_by: Optional[List[str]] = None,
    max_results: int = 100,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
) -> tuple:
    """
    Search registered models in MLflow's registry using optional filters.

    Parameters
    ----------
    filter_string : str, optional
        e.g. "name LIKE 'my_model%'" or "tags.stage = 'production'".
    order_by : list, optional
        e.g. ["name ASC"] or ["timestamp DESC"].
    max_results : int, optional
        Max number of results.
    tracking_uri : str, optional
    registry_uri : str, optional

    Returns
    -------
    tuple
        (summary_message, model_dict_list)
    """
    print("    * Tool: mlflow_search_registered_models")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    models = client.search_registered_models(
        filter_string=filter_string, order_by=order_by, max_results=max_results
    )

    models_data = []
    for m in models:
        models_data.append(
            {
                "name": m.name,
                "description": m.description,
                "creation_timestamp": m.creation_timestamp,
                "last_updated_timestamp": m.last_updated_timestamp,
                "latest_versions": [
                    {
                        "version": v.version,
                        "run_id": v.run_id,
                        "current_stage": v.current_stage,
                    }
                    for v in m.latest_versions
                ],
            }
        )

    return (
        f"Found {len(models_data)} models matching filter={filter_string}.",
        models_data,
    )


@tool(response_format="content_and_artifact")
def mlflow_get_model_version_details(
    name: str,
    version: str,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
) -> tuple:
    """
    Retrieve details about a specific model version in the MLflow registry.

    Parameters
    ----------
    name : str
        Name of the registered model.
    version : str
        Version number of that model.
    tracking_uri : str, optional
    registry_uri : str, optional

    Returns
    -------
    tuple
        (summary_message, version_data_dict)
    """
    print("    * Tool: mlflow_get_model_version_details")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    version_details = client.get_model_version(name, version)

    data = {
        "name": version_details.name,
        "version": version_details.version,
        "run_id": version_details.run_id,
        "creation_timestamp": version_details.creation_timestamp,
        "current_stage": version_details.current_stage,
        "description": version_details.description,
        "status": version_details.status,
    }

    return (f"Model version details retrieved for {name} v{version}", data)


@tool(response_format="content_and_artifact")
def mlflow_get_run_details(
    run_id: str,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
) -> tuple:
    """
    Retrieve run info, params, metrics, tags, and a shallow artifact listing.
    """
    print("    * Tool: mlflow_get_run_details")
    from mlflow.tracking import MlflowClient
    import pandas as pd

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    run = client.get_run(run_id)
    info = run.info
    data = run.data

    # Shallow artifact listing at root
    artifacts = client.list_artifacts(run_id, "")
    artifacts_data = [
        {"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size}
        for a in artifacts
    ]

    flattened = {
        "run_id": info.run_id,
        "run_name": info.run_name,
        "status": info.status,
        "start_time": pd.to_datetime(info.start_time, unit="ms"),
        "end_time": pd.to_datetime(info.end_time, unit="ms") if info.end_time else None,
        "experiment_id": info.experiment_id,
        "user_id": info.user_id,
        "artifact_uri": info.artifact_uri,
        "metrics": data.metrics,
        "params": data.params,
        "tags": data.tags,
        "artifacts": artifacts_data,
    }
    return (f"Details retrieved for run_id='{run_id}'.", flattened)


@tool(response_format="content")
def mlflow_transition_model_version_stage(
    name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool = False,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
) -> str:
    """
    Transition a registered model version to a new stage (e.g., Staging, Production, Archived).
    """
    print("    * Tool: mlflow_transition_model_version_stage")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )
    return (
        f"Model '{name}' version '{version}' transitioned to stage '{stage}'. "
        f"archive_existing_versions={archive_existing_versions}"
    )


@tool(response_format="content_and_artifact")
def mlflow_tracking_info() -> tuple:
    """
    Return current tracking URI, registry URI, and active run info (if any).
    """
    print("    * Tool: mlflow_tracking_info")
    import mlflow

    tracking_uri = mlflow.get_tracking_uri()
    registry_uri = mlflow.get_registry_uri()
    active = mlflow.active_run()
    active_info = None
    if active:
        active_info = {
            "run_id": active.info.run_id,
            "experiment_id": active.info.experiment_id,
            "artifact_uri": active.info.artifact_uri,
            "status": active.info.status,
        }
    msg = "MLflow tracking info retrieved."
    artifact = {
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
        "active_run": active_info,
    }
    return msg, artifact


@tool(response_format="content_and_artifact")
def mlflow_ui_status(port: int = 5000) -> tuple:
    """
    Check if a process appears to be serving MLflow UI on the given port.
    """
    print("    * Tool: mlflow_ui_status")
    ui_procs = []
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if any("mlflow" in part for part in cmdline) and "ui" in cmdline:
                    ui_procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        ui_procs = []

    listening = []
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid is not None:
                listening.append(conn.pid)
    except Exception:
        listening = []

    running = any(p["pid"] in listening for p in ui_procs) if ui_procs else bool(listening)
    msg = (
        f"MLflow UI {'appears to be running' if running else 'not detected'} "
        f"on port {port}."
    )
    return msg, {"ui_processes": ui_procs, "listening_pids_on_port": listening}


# @tool
# def get_or_create_experiment(experiment_name):
#     """
#     Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

#     This function checks if an experiment with the given name exists within MLflow.
#     If it does, the function returns its ID. If not, it creates a new experiment
#     with the provided name and returns its ID.

#     Parameters:
#     - experiment_name (str): Name of the MLflow experiment.

#     Returns:
#     - str: ID of the existing or newly created MLflow experiment.
#     """
#     import mlflow
#     if experiment := mlflow.get_experiment_by_name(experiment_name):
#         return experiment.experiment_id
#     else:
#         return mlflow.create_experiment(experiment_name)


# @tool("mlflow_set_tracking_uri", return_direct=True)
# def mlflow_set_tracking_uri(tracking_uri: str) -> str:
#     """
#     Set or change the MLflow tracking URI.

#     Parameters
#     ----------
#     tracking_uri : str
#         The URI/path where MLflow logs & metrics are stored.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.set_tracking_uri(tracking_uri)
#     return f"MLflow tracking URI set to: {tracking_uri}"


# @tool("mlflow_list_experiments", return_direct=True)
# def mlflow_list_experiments() -> str:
#     """
#     List existing MLflow experiments.

#     Returns
#     -------
#     str
#         JSON-serialized list of experiment metadata (ID, name, etc.).
#     """
#     from mlflow.tracking import MlflowClient
#     import json

#     client = MlflowClient()
#     experiments = client.list_experiments()
#     # Convert to a JSON-like structure
#     experiments_data = [
#         dict(experiment_id=e.experiment_id, name=e.name, artifact_location=e.artifact_location)
#         for e in experiments
#     ]

#     return json.dumps(experiments_data)


# @tool("mlflow_create_experiment", return_direct=True)
# def mlflow_create_experiment(experiment_name: str) -> str:
#     """
#     Create a new MLflow experiment by name.

#     Parameters
#     ----------
#     experiment_name : str
#         The name of the experiment to create.

#     Returns
#     -------
#     str
#         The experiment ID or an error message if creation failed.
#     """
#     from mlflow.tracking import MlflowClient

#     client = MlflowClient()
#     exp_id = client.create_experiment(experiment_name)
#     return f"Experiment created with ID: {exp_id}"


# @tool("mlflow_set_experiment", return_direct=True)
# def mlflow_set_experiment(experiment_name: str) -> str:
#     """
#     Set or create an MLflow experiment for subsequent logging.

#     Parameters
#     ----------
#     experiment_name : str
#         The name of the experiment to set.

#     Returns
#     -------
#     str
#         Confirmation of the chosen experiment name.
#     """
#     import mlflow
#     mlflow.set_experiment(experiment_name)
#     return f"Active MLflow experiment set to: {experiment_name}"


# @tool("mlflow_start_run", return_direct=True)
# def mlflow_start_run(run_name: Optional[str] = None) -> str:
#     """
#     Start a new MLflow run under the current experiment.

#     Parameters
#     ----------
#     run_name : str, optional
#         Optional run name.

#     Returns
#     -------
#     str
#         The run_id of the newly started MLflow run.
#     """
#     import mlflow
#     with mlflow.start_run(run_name=run_name) as run:
#         run_id = run.info.run_id
#     return f"MLflow run started with run_id: {run_id}"


# @tool("mlflow_log_params", return_direct=True)
# def mlflow_log_params(params: Dict[str, Any]) -> str:
#     """
#     Log a batch of parameters to the current MLflow run.

#     Parameters
#     ----------
#     params : dict
#         A dictionary of parameter name -> parameter value.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.log_params(params)
#     return f"Logged parameters: {params}"


# @tool("mlflow_log_metrics", return_direct=True)
# def mlflow_log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> str:
#     """
#     Log a dictionary of metrics to the current MLflow run.

#     Parameters
#     ----------
#     metrics : dict
#         Metric name -> numeric value.
#     step : int, optional
#         The training step or iteration number.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.log_metrics(metrics, step=step)
#     return f"Logged metrics: {metrics} at step {step}"


# @tool("mlflow_log_artifact", return_direct=True)
# def mlflow_log_artifact(artifact_path: str, artifact_folder_name: Optional[str] = None) -> str:
#     """
#     Log a local file or directory as an MLflow artifact.

#     Parameters
#     ----------
#     artifact_path : str
#         The local path to the file/directory to be logged.
#     artifact_folder_name : str, optional
#         Subfolder within the run's artifact directory.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     if artifact_folder_name:
#         mlflow.log_artifact(artifact_path, artifact_folder_name)
#         return f"Artifact logged from {artifact_path} into folder '{artifact_folder_name}'"
#     else:
#         mlflow.log_artifact(artifact_path)
#         return f"Artifact logged from {artifact_path}"


# @tool("mlflow_log_model", return_direct=True)
# def mlflow_log_model(model_path: str, registered_model_name: Optional[str] = None) -> str:
#     """
#     Log a model artifact (e.g., an H2O-saved model directory) to MLflow.

#     Parameters
#     ----------
#     model_path : str
#         The local filesystem path containing the model artifacts.
#     registered_model_name : str, optional
#         If provided, will also attempt to register the model under this name.

#     Returns
#     -------
#     str
#         Confirmation message with any relevant registration details.
#     """
#     import mlflow
#     if registered_model_name:
#         mlflow.pyfunc.log_model(
#             artifact_path="model",
#             python_model=None,  # if you have a pyfunc wrapper, specify it
#             registered_model_name=registered_model_name,
#             code_path=None,
#             conda_env=None,
#             model_path=model_path  # for certain model flavors, or use flavors
#         )
#         return f"Model logged and registered under '{registered_model_name}' from path {model_path}"
#     else:
#         # Simple log as generic artifact
#         mlflow.pyfunc.log_model(
#             artifact_path="model",
#             python_model=None,
#             code_path=None,
#             conda_env=None,
#             model_path=model_path
#         )
#         return f"Model logged (no registration) from path {model_path}"


# @tool("mlflow_end_run", return_direct=True)
# def mlflow_end_run() -> str:
#     """
#     End the current MLflow run (if one is active).

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.end_run()
#     return "MLflow run ended."


# @tool("mlflow_search_runs", return_direct=True)
# def mlflow_search_runs(
#     experiment_names_or_ids: Optional[Union[List[str], List[int], str, int]] = None,
#     filter_string: Optional[str] = None
# ) -> str:
#     """
#     Search runs within one or more MLflow experiments, optionally filtering by a filter_string.

#     Parameters
#     ----------
#     experiment_names_or_ids : list or str or int, optional
#         Experiment IDs or names.
#     filter_string : str, optional
#         MLflow filter expression, e.g. "metrics.rmse < 1.0".

#     Returns
#     -------
#     str
#         JSON-formatted list of runs that match the query.
#     """
#     import mlflow
#     import json
#     if experiment_names_or_ids is None:
#         experiment_names_or_ids = []
#     if isinstance(experiment_names_or_ids, (str, int)):
#         experiment_names_or_ids = [experiment_names_or_ids]

#     df = mlflow.search_runs(
#         experiment_names=experiment_names_or_ids if all(isinstance(e, str) for e in experiment_names_or_ids) else None,
#         experiment_ids=experiment_names_or_ids if all(isinstance(e, int) for e in experiment_names_or_ids) else None,
#         filter_string=filter_string
#     )
#     return df.to_json(orient="records")


# @tool("mlflow_get_run", return_direct=True)
# def mlflow_get_run(run_id: str) -> str:
#     """
#     Retrieve details (params, metrics, etc.) for a specific MLflow run by ID.

#     Parameters
#     ----------
#     run_id : str
#         The ID of the MLflow run to retrieve.

#     Returns
#     -------
#     str
#         JSON-formatted data containing run info, params, and metrics.
#     """
#     from mlflow.tracking import MlflowClient
#     import json

#     client = MlflowClient()
#     run = client.get_run(run_id)
#     data = {
#         "run_id": run.info.run_id,
#         "experiment_id": run.info.experiment_id,
#         "status": run.info.status,
#         "start_time": run.info.start_time,
#         "end_time": run.info.end_time,
#         "artifact_uri": run.info.artifact_uri,
#         "params": run.data.params,
#         "metrics": run.data.metrics,
#         "tags": run.data.tags
#     }
#     return json.dumps(data)


# @tool("mlflow_load_model", return_direct=True)
# def mlflow_load_model(model_uri: str) -> str:
#     """
#     Load an MLflow-model (PyFunc flavor or other) into memory, returning a handle reference.
#     For demonstration, we store the loaded model globally in a registry dict.

#     Parameters
#     ----------
#     model_uri : str
#         The URI of the model to load, e.g. "runs:/<RUN_ID>/model" or "models:/MyModel/Production".

#     Returns
#     -------
#     str
#         A reference key identifying the loaded model (for subsequent predictions),
#         or a direct message if you prefer to store it differently.
#     """
#     import mlflow.pyfunc
#     from uuid import uuid4

#     # For demonstration, create a global registry:
#     global _LOADED_MODELS
#     if "_LOADED_MODELS" not in globals():
#         _LOADED_MODELS = {}

#     loaded_model = mlflow.pyfunc.load_model(model_uri)
#     model_key = f"model_{uuid4().hex}"
#     _LOADED_MODELS[model_key] = loaded_model

#     return f"Model loaded with reference key: {model_key}"


# @tool("mlflow_predict", return_direct=True)
# def mlflow_predict(model_key: str, data: List[Dict[str, Any]]) -> str:
#     """
#     Predict using a previously loaded MLflow model (PyFunc), identified by its reference key.

#     Parameters
#     ----------
#     model_key : str
#         The reference key for the loaded model (returned by mlflow_load_model).
#     data : List[Dict[str, Any]]
#         The data rows for which predictions should be made.

#     Returns
#     -------
#     str
#         JSON-formatted prediction results.
#     """
#     import pandas as pd
#     import json

#     global _LOADED_MODELS
#     if model_key not in _LOADED_MODELS:
#         return f"No model found for key: {model_key}"

#     model = _LOADED_MODELS[model_key]
#     df = pd.DataFrame(data)
#     preds = model.predict(df)
#     # Convert to JSON (DataFrame or Series)
#     if hasattr(preds, "to_json"):
#         return preds.to_json(orient="records")
#     else:
#         # If preds is just a numpy array or list
#         return json.dumps(preds.tolist())
