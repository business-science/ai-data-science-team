from langchain.tools import tool

import pandas as pd
import os

from typing_extensions import Tuple, List, Dict, Optional

ALLOW_UNSAFE_PICKLE_ENV_VAR = "ALLOW_UNSAFE_PICKLE"


def _pickle_loading_allowed() -> bool:
    """
    Pickle deserialization executes arbitrary code.
    This helper enforces an explicit opt-in via env var to avoid RCE on untrusted data.
    """
    return os.getenv(ALLOW_UNSAFE_PICKLE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


@tool(response_format="content_and_artifact")
def load_directory(
    directory_path: str = os.getcwd(), file_type: Optional[str] = None
) -> Tuple[str, Dict]:
    """
    Tool: load_directory
    Description: Loads all recognized tabular files in a directory.
                 If file_type is specified (e.g., 'csv'), only files
                 with that extension are loaded.

    Parameters:
    ----------
    directory_path : str
        The path to the directory to load. Defaults to the current working directory.

    file_type : str, optional
        The extension of the file type you want to load exclusively
        (e.g., 'csv', 'xlsx', 'parquet'). If None or not provided,
        attempts to load all recognized tabular files.

    Returns:
    -------
    Tuple[str, Dict]
        A tuple containing a message and a dictionary of data frames.
    """
    print(f"    * Tool: load_directory | {directory_path}")

    import os
    import pandas as pd

    if directory_path is None:
        return "No directory path provided.", {}

    if not os.path.isdir(directory_path):
        return f"Directory not found: {directory_path}", {}

    data_frames = {}

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # If file_type is specified, only process files that match.
        if file_type:
            # Make sure extension check is case-insensitive
            if not filename.lower().endswith(f".{file_type.lower()}"):
                continue

        try:
            # Attempt to auto-detect and load the file
            df_or_error = auto_load_file(file_path)
            if isinstance(df_or_error, pd.DataFrame):
                data_frames[filename] = df_or_error.to_dict()
            else:
                data_frames[filename] = f"Error loading file: {df_or_error}"
        except Exception as e:
            # If loading fails, record the error message
            data_frames[filename] = f"Error loading file: {e}"

    return (
        f"Returned the following data frames: {list(data_frames.keys())}",
        data_frames,
    )


@tool(response_format="content_and_artifact")
def load_file(file_path: str) -> Tuple[str, Dict]:
    """
    Automatically loads a file based on its extension.

    Parameters:
    ----------
    file_path : str
        The path to the file to load.

    Returns:
    -------
    Tuple[str, Dict]
        A tuple containing a message and a dictionary of the data frame.
    """
    print(f"    * Tool: load_file | {file_path}")
    df_or_error = auto_load_file(file_path)

    if isinstance(df_or_error, pd.DataFrame):
        return (
            f"Returned the following data frame from this file: {file_path}",
            df_or_error.to_dict(),
        )

    return (
        f"Could not load file: {file_path}. {df_or_error}",
        {"file_path": file_path, "error": str(df_or_error)},
    )


@tool(response_format="content_and_artifact")
def list_directory_contents(
    directory_path: str = os.getcwd(), show_hidden: bool = False
) -> Tuple[List[str], List[Dict]]:
    """
    Tool: list_directory_contents
    Description: Lists all files and folders in the specified directory.
    Args:
        directory_path (str): The path of the directory to list.
        show_hidden (bool): Whether to include hidden files (default: False).
    Returns:
        tuple:
            - content (list[str]): A list of filenames/folders (suitable for display)
            - artifact (list[dict]): A list of dictionaries where each dict includes
              the keys {"filename": <name>, "type": <'file' or 'directory'>}.
              This structure can be easily converted to a pandas DataFrame.
    """
    print(f"    * Tool: list_directory_contents | {directory_path}")
    import os

    if directory_path is None:
        return "No directory path provided.", []

    if not os.path.isdir(directory_path):
        return f"Directory not found: {directory_path}", []

    items = []
    for item in os.listdir(directory_path):
        # If show_hidden is False, skip items starting with '.'
        if not show_hidden and item.startswith("."):
            continue
        items.append(item)
    items.reverse()

    # content: just the raw list of item names (files/folders).
    content = items.copy()

    content.append(f"Total items: {len(items)}")
    content.append(f"Directory: {directory_path}")

    # artifact: list of dicts with both "filename" and "type" keys.
    artifact = []
    for item in items:
        item_path = os.path.join(directory_path, item)
        artifact.append(
            {
                "filename": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
            }
        )

    return content, artifact


@tool(response_format="content_and_artifact")
def list_directory_recursive(
    directory_path: str = os.getcwd(), show_hidden: bool = False
) -> Tuple[str, List[Dict]]:
    """
    Tool: list_directory_recursive
    Description:
        Recursively lists all files and folders within the specified directory.
        Returns a two-tuple:
          (1) A human-readable tree representation of the directory (content).
          (2) A list of dicts (artifact) that can be easily converted into a DataFrame.

    Args:
        directory_path (str): The path of the directory to list.
        show_hidden (bool): Whether to include hidden files (default: False).

    Returns:
        Tuple[str, List[dict]]:
            content: A multiline string showing the directory tree.
            artifact: A list of dictionaries, each with information about a file or directory.

    Example:
        content, artifact = list_directory_recursive("/path/to/folder", show_hidden=False)
    """
    print(f"    * Tool: list_directory_recursive | {directory_path}")

    # We'll store two things as we recurse:
    # 1) lines for building the "tree" string
    # 2) records in a list of dicts for easy DataFrame creation
    import os

    if directory_path is None:
        return "No directory path provided.", {}

    if not os.path.isdir(directory_path):
        return f"Directory not found: {directory_path}", {}

    lines = []
    records = []

    def recurse(path: str, indent_level: int = 0):
        # List items in the current directory
        try:
            items = os.listdir(path)
        except PermissionError:
            # If we don't have permission to read the directory, just note it.
            lines.append("  " * indent_level + "[Permission Denied]")
            return

        # Sort items for a consistent order (optional)
        items.sort()

        for item in items:
            if not show_hidden and item.startswith("."):
                continue

            full_path = os.path.join(path, item)
            # Build an indented prefix for the tree
            prefix = "  " * indent_level

            if os.path.isdir(full_path):
                # Directory
                lines.append(f"{prefix}{item}/")
                records.append(
                    {
                        "type": "directory",
                        "name": item,
                        "parent_path": path,
                        "absolute_path": full_path,
                    }
                )
                # Recursively descend
                recurse(full_path, indent_level + 1)
            else:
                # File
                lines.append(f"{prefix}- {item}")
                records.append(
                    {
                        "type": "file",
                        "name": item,
                        "parent_path": path,
                        "absolute_path": full_path,
                    }
                )

    # Kick off recursion
    if os.path.isdir(directory_path):
        # Add the top-level directory to lines/records if you like
        dir_name = os.path.basename(os.path.normpath(directory_path)) or directory_path
        lines.append(f"{dir_name}/")  # Show the root as well
        records.append(
            {
                "type": "directory",
                "name": dir_name,
                "parent_path": os.path.dirname(directory_path),
                "absolute_path": os.path.abspath(directory_path),
            }
        )
        recurse(directory_path, indent_level=1)
    else:
        # If the given path is not a directory, just return a note
        lines.append(f"{directory_path} is not a directory.")
        records.append(
            {
                "type": "error",
                "name": directory_path,
                "parent_path": None,
                "absolute_path": os.path.abspath(directory_path),
            }
        )

    # content: multiline string with the entire tree
    content = "\n".join(lines)
    # artifact: list of dicts, easily converted into a DataFrame
    artifact = records

    return content, artifact


@tool(response_format="content_and_artifact")
def get_file_info(file_path: str) -> Tuple[str, List[Dict]]:
    """
    Tool: get_file_info
    Description: Retrieves metadata (size, modification time, etc.) about a file.
                 Returns a tuple (content, artifact):
                   - content (str): A textual summary of the file info.
                   - artifact (List[Dict]): A list with a single dictionary of file metadata.
                                            Useful for direct conversion into a DataFrame.
    Args:
        file_path (str): The path of the file to inspect.
    Returns:
        Tuple[str, List[dict]]:
            content: Summary text
            artifact: A list[dict] of file metadata
    Example:
        content, artifact = get_file_info("/path/to/mydata.csv")
    """
    print(f"    * Tool: get_file_info | {file_path}")

    # Ensure the file exists
    import os
    import time

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} is not a valid file.")

    file_stats = os.stat(file_path)

    # Construct the data dictionary
    file_data = {
        "file_name": os.path.basename(file_path),
        "size_bytes": file_stats.st_size,
        "modification_time": time.ctime(file_stats.st_mtime),
        "absolute_path": os.path.abspath(file_path),
    }

    # Create a user-friendly summary (content)
    content_str = (
        f"File Name: {file_data['file_name']}\n"
        f"Size (bytes): {file_data['size_bytes']}\n"
        f"Last Modified: {file_data['modification_time']}\n"
        f"Absolute Path: {file_data['absolute_path']}"
    )

    # Artifact should be a list of dict(s) to easily convert to DataFrame
    artifact = [file_data]

    return content_str, artifact


@tool(response_format="content_and_artifact")
def search_files_by_pattern(
    directory_path: str = os.getcwd(), pattern: str = "*.csv", recursive: bool = False
) -> Tuple[str, List[Dict]]:
    """
    Tool: search_files_by_pattern
    Description:
        Searches for files (optionally in subdirectories) that match a given
        wildcard pattern (e.g. "*.csv", "*.xlsx", etc.), returning a tuple:
          (1) content (str): A multiline summary of the matched files.
          (2) artifact (List[Dict]): A list of dicts with file path info.

    Args:
        directory_path (str): Directory path to start searching from.
        pattern (str): A wildcard pattern, e.g. "*.csv". Default is "*.csv".
        recursive (bool): Whether to search in subdirectories. Default is False.

    Returns:
        Tuple[str, List[Dict]]:
            content: A user-friendly string showing matched file paths.
            artifact: A list of dictionaries, each representing a matched file.

    Example:
        content, artifact = search_files_by_pattern("/path/to/folder", "*.csv", recursive=True)
    """
    print(f"    * Tool: search_files_by_pattern | {directory_path}")

    import os
    import fnmatch

    matched_files = []
    if recursive:
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    matched_files.append(os.path.join(root, filename))
    else:
        # Non-recursive
        for filename in os.listdir(directory_path):
            full_path = os.path.join(directory_path, filename)
            if os.path.isfile(full_path) and fnmatch.fnmatch(filename, pattern):
                matched_files.append(full_path)

    # Create a human-readable summary (content)
    if matched_files:
        lines = [f"Found {len(matched_files)} file(s) matching '{pattern}':"]
        for f in matched_files:
            lines.append(f" - {f}")
        content = "\n".join(lines)
    else:
        content = f"No files found matching '{pattern}'."

    # Create artifact as a list of dicts for DataFrame conversion
    artifact = [{"file_path": path} for path in matched_files]

    return content, artifact


# Loaders


def auto_load_file(file_path: str) -> pd.DataFrame:
    """
    Auto loads a file based on its extension.

    Parameters:
    ----------
    file_path : str
        The path to the file to load.

    Returns:
    -------
    pd.DataFrame
    """
    import pandas as pd

    try:
        ext = file_path.split(".")[-1].lower()
        if ext == "csv":
            return load_csv(file_path)
        elif ext in ["xlsx", "xls"]:
            return load_excel(file_path)
        elif ext == "json":
            return load_json(file_path)
        elif ext == "parquet":
            return load_parquet(file_path)
        elif ext == "pkl":
            return load_pickle(file_path)
        else:
            return f"Unsupported file extension: {ext}"
    except Exception as e:
        return f"Error loading file: {e}"


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Tool: load_csv
    Description: Loads a CSV file into a pandas DataFrame.
    Args:
      file_path (str): Path to the CSV file.
    Returns:
      pd.DataFrame
    """
    import pandas as pd

    return pd.read_csv(file_path)


def load_excel(file_path: str, sheet_name=None) -> pd.DataFrame:
    """
    Tool: load_excel
    Description: Loads an Excel file into a pandas DataFrame.
    """
    import pandas as pd

    return pd.read_excel(file_path, sheet_name=sheet_name)


def load_json(file_path: str) -> pd.DataFrame:
    """
    Tool: load_json
    Description: Loads a JSON file or NDJSON into a pandas DataFrame.
    """
    import pandas as pd

    # For simple JSON arrays
    return pd.read_json(file_path, orient="records", lines=False)


def load_parquet(file_path: str) -> pd.DataFrame:
    """
    Tool: load_parquet
    Description: Loads a Parquet file into a pandas DataFrame.
    """
    import pandas as pd

    return pd.read_parquet(file_path)


def load_pickle(file_path: str) -> pd.DataFrame:
    """
    Tool: load_pickle
    Description: Loads a Pickle file into a pandas DataFrame.
    Security:
    Pickle deserialization can execute arbitrary code. Loading pickle files is
    disabled by default and requires explicit opt-in via the environment
    variable ALLOW_UNSAFE_PICKLE=1. Only enable this for trusted data sources.
    """
    import pandas as pd

    if not _pickle_loading_allowed():
        raise ValueError(
            "Pickle loading is disabled by default to avoid arbitrary code execution. "
            f"Set {ALLOW_UNSAFE_PICKLE_ENV_VAR}=1 only when loading trusted data."
        )

    return pd.read_pickle(file_path)
