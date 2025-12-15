from langchain.tools import tool

import pandas as pd
import os
from pathlib import Path

from typing_extensions import Tuple, List, Dict, Optional

ALLOW_UNSAFE_PICKLE_ENV_VAR = "ALLOW_UNSAFE_PICKLE"
DEFAULT_MAX_MB = 20  # cap file size we attempt to load
DEFAULT_MAX_ROWS = 5000  # cap rows read per file to avoid OOM
DEFAULT_MAX_ENTRIES = 1000  # cap directory recursion output
DEFAULT_MAX_DEPTH = 5  # cap directory recursion depth


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
    directory_path: str = os.getcwd(),
    file_type: Optional[str] = None,
    max_mb: int = DEFAULT_MAX_MB,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Tuple[str, Dict]:
    """
    Tool: load_directory
    Description: Loads (reads) all recognized tabular files in a directory into memory.
                 If you only need filenames (a directory listing), use
                 `list_directory_contents` or `search_files_by_pattern` instead.
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

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}", {}

    if not base_path.is_dir():
        return f"Directory not found: {base_path}", {}

    data_frames: Dict[str, Dict] = {}
    max_bytes = max_mb * 1024 * 1024 if max_mb else None

    for filename in sorted(os.listdir(base_path)):
        file_path = base_path / filename

        # Skip directories
        if file_path.is_dir():
            continue

        # If file_type is specified, only process files that match.
        if file_type:
            # Make sure extension check is case-insensitive
            if not filename.lower().endswith(f".{file_type.lower()}"):
                continue

        if max_bytes is not None and file_path.stat().st_size > max_bytes:
            data_frames[filename] = {
                "status": "skipped",
                "data": None,
                "error": f"Skipped: file larger than {max_mb}MB",
            }
            continue

        try:
            # Attempt to auto-detect and load the file
            df_or_error = auto_load_file(str(file_path), max_rows=max_rows)
            if isinstance(df_or_error, pd.DataFrame):
                data_frames[filename] = {
                    "status": "ok",
                    "data": df_or_error.to_dict(),
                    "error": None,
                }
            else:
                data_frames[filename] = {
                    "status": "error",
                    "data": None,
                    "error": f"{df_or_error}",
                }
        except Exception as e:
            # If loading fails, record the error message
            data_frames[filename] = {
                "status": "error",
                "data": None,
                "error": f"Error loading file: {e}",
            }

    return (
        f"Returned the following files: {list(data_frames.keys())}",
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
    resolved_path, _matches = resolve_existing_file_path(file_path)
    resolved_path_str = str(resolved_path) if resolved_path is not None else str(file_path)
    df_or_error = auto_load_file(file_path, max_rows=DEFAULT_MAX_ROWS)

    if isinstance(df_or_error, pd.DataFrame):
        return (
            f"Returned the following data frame from this file: {file_path}",
            {
                "status": "ok",
                "data": df_or_error.to_dict(),
                "error": None,
                "file_path": resolved_path_str,
            },
        )

    return (
        f"Could not load file: {file_path}. {df_or_error}",
        {
            "status": "error",
            "data": None,
            "error": str(df_or_error),
            "file_path": resolved_path_str,
        },
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

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}", []

    if not base_path.is_dir():
        return f"Directory not found: {base_path}", []

    items = []
    for item in os.listdir(base_path):
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
    directory_path: str = os.getcwd(),
    show_hidden: bool = False,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
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
        return "No directory path provided.", []

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}", []

    if not base_path.is_dir():
        return f"Directory not found: {base_path}", []

    lines = []
    records = []
    entry_count = 0

    def recurse(path: str, indent_level: int = 0):
        nonlocal entry_count
        if indent_level > max_depth:
            lines.append("  " * indent_level + "[max depth reached]")
            return
        if entry_count >= max_entries:
            return
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
                if entry_count >= max_entries:
                    continue
                lines.append(f"{prefix}{item}/")
                records.append(
                    {
                        "type": "directory",
                        "name": item,
                        "parent_path": path,
                        "absolute_path": full_path,
                    }
                )
                entry_count += 1
                # Recursively descend
                recurse(full_path, indent_level + 1)
            else:
                # File
                if entry_count >= max_entries:
                    continue
                lines.append(f"{prefix}- {item}")
                records.append(
                    {
                        "type": "file",
                        "name": item,
                        "parent_path": path,
                        "absolute_path": full_path,
                    }
                )
                entry_count += 1

    # Kick off recursion
    # Add the top-level directory to lines/records if you like
    dir_name = os.path.basename(os.path.normpath(base_path)) or str(base_path)
    lines.append(f"{dir_name}/")  # Show the root as well
    records.append(
        {
            "type": "directory",
            "name": dir_name,
            "parent_path": os.path.dirname(base_path),
            "absolute_path": os.path.abspath(base_path),
        }
    )
    entry_count += 1
    recurse(str(base_path), indent_level=1)

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
        return f"{file_path} is not a valid file.", [
            {"type": "error", "file_path": file_path}
        ]

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

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}", []

    if not base_path.is_dir():
        return f"Directory not found: {base_path}", []

    matched_files = []
    if recursive:
        for root, dirs, files in os.walk(base_path):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    matched_files.append(os.path.join(root, filename))
    else:
        # Non-recursive
        for filename in os.listdir(base_path):
            full_path = os.path.join(base_path, filename)
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


def resolve_existing_file_path(file_path: str) -> tuple[Path | None, list[str]]:
    """
    Resolve `file_path` to an existing file path (best effort).

    Returns:
      (resolved_path, candidate_matches_for_error)
    """
    raw = (file_path or "").strip()
    if not raw:
        return None, []

    path = Path(raw).expanduser()
    if path.is_file():
        return path, []

    def _candidate_roots() -> list[Path]:
        roots: list[Path] = []
        try:
            cwd = Path(os.getcwd()).expanduser().resolve()
            roots.append(cwd)
            # Walk up a few levels so relative paths work even when Streamlit
            # is launched from a subdirectory (e.g., apps/...).
            for parent in list(cwd.parents)[: DEFAULT_MAX_DEPTH + 1]:
                roots.append(parent)
        except Exception:
            roots.append(Path("."))

        # Also try relative to the package location (helps when cwd isn't the repo root).
        try:
            here = Path(__file__).expanduser().resolve()
            pkg_root = here.parents[2] if len(here.parents) > 2 else here.parent
            roots.append(pkg_root)
            for parent in list(pkg_root.parents)[: DEFAULT_MAX_DEPTH + 1]:
                roots.append(parent)
        except Exception:
            pass

        # De-dupe while preserving order.
        out: list[Path] = []
        seen: set[str] = set()
        for r in roots:
            try:
                key = str(r)
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    # If the user provided a relative filename (common in chat), try a few conventional locations.
    candidates: list[Path] = []
    if not path.is_absolute():
        base = path.name
        for root in _candidate_roots():
            # As-given relative to each candidate root (handles `data/foo.csv`)
            candidates.append(root / path)
            if base:
                # Common project folders
                candidates.extend(
                    [
                        root / "data" / base,
                        root / "temp" / base,
                        root / "temp" / "uploads" / base,
                    ]
                )

    for cand in candidates:
        try:
            resolved = cand.expanduser().resolve()
        except Exception:
            resolved = cand
        if resolved.is_file():
            return resolved, []

    # Last resort: shallow, depth-limited search for the basename.
    base = path.name
    if not base or path.is_absolute():
        return None, []

    matches: list[str] = []
    roots: list[Path] = []
    for root in _candidate_roots()[:3]:
        roots.extend([root / "data", root])
    for root in roots:
        try:
            root = root.expanduser().resolve()
        except Exception:
            continue
        if not root.exists() or not root.is_dir():
            continue

        root_depth = len(root.parts)
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                current = Path(dirpath)
                depth = len(current.parts) - root_depth
                if depth >= DEFAULT_MAX_DEPTH:
                    dirnames[:] = []
                for fn in filenames:
                    if fn == base:
                        try:
                            matches.append(str((current / fn).resolve()))
                        except Exception:
                            matches.append(str(current / fn))
                        if len(matches) >= 5:
                            break
                if len(matches) >= 5:
                    break
        except Exception:
            continue
        if matches:
            break

    if len(matches) == 1:
        return Path(matches[0]), []
    return None, matches


def auto_load_file(file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
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

    resolved_path, matches = resolve_existing_file_path(file_path)
    if resolved_path is None:
        if matches:
            shown = "\n".join([f"- {m}" for m in matches])
            return (
                f"File not found: {file_path}. Multiple matches found; please specify a full path:\n{shown}"
            )
        hint = " Try `data/<filename>` if it's in the project data folder."
        return f"File not found: {file_path}.{hint}"

    path = resolved_path

    suffixes = "".join(path.suffixes).lower()
    ext = path.suffix.lower()

    try:
        if suffixes in {".csv", ".csv.gz"} or ext in {".csv", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            return load_csv(str(path), sep=sep, nrows=max_rows)
        if ext in [".xlsx", ".xls"]:
            return load_excel(str(path), nrows=max_rows)
        if suffixes in {".jsonl", ".ndjson"} or ext in {".jsonl", ".ndjson"}:
            return load_json(str(path), lines=True, nrows=max_rows)
        if ext == ".json":
            return load_json(str(path), lines=False, nrows=max_rows)
        if ext == ".parquet":
            return load_parquet(str(path), max_rows=max_rows)
        if ext == ".pkl":
            return load_pickle(str(path))
        return f"Unsupported file extension: {suffixes or ext}"
    except Exception as e:
        return f"Error loading file: {e}"


def load_csv(file_path: str, sep: str = ",", nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Tool: load_csv
    Description: Loads a CSV file into a pandas DataFrame.
    Args:
      file_path (str): Path to the CSV file.
    Returns:
      pd.DataFrame
    """
    import pandas as pd

    return pd.read_csv(file_path, sep=sep, nrows=nrows)


def load_excel(file_path: str, sheet_name=None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Tool: load_excel
    Description: Loads an Excel file into a pandas DataFrame.
    """
    import pandas as pd

    return pd.read_excel(file_path, sheet_name=sheet_name, nrows=nrows)


def load_json(file_path: str, lines: bool = False, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Tool: load_json
    Description: Loads a JSON file or NDJSON into a pandas DataFrame.
    """
    import pandas as pd

    # For simple JSON arrays or line-delimited JSON
    return pd.read_json(file_path, orient="records", lines=lines, nrows=nrows)


def load_parquet(file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Tool: load_parquet
    Description: Loads a Parquet file into a pandas DataFrame.
    """
    import pandas as pd

    df = pd.read_parquet(file_path)
    if max_rows is not None and len(df) > max_rows:
        return df.head(max_rows)
    return df


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
