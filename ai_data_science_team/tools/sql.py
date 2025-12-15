import pandas as pd
import sqlalchemy as sql
from sqlalchemy import inspect


def get_database_metadata(connection, n_samples=10) -> dict:
    """
    Collects metadata and sample data from a database, with safe identifier quoting and
    basic dialect-aware row limiting. Prevents issues with spaces/reserved words in identifiers.

    Parameters
    ----------
    connection : Union[sql.engine.base.Connection, sql.engine.base.Engine]
        An active SQLAlchemy connection or engine.
    n_samples : int
        Number of sample values to retrieve for each column.

    Returns
    -------
    dict
        A dictionary with database metadata, including some sample data from each column.
    """
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection

    metadata = {
        "dialect": None,
        "driver": None,
        "connection_url": None,
        "schemas": [],
    }

    try:
        sql_engine = conn.engine
        dialect_name = sql_engine.dialect.name.lower()

        metadata["dialect"] = sql_engine.dialect.name
        metadata["driver"] = sql_engine.driver
        try:
            metadata["connection_url"] = sql_engine.url.render_as_string(
                hide_password=True
            )
        except Exception:
            metadata["connection_url"] = str(sql_engine.url)

        inspector = inspect(sql_engine)
        preparer = inspector.bind.dialect.identifier_preparer

        # For each schema
        for schema_name in inspector.get_schema_names():
            schema_obj = {"schema_name": schema_name, "tables": []}

            tables = inspector.get_table_names(schema=schema_name)
            for table_name in tables:
                table_info = {
                    "table_name": table_name,
                    "columns": [],
                    "primary_key": [],
                    "foreign_keys": [],
                    "indexes": [],
                }
                # Get columns
                columns = inspector.get_columns(table_name, schema=schema_name)
                for col in columns:
                    col_name = col["name"]
                    col_type = str(col["type"])
                    table_name_quoted = f"{preparer.quote_identifier(schema_name)}.{preparer.quote_identifier(table_name)}"
                    col_name_quoted = preparer.quote_identifier(col_name)

                    # Build query for sample data
                    query = build_query(
                        col_name_quoted, table_name_quoted, n_samples, dialect_name
                    )

                    # Retrieve sample data
                    try:
                        df = pd.read_sql(query, conn)
                        samples = df[col_name].head(n_samples).tolist()
                    except Exception as e:
                        samples = [f"Error retrieving data: {str(e)}"]

                    table_info["columns"].append(
                        {"name": col_name, "type": col_type, "sample_values": samples}
                    )

                # Primary keys
                pk_constraint = inspector.get_pk_constraint(
                    table_name, schema=schema_name
                )
                table_info["primary_key"] = pk_constraint.get("constrained_columns", [])

                # Foreign keys
                fks = inspector.get_foreign_keys(table_name, schema=schema_name)
                table_info["foreign_keys"] = [
                    {
                        "local_cols": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_cols": fk["referred_columns"],
                    }
                    for fk in fks
                ]

                # Indexes
                idxs = inspector.get_indexes(table_name, schema=schema_name)
                table_info["indexes"] = idxs

                schema_obj["tables"].append(table_info)

            metadata["schemas"].append(schema_obj)

    finally:
        if is_engine:
            conn.close()

    return metadata


def build_query(
    col_name_quoted: str, table_name_quoted: str, n: int, dialect_name: str
) -> str:
    # Example: expand your build_query to handle random sampling if possible
    if "postgres" in dialect_name:
        return f"SELECT {col_name_quoted} FROM {table_name_quoted} ORDER BY RANDOM() LIMIT {n}"
    if "mysql" in dialect_name:
        return f"SELECT {col_name_quoted} FROM {table_name_quoted} ORDER BY RAND() LIMIT {n}"
    if "sqlite" in dialect_name:
        return f"SELECT {col_name_quoted} FROM {table_name_quoted} ORDER BY RANDOM() LIMIT {n}"
    if "mssql" in dialect_name:
        return f"SELECT TOP {n} {col_name_quoted} FROM {table_name_quoted} ORDER BY NEWID()"
    # Oracle or fallback
    return f"SELECT {col_name_quoted} FROM {table_name_quoted} WHERE ROWNUM <= {n}"
