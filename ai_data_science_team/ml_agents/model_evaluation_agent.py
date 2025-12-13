# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Model Evaluation Agent

from __future__ import annotations

from typing import Any, Optional, Sequence, Dict

import pandas as pd

from langchain_core.messages import BaseMessage, AIMessage

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.messages import get_last_user_message_content


AGENT_NAME = "model_evaluation_agent"


def _infer_task_type(y: pd.Series) -> str:
    try:
        if y.dtype == "bool":
            return "classification"
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            return "classification"
        uniq = y.dropna().unique()
        if len(uniq) <= 20 and pd.api.types.is_integer_dtype(y):
            return "classification"
    except Exception:
        pass
    return "regression"


def _choose_positive_label(labels) -> Optional[str]:
    labels = [str(x) for x in labels if x is not None]
    if not labels:
        return None
    for candidate in ("yes", "true", "1", "churn", "positive"):
        for lab in labels:
            if lab.strip().lower() == candidate:
                return lab
    # Prefer "Yes" if present
    for lab in labels:
        if lab.strip().lower() == "yes":
            return lab
    # Fall back to the last label to be stable (often the positive class)
    return labels[-1]


class ModelEvaluationAgent(BaseAgent):
    """
    Deterministically evaluates an H2O model on a holdout split and returns
    standardized metrics + a Plotly figure for inspection in Streamlit.
    """

    def __init__(self, model: Any = None, log: bool = False):
        self._params = {"model": model, "log": log}
        self.response: Optional[dict] = None

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self._params[k] = v

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        data_raw: pd.DataFrame,
        model_artifacts: Optional[Dict[str, Any]] = None,
        target_variable: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ):
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            user_instructions = get_last_user_message_content(messages)

        model_artifacts = model_artifacts or {}

        if data_raw is None or not isinstance(data_raw, pd.DataFrame) or data_raw.empty:
            self.response = {
                "messages": [
                    AIMessage(
                        content="No dataset is available to evaluate. Load data and train a model first.",
                        name=AGENT_NAME,
                    )
                ],
                "eval_artifacts": None,
            }
            return None

        if not target_variable:
            self.response = {
                "messages": [
                    AIMessage(
                        content=(
                            "To evaluate the model, tell me the target column name "
                            "(e.g., `target=Churn`) so I can compute holdout metrics."
                        ),
                        name=AGENT_NAME,
                    )
                ],
                "eval_artifacts": None,
            }
            return None

        if target_variable not in data_raw.columns:
            cols = ", ".join(list(map(str, data_raw.columns[:40])))
            more = "..." if len(data_raw.columns) > 40 else ""
            self.response = {
                "messages": [
                    AIMessage(
                        content=(
                            f"Target column `{target_variable}` was not found in the dataset. "
                            f"Available columns: {cols}{more}"
                        ),
                        name=AGENT_NAME,
                    )
                ],
                "eval_artifacts": None,
            }
            return None

        model_path = None
        best_model_id = None
        if isinstance(model_artifacts, dict):
            model_path = model_artifacts.get("model_path") or model_artifacts.get("modelPath")
            best_model_id = model_artifacts.get("best_model_id") or model_artifacts.get("bestModelId")

        df = data_raw.copy()
        y = df[target_variable]
        X = df.drop(columns=[target_variable])
        task_type = _infer_task_type(y)

        # Load/resolve the H2O model.
        try:
            import h2o

            h2o.init()
            h2o_model = None
            if model_path:
                try:
                    h2o_model = h2o.load_model(model_path)
                except Exception:
                    h2o_model = None
            if h2o_model is None and best_model_id:
                h2o_model = h2o.get_model(best_model_id)
        except Exception as e:
            self.response = {
                "messages": [
                    AIMessage(
                        content=f"H2O is required to evaluate the trained model but could not be initialized/loaded: {e}",
                        name=AGENT_NAME,
                    )
                ],
                "eval_artifacts": None,
            }
            return None

        if h2o_model is None:
            self.response = {
                "messages": [
                    AIMessage(
                        content=(
                            "I couldn't load the trained H2O model for evaluation. "
                            "Re-run training with model saving enabled, or provide a valid model path."
                        ),
                        name=AGENT_NAME,
                    )
                ],
                "eval_artifacts": None,
            }
            return None

        # Prefer cross-validation holdout predictions (no leakage) when available.
        eval_source = "cross_validation_holdout"
        preds = None
        y_true = None
        try:
            cv_preds = getattr(h2o_model, "cross_validation_holdout_predictions", None)
            if callable(cv_preds):
                cv_frame = cv_preds()
                if cv_frame is not None:
                    preds = cv_frame.as_data_frame()
                    y_true = y
        except Exception:
            preds = None
            y_true = None

        # Fallback: predict on a random test split (note: model was trained on full data; this can be optimistic).
        if preds is None:
            eval_source = "random_split_in_sample"
            try:
                from sklearn.model_selection import train_test_split

                stratify = None
                if task_type == "classification":
                    stratify = y
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
                test_df = pd.concat([X_test, y_test], axis=1)
                import h2o

                test_h2o = h2o.H2OFrame(test_df)
                preds_h2o = h2o_model.predict(test_h2o)
                preds = preds_h2o.as_data_frame()
                y_true = y_test
            except Exception as e:
                self.response = {
                    "messages": [
                        AIMessage(
                            content=f"Evaluation failed during prediction: {e}",
                            name=AGENT_NAME,
                        )
                    ],
                    "eval_artifacts": None,
                }
                return None

        eval_artifacts: dict = {
            "target_variable": target_variable,
            "task_type": task_type,
            "test_size": test_size,
            "model_path": model_path,
            "best_model_id": best_model_id,
            "evaluation_source": eval_source,
        }

        plotly_graph = None
        summary_lines: list[str] = []

        if task_type == "classification":
            try:
                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    roc_auc_score,
                    confusion_matrix,
                    roc_curve,
                )
                import plotly.graph_objects as go

                y_true = y_true.astype(str)
                y_pred = preds.get("predict")
                if y_pred is None:
                    raise ValueError("H2O prediction output missing 'predict' column.")
                y_pred = y_pred.astype(str)

                labels = sorted(set(y_true.dropna().unique().tolist()))
                pos_label = _choose_positive_label(labels)

                metrics: dict = {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                }
                if pos_label is not None:
                    metrics.update(
                        {
                            "precision": float(
                                precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
                            ),
                            "recall": float(
                                recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
                            ),
                            "f1": float(
                                f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
                            ),
                        }
                    )

                # Probability column for AUC/ROC if present
                auc = None
                roc_fig = None
                if pos_label is not None and pos_label in preds.columns:
                    y_score = preds[pos_label].astype(float)
                    try:
                        auc = float(roc_auc_score((y_true == pos_label).astype(int), y_score))
                        fpr, tpr, _ = roc_curve((y_true == pos_label).astype(int), y_score)
                        roc_fig = go.Figure()
                        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
                        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
                        roc_fig.update_layout(
                            title="ROC Curve (Holdout)",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                        )
                    except Exception:
                        auc = None
                if auc is not None:
                    metrics["auc"] = auc

                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])

                cm_fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=labels,
                        y=labels,
                        colorscale="Viridis",
                        showscale=True,
                    )
                )
                cm_fig.update_layout(
                    title="Confusion Matrix (Holdout)",
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                )

                # Prefer confusion matrix as the default plot; attach ROC separately
                plotly_graph = cm_fig.to_dict()
                eval_artifacts.update(
                    {
                        "metrics": metrics,
                        "positive_label": pos_label,
                        "confusion_matrix": cm_df.to_dict(),
                        "roc_curve": roc_fig.to_dict() if roc_fig is not None else None,
                    }
                )
                summary_lines.append("Holdout classification metrics:")
                if eval_source == "random_split_in_sample":
                    summary_lines.append(
                        "- note: evaluation used a random split but the model may have been trained on the full dataset (optimistic)."
                    )
                for k in ["auc", "accuracy", "precision", "recall", "f1"]:
                    if k in metrics:
                        summary_lines.append(f"- {k}: {metrics[k]:.4f}")
            except Exception as e:
                eval_artifacts["error"] = str(e)
                summary_lines.append(f"Evaluation failed: {e}")
        else:
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                import numpy as np
                import plotly.express as px

                y_true = y_true.astype(float)
                y_pred = preds.get("predict")
                if y_pred is None:
                    raise ValueError("H2O prediction output missing 'predict' column.")
                y_pred = y_pred.astype(float)

                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))
                metrics = {"rmse": rmse, "mae": mae, "r2": r2}

                resid = (y_true - y_pred).rename("residual")
                plot_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": resid})
                fig = px.scatter(plot_df, x="y_pred", y="residual", title="Residuals vs Predicted (Holdout)")
                fig.update_layout(xaxis_title="Predicted", yaxis_title="Residual")
                plotly_graph = fig.to_dict()

                eval_artifacts.update({"metrics": metrics})
                summary_lines.append("Holdout regression metrics:")
                if eval_source == "random_split_in_sample":
                    summary_lines.append(
                        "- note: evaluation used a random split but the model may have been trained on the full dataset (optimistic)."
                    )
                summary_lines.append(f"- rmse: {rmse:.4f}")
                summary_lines.append(f"- mae: {mae:.4f}")
                summary_lines.append(f"- r2: {r2:.4f}")
            except Exception as e:
                eval_artifacts["error"] = str(e)
                summary_lines.append(f"Evaluation failed: {e}")

        msg = "\n".join(summary_lines) if summary_lines else "Model evaluation complete."
        self.response = {
            "messages": [AIMessage(content=msg, name=AGENT_NAME)],
            "eval_artifacts": eval_artifacts,
            "plotly_graph": plotly_graph,
        }
        return None

    def get_eval_artifacts(self) -> Optional[dict]:
        if not self.response:
            return None
        return self.response.get("eval_artifacts")
