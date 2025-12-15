# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Workflow Planner Agent

from __future__ import annotations

from typing import Any, Optional, Sequence, Dict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.messages import get_last_user_message_content


AGENT_NAME = "workflow_planner_agent"


def _safe_json_loads(text: str) -> dict:
    import json
    import re

    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract the first JSON object or array from the text.
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return {}
    candidate = m.group(1)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        return {"steps": parsed}
    except Exception:
        return {}


class WorkflowPlannerAgent(BaseAgent):
    """
    Produces a structured, ordered workflow plan for the supervisor-led DS team.

    This agent does not execute data tasks; it only returns a plan + any questions
    needed to proceed (e.g., missing file path or target column).
    """

    def __init__(self, model: Any, log: bool = False):
        self._params = {"model": model, "log": log}
        self.response: Optional[dict] = None

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self._params[k] = v

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        context: Optional[Dict[str, Any]] = None,
        user_instructions: Optional[str] = None,
        **kwargs,
    ):
        llm = self._params["model"]
        if user_instructions is None:
            user_instructions = get_last_user_message_content(messages)
        context = context or {}
        proactive_mode = bool(context.get("proactive_workflow_mode"))

        system = (
            "You are a workflow planning agent for a supervisor-led data science team.\n"
            "Return ONLY valid JSON.\n\n"
            "You can plan ONLY these executable steps (in order):\n"
            "- list_files (list files in a directory; do not load file contents)\n"
            "- load (load file from disk)\n"
            "- merge (merge/join/concat multiple datasets)\n"
            "- sql (run a SQL query)\n"
            "- wrangle (reshape/transform)\n"
            "- clean (impute/fix types/outliers)\n"
            "- eda (describe/missingness/correlation/reports)\n"
            "- viz (plotly chart)\n"
            "- feature (feature engineering)\n"
            "- model (H2O AutoML training)\n"
            "- evaluate (holdout evaluation: metrics + plots)\n"
            "- mlflow_log (log workflow artifacts: metrics/tables/figures to MLflow)\n"
            "- mlflow_tools (inspect MLflow: list/search runs/artifacts, launch UI)\n\n"
            "Rules:\n"
            "- Output schema: {\"steps\": [..], \"target_variable\": str|null, \"questions\": [..], \"notes\": [..]}.\n"
            "- steps must be a de-duplicated ordered list of step IDs from the allowed set.\n"
            "- The word \"model\" can be ambiguous (e.g., a product \"bike model\" vs an ML model). "
            "Only include the ML step `model` when the user explicitly asks to train/build/predict with an ML model.\n"
            "- If required info is missing (e.g., file path for load, target column for model), "
            "put a short question in questions and omit dependent steps.\n"
            "- If you include 'model' or 'evaluate', you MUST set target_variable or ask for it and omit those steps.\n"
            "- Prefer a minimal plan that satisfies the user request.\n"
            "- If proactive_workflow_mode is OFF, include ONLY the steps explicitly requested (plus prerequisites).\n"
            "- If proactive_workflow_mode is ON, you MAY propose a reasonable end-to-end workflow for broad requests "
            "(e.g., \"analyze\", \"explore\", \"full workflow\"), but keep narrow requests narrow.\n"
            f"- proactive_workflow_mode={'ON' if proactive_mode else 'OFF'}.\n"
        )

        human = (
            "User request:\n{user_instructions}\n\n"
            "Current context (may be incomplete):\n{context_json}\n\n"
            "Return JSON only."
        )

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        import json

        resp = (prompt | llm).invoke(
            {
                "user_instructions": user_instructions or "",
                "context_json": json.dumps(context, default=str),
            }
        )
        content = getattr(resp, "content", "") or str(resp)
        plan = _safe_json_loads(content)

        # Normalize minimal shape
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if isinstance(steps, str):
            steps = [steps]
        if not isinstance(steps, list):
            steps = []
        steps = [str(s).strip() for s in steps if str(s).strip()]

        allowed = {
            "list_files",
            "load",
            "merge",
            "sql",
            "wrangle",
            "clean",
            "eda",
            "viz",
            "feature",
            "model",
            "evaluate",
            "mlflow_log",
            "mlflow_tools",
        }
        deduped: list[str] = []
        seen: set[str] = set()
        for s in steps:
            if s in allowed and s not in seen:
                deduped.append(s)
                seen.add(s)

        questions = plan.get("questions") if isinstance(plan, dict) else None
        if isinstance(questions, str):
            questions = [questions]
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if str(q).strip()]

        notes = plan.get("notes") if isinstance(plan, dict) else None
        if isinstance(notes, str):
            notes = [notes]
        if not isinstance(notes, list):
            notes = []
        notes = [str(n).strip() for n in notes if str(n).strip()]

        target_variable = plan.get("target_variable") if isinstance(plan, dict) else None
        if target_variable is not None:
            target_variable = str(target_variable).strip() or None

        # Enforce that model/evaluate require a target variable.
        if any(s in deduped for s in ("model", "evaluate")) and not target_variable:
            # Remove dependent steps and ask for target.
            deduped = [s for s in deduped if s not in ("model", "evaluate")]
            questions.insert(
                0,
                "What is the target column name for modeling/evaluation (e.g., `Churn`)?",
            )

        self.response = {
            "steps": deduped,
            "target_variable": target_variable,
            "questions": questions,
            "notes": notes,
        }
        return None

    def get_plan(self) -> Optional[dict]:
        return self.response
