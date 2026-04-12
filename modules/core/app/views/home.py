import streamlit as st
import os
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from utils.streamlit_helper import get_user_info
from genesis_workbench.workbench import get_user_settings

st.title(":material/home: Home")

user_info = get_user_info()

if "home_user_settings" not in st.session_state:
    with st.spinner("Getting user information"):
        st.session_state["home_user_settings"] = get_user_settings(user_email=user_info.user_email)

user_settings = st.session_state["home_user_settings"]

doc_index = st.session_state.get("doc_index", [])

if "setup_done" not in user_settings or user_settings["setup_done"] != "Y":
    st.warning("#### :warning: Your profile setup is incomplete. Finish your setup using the **Profile** tab.")
elif "mlflow_access_verified" not in st.session_state:
    mlflow_folder = user_settings.get("mlflow_experiment_folder", "")
    if mlflow_folder:
        try:
            w = WorkspaceClient()
            mlflow_base_path = f"Users/{user_info.user_email}/{mlflow_folder}"
            w.workspace.mkdirs(f"/Workspace/{mlflow_base_path}")
            experiment_path = f"/{mlflow_base_path}/__test__"
            mlflow.set_registry_uri("databricks-uc")
            mlflow.set_tracking_uri("databricks")
            experiment = mlflow.set_experiment(experiment_path)
            mlflow.delete_experiment(experiment_id=experiment.experiment_id)
            st.session_state["mlflow_access_verified"] = True
        except Exception:
            st.session_state["mlflow_access_verified"] = False
    else:
        st.session_state["mlflow_access_verified"] = False

if st.session_state.get("mlflow_access_verified") is False:
    st.error(
        "**MLflow Experiment Access Error**\n\n"
        "The application cannot access your MLflow experiment folder. "
        "This may be because the folder was deleted or the application's service principal lost permissions.\n\n"
        "Please go to the **Profile** tab to reconfigure your MLflow experiment folder."
    )


def _build_doc_context(docs):
    """Build a compact summary of available workflows for the LLM prompt."""
    lines = []
    for doc in docs:
        title = doc["title"]
        intro = ""
        in_intro = False
        for line in doc["content"].splitlines():
            if line.startswith("# ") and not in_intro:
                in_intro = True
                continue
            if line.startswith("## "):
                break
            if in_intro:
                intro += line + " "
        lines.append(f"- **{title}**: {intro.strip()}")
    return "\n".join(lines)


def _ask_assistant(user_query, doc_context):
    """Call Claude Sonnet on Databricks to suggest workflows for the user's query."""
    endpoint = os.environ.get("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-6")

    system_prompt = f"""You are a helpful assistant for Genesis Workbench, a bioinformatics platform.
Based on the user's request, suggest which workflows they should use and briefly explain why.
If the request doesn't match any available workflow, say so politely and suggest they check back later as new workflows are added regularly.

Keep your response concise — use bullet points with workflow names in bold.

Available workflows:
{doc_context}"""

    try:
        print(f"[Assistant] Querying endpoint: {endpoint}")
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=ChatMessageRole.USER, content=user_query),
            ],
            max_tokens=512,
        )
        print(f"[Assistant] Response received")
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Assistant] Exception: {e}")
        return None


_EXAMPLE_QUESTIONS = [
    "Run a GWAS analysis",
    "Predict a protein structure",
    "Dock a molecule to a protein",
    "Annotate genetic variants",
    "Design a protein binder",
    "Analyze single cell data",
]

# ── Tabs ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 0.6rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

assistant_tab, search_tab = st.tabs([
    ":material/assistant: What do you want to do?",
    ":material/search: Search Documentation",
])

# ── What do you want to do? ───────────────────────────────────────────────

with assistant_tab:
    st.markdown("")  # spacer between tabs and content
    st.markdown("")  # spacer between tabs and content
    st.markdown("I can guide you. Ask me how to do anything in this application!")
    @st.fragment
    def _assistant_fragment():
        user_query = st.text_input(
            "what_do_you_want",
            value=st.session_state.get("assistant_query", ""),
            placeholder="Type your question...",
            label_visibility="collapsed",
            key="assistant_input",
        )

        selected_example = st.pills(
            "Examples",
            _EXAMPLE_QUESTIONS,
            label_visibility="collapsed",
            selection_mode="single",
            key="example_pills",
        )

        active_query = user_query.strip()
        if not active_query and selected_example:
            active_query = selected_example

        if active_query and active_query != st.session_state.get("_last_assistant_query"):
            st.session_state["_last_assistant_query"] = active_query
            with st.spinner("Thinking..."):
                answer = _ask_assistant(active_query, _build_doc_context(doc_index))
            st.session_state["_last_assistant_answer"] = answer

        answer = st.session_state.get("_last_assistant_answer")
        if active_query and answer:
            st.markdown(answer)
        elif active_query and answer is None:
            st.info(
                "I'm not able to process your request right now. "
                "The AI assistant endpoint may not be configured. "
                "Please check back later or browse the documentation."
            )

    _assistant_fragment()

# ── Search Documentation ──────────────────────────────────────────────────

with search_tab:
    st.markdown("")  # spacer between tabs and content
    st.markdown("")  # spacer between tabs and content

    doc_query = st.text_input(
        "doc_search",
        value="",
        placeholder="Search workflows, methods, inputs...",
        label_visibility="collapsed",
        key="doc_search_input",
    )

    if doc_query.strip():
        words = doc_query.strip().lower().split()
        results = [
            doc for doc in doc_index
            if all(w in doc["title"].lower() or w in doc["content"].lower() for w in words)
        ]
        if results:
            for r in results:
                with st.expander(r["title"]):
                    st.markdown(r["content"])
        else:
            st.caption("No results found.")
    else:
        st.caption(f"{len(doc_index)} workflow documents available. Type to search.")
