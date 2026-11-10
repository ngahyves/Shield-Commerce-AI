"""LangGraph assembly: router → (fast_report | retrieval → investigator → reporter).

Three paths:
- fast_approve  → fast_report → END  (~0 ms)
- fast_block    → fast_report → END  (~0 ms)
- full_investigation → retrieval → investigator → reporter → END  (~6.7 s)

The router is a conditional edge based on `state["route"]`.
"""

from langgraph.graph import START, END, StateGraph

from src.agents.fast_report import fast_report_node
from src.agents.investigator import investigator_node
from src.agents.reporter import reporter_node
from src.agents.retrieval import retrieval_node
from src.agents.router import router_node
from src.agents.state import AgentState
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


def _route_fn(state: AgentState) -> str:
    """Read the route from state (written by router_node)."""
    return state["route"]


def build_graph():
    """Compile the LangGraph state machine."""
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("fast_report", fast_report_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("investigator", investigator_node)
    g.add_node("reporter", reporter_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route_fn,
        {
            "fast_approve": "fast_report",
            "fast_block": "fast_report",
            "full_investigation": "retrieval",
        },
    )
    g.add_edge("fast_report", END)
    g.add_edge("retrieval", "investigator")
    g.add_edge("investigator", "reporter")
    g.add_edge("reporter", END)

    compiled = g.compile()
    log.info("graph compiled: %d nodes", len(g.nodes))
    return compiled