
from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    redefine_input, anime_recommendation, anime_semantic_search
)

graph = StateGraph(GraphState)
graph.add_node('redefine_input', redefine_input)
graph.add_node('anime_recommendation', anime_recommendation)
graph.add_node('anime_semantic_search', anime_semantic_search)

graph.set_entry_point('redefine_input')
graph.add_edge('redefine_input', "anime_semantic_search")
graph.add_edge('anime_semantic_search', "anime_recommendation")
graph.add_edge('anime_recommendation',END)

app = graph.compile()
