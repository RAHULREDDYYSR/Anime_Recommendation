from typing import Optional, TypedDict, List

class GraphState(TypedDict):
    '''Defines the state for our LangGraph workflow'''
    input_text: str
    redefine_input_content: str
    recommended_anime: List[str]
    context: List[str]