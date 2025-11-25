from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import RecommendedAnime, RefinedQuery
from langchain_openai import ChatOpenAI
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
#llm = ChatOpenAI(model='gpt-4.1-nano')
#llm = ChatGroq(model='llama-3.3-70b-versatile')
# create structured LLMs for specific tasks
redefine_input_llm = llm.with_structured_output(RefinedQuery)
recommended_Anime_llm = llm.with_structured_output(RecommendedAnime)