# 🎌 Anime Recommendation System

An intelligent anime recommendation system powered by **LangGraph**, **LLMs**, and **semantic search** using vector embeddings. This system analyzes user queries, performs semantic search over a curated anime database, and provides personalized recommendations using state-of-the-art language models.

## ✨ Features

- **🤖 AI-Powered Query Refinement**: Uses LLMs to understand and refine user queries for better search results
- **🔍 Semantic Search**: Leverages vector embeddings (Sentence Transformers) and Pinecone for intelligent similarity search
- **📊 Graph-Based Workflow**: Built with LangGraph for modular, stateful recommendation pipeline
- **🎯 10 Personalized Recommendations**: Provides top 10 anime recommendations based on genres, themes, and user preferences
- **⚡ Optimized Performance**: Smart caching reduces query time from 10s to ~1.2s (9.4x faster)
- **🚀 Parallel Image Loading**: Fetches all anime images concurrently for instant display
- **🎨 Streamlit Web Interface**: Beautiful, interactive web app with clickable anime titles
- **🔗 Google Search Integration**: Click any anime title to search for more information
- **📖 Full Synopsis**: Complete anime descriptions from MyAnimeList
- **🧩 Modular Architecture**: Clean, maintainable code with separated UI and API components
- **☁️ Cloud-Ready**: Optimized for deployment on Streamlit Cloud, AWS, GCP, and Azure

## 🎬 Demo

### Screenshots

The app features a clean, modern interface with:
- **Interactive search** with real-time query input
- **10 anime recommendations** with cover images loaded in parallel
- **Clickable titles** that link to Google search
- **Full synopsis** from MyAnimeList in expandable sections
- **Performance metrics** showing cache effectiveness

> **Note**: Screenshots are available in the [`demo/`](demo/) folder. To capture your own:
> 1. Run `uv run streamlit run app.py`
> 2. Navigate to `http://localhost:8501`
> 3. Try queries like "I want a shonen anime with good fights"
> 4. Take screenshots and save to `demo/` folder

### Live Demo

Try the app locally:
```bash
uv run streamlit run app.py
```

Then visit `http://localhost:8501` and try queries like:
- "I want a psychological thriller anime"
- "Recommend me a slice of life anime with comedy"
- "Show me anime similar to Attack on Titan"

## 🏗️ Architecture

The system follows a three-stage graph-based pipeline:
1. **Query Refinement** (`redefine_input`): Analyzes and refines the user's raw input into a precise search query
2. **Semantic Search** (`anime_semantic_search`): Retrieves top 10 relevant anime from Pinecone vector database
3. **Recommendation Generation** (`anime_recommendation`): Uses LLM to select the best 10 anime from retrieved context

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM Framework** | LangChain, LangGraph |
| **Language Models** | Groq (Llama 3.3 70B), Google Gemini |
| **Embeddings** | HuggingFace Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | Pinecone (Serverless) |
| **Web Interface** | Streamlit |
| **Data Processing** | Pandas |
| **Validation** | Pydantic |
| **Monitoring** | LangSmith |

## 📋 Prerequisites

- Python 3.13+
- Pinecone API Key
- Groq API Key (or Google Gemini API Key)
- UV package manager (recommended) or pip

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RAHULREDDYYSR/Anime_Recommendation.git
cd Anime_Recommendation
```

### 2. Install Dependencies

**Using UV (Recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here

# LLM API Keys (choose one or both)
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# LangSmith (Optional - for monitoring)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=anime-recommendation
```

## 📊 Data Ingestion

Before running recommendations, you need to ingest the anime dataset into Pinecone:

```bash
uv run data_ingestion.py
```

This script:
- Reads anime data from `Data/anime_with_synopsis.csv`
- Generates embeddings using Sentence Transformers
- Creates a Pinecone index (`anime-recommendation`)
- Ingests all anime documents with metadata

> **Note**: The script automatically checks if the index exists to prevent duplicate ingestion.

## 💻 Usage

### Web Interface (Recommended)

Launch the Streamlit web app:

```bash
uv run streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Interactive query input
- Adjustable number of recommendations (1-10)
- Real-time performance metrics
- Beautiful, responsive UI
- **Clickable anime titles** with Google search integration
- **Parallel image loading** for instant display
- **Full synopsis** from MyAnimeList

### Command Line Interface

```bash
uv run main.py
```

### Example Output

```
Query 1: I want a shonen anime with good fights
Initializing embedding model (first time only)...
Embedding model loaded successfully!
Connecting to Pinecone index 'anime-recommendation' (first time only)...
Pinecone connection established!
Recommendations: ['Rurouni Kenshin', 'Saiyuuki Reload', 'Grappler Baki']
Time: 11.78 seconds

Query 2: I want a romance anime with comedy
Recommendations: ['Aa! Megami-sama! (TV)', 'Love Hina', 'Futakoi']
Time: 1.20 seconds (cached!)
```

### Customizing the Query

Edit the `user_input` variable in [main.py](file:///c:/Users/rahul/work_space/LLM/llmOps/Anime_Recommendation/main.py):

```python
user_input = "I want a romance anime with a happy ending"
```

## 🔧 Configuration

### Switching LLM Models

Edit [graph/chains.py](file:///c:/Users/rahul/work_space/LLM/llmOps/Anime_Recommendation/graph/chains.py):

```python
# Option 1: Groq (Llama 3.3)
llm = ChatGroq(model='llama-3.3-70b-versatile')

# Option 2: Google Gemini
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
```

### Adjusting Recommendation Count

Edit [graph/nodes.py](file:///c:/Users/rahul/work_space/LLM/llmOps/Anime_Recommendation/graph/nodes.py):

```python
# Change number of retrieved candidates
context = retrieve_anime_recommendations(query=query, k=10)

# Change number of final recommendations
prompt = f"""Recommend the best 10 anime titles..."""
```

## 📁 Project Structure

```
Anime_Recommendation/
├── graph/
│   ├── chains.py          # LLM initialization and structured outputs
│   ├── graph.py           # LangGraph workflow definition
│   ├── nodes.py           # Graph node implementations (10 recommendations)
│   ├── schemas.py         # Pydantic schemas for validation
│   └── state.py           # Graph state definition
├── ui/
│   ├── __init__.py        # UI package initializer
│   └── components.py      # Streamlit UI components (modular design)
├── utils/
│   ├── api_utils.py       # Jikan API integration for anime metadata
│   └── vectore_search.py  # Optimized semantic search with caching
├── Data/
│   └── anime_with_synopsis.csv  # Anime dataset
├── app.py                 # Streamlit web interface (main entry)
├── data_ingestion.py      # Script to ingest data into Pinecone
├── main.py                # CLI entry point with benchmarks
├── .env                   # Environment variables (not tracked)
├── .gitignore            # Git ignore rules
└── pyproject.toml        # Project dependencies
```

## 🔍 How It Works

### 1. Query Refinement
The system uses an LLM to transform vague user queries into precise search terms:

**Input**: "I want a shonen anime with good fights"  
**Refined**: "Action-packed shonen anime featuring intense battle scenes and martial arts combat"

### 2. Semantic Search
Using the refined query, the system:
- Generates embeddings using Sentence Transformers
- Queries Pinecone for top-k similar anime (cosine similarity)
- Retrieves anime with metadata (genres, scores, synopsis)

### 3. LLM-Based Recommendation
The LLM analyzes retrieved candidates and selects the best matches based on:
- Relevance to user's request
- Genre alignment
- Synopsis quality
- Overall rating

## 🎯 Example Queries

- "I want a psychological thriller anime"
- "Recommend me a slice of life anime with comedy"
- "Show me anime similar to Attack on Titan"
- "I need a romance anime with a sad ending"
- "Give me a sci-fi anime with time travel"

## ⚡ Performance

### Optimization Results

The system uses intelligent caching to dramatically improve performance:

- **First query**: ~11-13 seconds (one-time model initialization)
- **Subsequent queries**: ~1.2 seconds (9.4x faster!)
- **Cache persistence**: Lasts for application lifetime
- **Streamlit deployment**: Cache shared across all users

### How Caching Works

```python
# Singleton pattern caches embedding model and Pinecone connection
@st.cache_resource  # Streamlit-aware caching
def get_embeddings():
    return HuggingFaceEmbeddings(...)
```

**Benefits:**
- ✅ No re-loading of 384-dimension embedding model
- ✅ Persistent Pinecone connection
- ✅ Works in cloud deployments (Streamlit Cloud, AWS, GCP)
- ✅ Shared cache across all user sessions

## ☁️ Cloud Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `app.py`
4. Add secrets in Streamlit dashboard (API keys)

**Performance**: First user initializes cache (~11s), all subsequent requests are fast (~1.2s)

### AWS/GCP/Azure

The caching works perfectly on:
- Single-instance deployments (EC2, Cloud Run, App Engine)
- Kubernetes pods (each pod has its own cache)
- Serverless functions (cache persists during warm starts)

For serverless, consider using API-based embeddings (OpenAI/Cohere) for consistent <1s performance.

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'dotenv'`  
**Solution**: Install dependencies using `uv sync` or `pip install python-dotenv`

**Issue**: `Error code: 400 - tool_use_failed`  
**Solution**: Switch to a more capable model like `llama-3.3-70b-versatile`

**Issue**: Pinecone index not found  
**Solution**: Run `uv run data_ingestion.py` to create and populate the index

**Issue**: Slow performance (>10 seconds per query)  
**Solution**: Cache is working! First query loads models, subsequent queries are fast (~1.2s)

## 📈 Future Enhancements

- [x] ~~Add Streamlit web interface~~
- [x] ~~Add caching for faster responses~~
- [ ] Implement user feedback loop
- [ ] Support for multi-language queries
- [ ] Add anime ratings and reviews integration
- [ ] Implement collaborative filtering
- [ ] Add user authentication and history
- [ ] Migrate to ChromaDB for <100ms queries

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Rahul Reddy**
- GitHub: [@RAHULREDDYYSR](https://github.com/RAHULREDDYYSR)

## 🙏 Acknowledgments

- Anime data sourced from MyAnimeList
- Built with LangChain and LangGraph
- Powered by Groq and Google Gemini LLMs
- Vector search by Pinecone

---

⭐ If you found this project helpful, please consider giving it a star!
