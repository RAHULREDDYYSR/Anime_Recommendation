import streamlit as st
import time
from graph.graph import app

st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="wide"
)

st.title("🎌 Anime Recommendation System")
st.markdown("Powered by AI and semantic search")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses:
    - **LangGraph** for workflow
    - **LLMs** for query refinement
    - **Semantic Search** for recommendations
    - **Pinecone** vector database
    """)
    
    st.header("Settings")
    num_recommendations = st.slider("Number of recommendations", 1, 10, 3)

# Main interface
user_query = st.text_input(
    "What kind of anime are you looking for?",
    placeholder="e.g., I want a shonen anime with good fights"
)

if st.button("Get Recommendations", type="primary") or user_query:
    if user_query:
        with st.spinner("Finding the perfect anime for you..."):
            start_time = time.time()
            
            try:
                # Update the number of recommendations in the graph
                result = app.invoke({"input_text": user_query})
                
                end_time = time.time()
                
                # Display results
                st.success(f"Found recommendations in {end_time - start_time:.2f} seconds!")
                
                st.subheader("Recommended Anime:")
                
                recommendations = result.get('recommended_anime', [])
                
                if recommendations:
                    for idx, anime in enumerate(recommendations[:num_recommendations], 1):
                        st.markdown(f"**{idx}. {anime}**")
                else:
                    st.warning("No recommendations found. Try a different query!")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to get recommendations!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using LangChain, LangGraph, and Streamlit")
