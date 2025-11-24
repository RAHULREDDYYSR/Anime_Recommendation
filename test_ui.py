import streamlit as st
from ui.components import render_custom_css, render_anime_card_with_image

st.set_page_config(layout="wide")

render_custom_css()

st.title("UI Component Test")

# Dummy data
anime_info = {
    "image_url": "https://cdn.myanimelist.net/images/anime/1223/96541.jpg",
    "score": "9.14",
    "episodes": "64",
    "year": "2009",
    "synopsis": "Fullmetal Alchemist: Brotherhood is a 2009 Japanese anime television series adapted from the original Fullmetal Alchemist manga series by Hiromu Arakawa."
}

st.header("Modern Anime Card")
render_anime_card_with_image(1, "Fullmetal Alchemist: Brotherhood", anime_info)

st.header("Card without Info")
render_anime_card_with_image(2, "Unknown Anime", None)
