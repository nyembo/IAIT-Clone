import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from PIL import Image
import base64

st.set_page_config(page_title="Grant Matcher", layout="centered")

# Hugging Face token injection
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

@contextmanager
def show_traceback_on_streamlit():
    try:
        yield
    except Exception as e:
        st.error(f"\u274c App crashed: {e}")
        st.text("Traceback:")
        st.text("".join(traceback.format_exception(*sys.exc_info())))
        st.stop()

def main():
    raise ValueError("This is a forced test crash to validate diagnostics")

# Entry point
with show_traceback_on_streamlit():
    main()
