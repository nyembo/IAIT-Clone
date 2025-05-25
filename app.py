import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from PIL import Image
import base64

st.set_page_config(page_title="Grant Matcher", layout="centered")

# Inject Hugging Face token
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Load logo safely
logo_path = "logo.png"
try:
    with open(logo_path, "rb") as f:
        logo_html = f"""
        <div style='text-align:center;'>
            <img src='data:image/png;base64,{base64.b64encode(f.read()).decode()}' 
                 style='width: 20%; max-width: 200px; height: auto;'>
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ logo.png not found. Skipping logo.")

@st.cache_resource
def load_model():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    return SentenceTransformer(model_name, cache_folder="./hf_model_cache", device="cpu", trust_remote_code=True)

model = load_model()

@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/iati-dataset/IATI-updated.csv"
    try:
        df = pd.read_csv(url, engine="python", encoding="utf-8")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.lower()

    required_cols = ['recipient-country', 'sector', 'description append',
                     'total-commitment-usd', 'start-actual', 'reporting-org-type']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"🚫 Required column missing: `{col}`")
            return pd.DataFrame()

    df['recipient-country'] = df['recipient-country'].astype(str).str.lower()
    df['sector'] = df['sector'].astype(str).str.lower()
    df['description append'] = df['description append'].astype(str)
    df['reporting-org-type'] = df['reporting-org-type'].astype(str).str.strip()
    df['total-commitment-usd'] = pd.to_numeric(df['total-commitment-usd'], errors='coerce')
    df['start-actual'] = pd.to_datetime(df['start-actual'], errors='coerce')

    df = df.dropna(subset=required_cols)
    for col in ['recipient-country', 'sector', 'description append', 'reporting-org-type']:
        df = df[df[col].str.strip() != '']

    df['sector_list'] = df['sector'].str.split(';').apply(lambda x: [i.strip() for i in x])
    df = df.explode('sector_list')
    return df

df = load_data()

if df.empty:
    st.error("❌ Dataset could not be loaded or cleaned.")
    st.stop()

@st.cache_resource
def build_nn(df):
    texts = df['description append'].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    nn_model = NearestNeighbors(n_neighbors=min(5000, len(df)), metric='cosine')
    nn_model.fit(embeddings)
    return nn_model, embeddings

nn_model, embeddings = build_nn(df)

def match_fuzzy(input_text, choices, threshold=65):
    input_list = [x.strip().lower() for x in input_text.split(',') if x.strip()]
    matched = set()
    for item in input_list:
        for choice in choices:
            if fuzz.partial_ratio(item, choice) >= threshold:
                matched.add(choice)
    return list(matched)

def summarize_focus_area(df_org):
    if df_org.empty:
        return "Not enough information."
    sectors = df_org['sector_list'].value_counts().head(3).index.tolist()
    countries = [c.title() for c in df_org['recipient-country'].value_counts().head(3).index.tolist()]
    avg_amount = df_org['total-commitment-usd'].mean()
    return (
        f"Focuses on sectors: {', '.join(sectors)}.\n"
        f"Operates mainly in: {', '.join(countries)}.\n"
        f"Average funding: ${avg_amount:,.0f}."
    )

def match_projects(description, amount_min, amount_max, country_input, sector_input, selected_types):
    query_vec = model.encode(description, convert_to_numpy=True)
    D, I = nn_model.kneighbors([query_vec], n_neighbors=5000)
    D = 1 - D  # Convert cosine distance to similarity

    matched_countries = match_fuzzy(country_input, df['recipient-country'].unique()) if country_input.strip() else []
    matched_sectors = match_fuzzy(sector_input, df['sector_list'].unique()) if sector_input.strip() else []

    org_scores = {}
    for pos, idx in enumerate(I[0]):
        row = df.iloc[idx]
        if row['reporting-org-type'] not in selected_types:
            continue
        if country_input.strip() and row['recipient-country'] not in matched_countries:
            continue
        if sector_input.strip() and row['sector_list'] not in matched_sectors:
            continue
        if not (amount_min <= row['total-commitment-usd'] <= amount_max):
            continue

        org = row['reporting-org']
        if org not in org_scores:
            org_scores[org] = {
                'country_count': 0, 'amount_count': 0, 'sector_match': 0,
                'similarities': [], 'rows': []
            }

        org_scores[org]['amount_count'] += 1
        if country_input.strip():
            org_scores[org]['country_count'] += 1
        if sector_input.strip():
            org_scores[org]['sector_match'] += 1
        org_scores[org]['similarities'].append(D[0][pos])
        org_scores[org]['rows'].append(row)

    final_orgs = []
    for org, info in org_scores.items():
        if info['amount_count'] >= 3:
            score = (
                0.4 * min(info['amount_count'] / 10, 1) +
                0.3 * min(info['country_count'] / 10, 1) +
                0.1 * min(info['sector_match'] / 10, 1) +
                0.2 * np.mean(info['similarities']) -
                0.2 * (np.mean([r['total-commitment-usd'] for r in info['rows']]) / 10_000_000) -
                0.2 * (len(info['rows']) / 1000)
            )
            final_orgs.append((org, score, info['rows']))

    final_orgs.sort(key=lambda x: x[1], reverse=True)
    return final_orgs[:10]

# UI
st.markdown("### Grant Matcher: Find Matching Funders and Their Projects")

amount_range = st.slider("Select grant amount range (USD)", 0, 10000000, (50000, 1000000))
country_input = st.text_input("Enter recipient country/region(s) (comma separated)")
sector_input = st.text_input("Enter sector(s) (comma separated)")
description = st.text_area("Describe your project")

all_types = sorted(df['reporting-org-type'].dropna().unique())
if 'select_all_orgs' not in st.session_state:
    st.session_state.select_all_orgs = True
if 'selected_org_types' not in st.session_state:
    st.session_state.selected_org_types = all_types

select_all = st.checkbox("Select all organization types", value=st.session_state.select_all_orgs)
selected_types = st.multiselect("Filter by organization type", all_types,
                                default=all_types if select_all else st.session_state.selected_org_types)

st.session_state.select_all_orgs = select_all
st.session_state.selected_org_types = selected_types

if st.button("Find Matches"):
    with st.spinner("🔎 Matching your project..."):
        matches = match_projects(description, amount_range[0], amount_range[1],
                                 country_input, sector_input, selected_types)

    if not matches:
        st.warning("⚠️ No matches found. Try adjusting your filters.")
    else:
        st.subheader("✅ Top Matching Organizations and Their Projects")
        for org_name, score, rows in matches:
            org_row = next((r for r in rows if pd.notna(r['reporting-org']) and pd.notna(r['description append'])), None)
            if not org_row:
                continue
            org_display = f"[{org_name}]({org_row['publisher_url']})" if pd.notna(org_row.get('publisher_url')) else org_name

            st.markdown(f"### Organization: {org_display}")
            st.markdown(f"**Focus Area:**\n\n{summarize_focus_area(df[df['reporting-org'] == org_name])}")
            st.markdown(f"**Matched Project Description:**\n\n{org_row['description append']}")
            st.markdown(f"**Country:** {org_row['recipient-country'].title()}")
            st.markdown(f"**Sector:** {org_row['sector_list'].title()}")
            st.markdown(f"**Funding Amount:** ${org_row['total-commitment-usd']:,.0f}")
            st.markdown("---")

# Footer banner
banner_path = "banner.png"
try:
    with open(banner_path, "rb") as f:
        banner_html = f"""
        <div style='text-align:center; margin-top: 50px;'>
            <a href='https://www.econbook.biz' target='_blank'>
                <img src='data:image/png;base64,{base64.b64encode(f.read()).decode()}' 
                     style='width: 100%; max-width: 800px; height: auto;'>
            </a>
        </div>
        """
        st.markdown(banner_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ banner.png not found. Skipping footer banner.")

# Disclaimer
st.markdown("""
---
<small>
**Technical Disclaimer: Data & Model Use**

This tool helps NGOs identify funders based on historical projects from IATI data.  
Funders are ranked by how closely their past projects match your description.

Learn more or submit your NGO for inclusion at [virunga.ai](https://www.virunga.ai/submission).
</small>
""", unsafe_allow_html=True)
