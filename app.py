import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from PIL import Image
import base64

st.set_page_config(page_title="Grant Matcher", layout="centered")

# Load logo
logo_path = "logo.png"
logo_html = f"""
<div style='text-align:center;'>
    <img src='data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}' 
         style='width: 20%; max-width: 200px; height: auto;'>
</div>
"""
st.markdown(logo_html, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu", trust_remote_code=True)

model = load_model()

@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/iati-dataset/IATI-updated.csv"
    df = pd.read_csv(url)
    #df.columns = df.columns.str.strip().str.lower()
    
    df['recipient-country'] = df['recipient-country'].astype(str).str.lower()
    df['sector'] = df['sector'].astype(str).str.lower()
    df['description append'] = df['Description Append'].astype(str)
    df['reporting-org-type'] = df['reporting-org-type'].astype(str).str.strip()
    df['total-commitment-usd'] = pd.to_numeric(df['total-Commitment-USD'], errors='coerce')
    df['start-actual'] = pd.to_datetime(df['start-actual'], errors='coerce')
    important_cols = ['recipient-country', 'sector', 'description append', 'total-commitment-usd', 'start-actual', 'reporting-org-type']
    df = df.dropna(subset=important_cols)
    for col in ['recipient-country', 'sector', 'Description Append', 'reporting-org-type']:
        df = df[df[col].str.strip() != '']
    df['sector_list'] = df['sector'].str.split(';').apply(lambda x: [i.strip() for i in x])
    df = df.explode('sector_list')
    return df

df = load_data()

@st.cache_resource
def build_nn(df):
    embeddings = np.vstack(df['Description Append'].apply(lambda x: model.encode(x, convert_to_numpy=True)))
    nn_model = NearestNeighbors(n_neighbors=5000, metric='cosine')
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
    avg_amount = df_org['total-Commitment-USD'].mean()
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
        if not (amount_min <= row['total-Commitment-USD'] <= amount_max):
            continue

        org = row['reporting-org']
        if org not in org_scores:
            org_scores[org] = {
                'country_count': 0,
                'amount_count': 0,
                'sector_match': 0,
                'similarities': [],
                'rows': []
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
            funding_score = min(info['amount_count'] / 10, 1)
            country_score = min(info['country_count'] / 10, 1) if country_input.strip() else 0
            sector_score = min(info['sector_match'] / 10, 1) if sector_input.strip() else 0
            description_score = np.mean(info['similarities'])
            avg_grant = np.mean([r['total-Commitment-USD'] for r in info['rows']])
            avg_grant_score = avg_grant / 10_000_000
            project_count_score = len(info['rows']) / 1000
            score = (
                (0.4 * funding_score) +
                (0.3 * country_score) +
                (0.1 * sector_score) +
                (0.2 * description_score) -
                (0.2 * avg_grant_score) -
                (0.2 * project_count_score)
            )
            final_orgs.append((org, score, info['rows']))

    final_orgs.sort(key=lambda x: x[1], reverse=True)
    return final_orgs[:10]

# UI
st.markdown("### Grant Matcher: Find Matching Funders and Their Projects")

amount_range = st.slider("Select grant amount range (USD)", 0, 10000000, (50000, 1000000))
country_input = st.text_input("Enter recipient country/region(s) (comma separated) — Leave empty for all countries")
sector_input = st.text_input("Enter sector(s) (comma separated) — Leave empty for all sectors")
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
    with st.spinner("Matching your project..."):
        matches = match_projects(description, amount_range[0], amount_range[1],
                                 country_input, sector_input, selected_types)

    if not matches:
        st.warning("⚠️ No matches found. Try adjusting your filters.")
    else:
        st.subheader("✅ Top Matching Organizations and Their Projects")
        for org_name, score, rows in matches:
            filtered_rows = [r for r in rows if pd.notna(r['reporting-org']) and pd.notna(r['Description Append'])]
            if not filtered_rows:
                continue
            org_row = filtered_rows[0]
            org_display = f"[{org_name}]({org_row['publisher_url']})" if pd.notna(org_row['publisher_url']) else org_name

            st.markdown(f"### Organization: {org_display}")
            st.markdown(f"**Focus Area:**\n\n{summarize_focus_area(df[df['reporting-org'] == org_name])}")
            st.markdown(f"**Matched Project Description:**\n\n{org_row['Description Append']}")
            st.markdown(f"**Country:** {org_row['recipient-country'].title()}")
            st.markdown(f"**Sector:** {org_row['sector_list'].title()}")
            st.markdown(f"**Funding Amount:** ${org_row['total-Commitment-USD']:,.0f}")
            st.markdown("---")

# Footer banner
banner_path = "banner.png"
banner_html = f"""
<div style='text-align:center; margin-top: 50px;'>
    <a href='https://www.econbook.biz' target='_blank'>
        <img src='data:image/png;base64,{base64.b64encode(open(banner_path, "rb").read()).decode()}' 
             style='width: 100%; max-width: 800px; height: auto;'>
    </a>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

# Disclaimer
st.markdown("""
---
<small>
**Technical Disclaimer: Data & Model Use**

This tool was created in response to the defunding of the United States Agency for International Development (USAID) to help NGOs identify alternative funders that have historically supported similar types of projects.

The tool leverages a Large Language Model (LLM) enhanced with vector similarity search. The underlying dataset is compiled from publicly available records published through the International Aid Transparency Initiative (IATI), available at <https://iatistandard.org/en/>.

Funders are ranked based on the number of past projects in the dataset that closely match the user's project description.

If your NGO is not currently represented and you would like to be included in future versions of this tool, please complete our [submission form](https://www.virunga.ai/submission).
</small>
""", unsafe_allow_html=True)
