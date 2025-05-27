import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import base64
import os

# === Logo ===
def load_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Choose logo based on theme
theme = st.get_option("theme.base")  # 'light' or 'dark'
logo_path = "logo.png" if theme == "light" else "logo.png"
logo_base64 = load_image_base64(logo_path)

if logo_base64:
    st.markdown(f"""
    <div style='text-align:center;'>
        <img src='data:image/png;base64,{logo_base64}' style='width: 20%; max-width: 200px; height: auto;'>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning(f"⚠️ Logo not found: {logo_path}")

# === Load model ===
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu", trust_remote_code=True)

model = load_model()

# === Load data ===
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("https://storage.googleapis.com/iati-dataset/IATI_updated_compress.csv")
    df['recipient-country'] = df['recipient-country'].astype(str).str.lower()
    df['sector'] = df['sector'].astype(str).str.lower()
    df['Description Append'] = df['Description Append'].astype(str)
    df['reporting-org-type'] = df['reporting-org-type'].astype(str).str.strip()
    df['total-Commitment-USD'] = pd.to_numeric(df['total-Commitment-USD'], errors='coerce')
    df['start-actual'] = pd.to_datetime(df['start-actual'], errors='coerce')
    important_cols = ['recipient-country', 'sector', 'Description Append', 'total-Commitment-USD', 'start-actual', 'reporting-org-type']
    df = df.dropna(subset=important_cols)
    for col in ['recipient-country', 'sector', 'Description Append', 'reporting-org-type']:
        df = df[df[col].str.strip() != '']
    df['sector_list'] = df['sector'].str.split(';').apply(lambda x: [i.strip() for i in x])
    df = df.explode('sector_list')
    return df

df = load_data()

# === Build FAISS index ===
@st.cache_resource(show_spinner=False)
def build_faiss(df):
    embeddings = np.vstack(df['Description Append'].apply(lambda x: model.encode(x, convert_to_numpy=True)))
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_faiss(df)

# === Utility Functions ===
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
        f"Countries where it operates most: {', '.join(countries)}.\n"
        f"Average funding: ${avg_amount:,.0f}."
    )

# === Matching logic ===
def match_projects(description, amount_min, amount_max, country_input, sector_input, selected_types):
    query_vec = model.encode(description, convert_to_numpy=True)
    faiss.normalize_L2(np.expand_dims(query_vec, axis=0))
    D, I = index.search(np.expand_dims(query_vec, axis=0), 5000)

    matched_countries = match_fuzzy(country_input, df['recipient-country'].unique()) if country_input.strip() else []
    matched_sectors = match_fuzzy(sector_input, df['sector_list'].unique()) if sector_input.strip() else []

    org_scores = {}
    for idx in I[0]:
        if idx == -1:
            continue
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
        org_scores[org]['similarities'].append(D[0][np.where(I[0] == idx)[0][0]])
        org_scores[org]['rows'].append(row)

    final_orgs = []
    for org, info in org_scores.items():
        if info['amount_count'] >= 3:
            score = (
                (0.4 * min(info['amount_count'] / 10, 1)) +
                (0.3 * min(info['country_count'] / 10, 1) if country_input.strip() else 0) +
                (0.1 * min(info['sector_match'] / 10, 1) if sector_input.strip() else 0) +
                (0.2 * np.mean(info['similarities'])) -
                (0.2 * (np.mean([r['total-Commitment-USD'] for r in info['rows']]) / 10_000_000)) -
                (0.2 * len(info['rows']) / 1000)
            )
            final_orgs.append((org, score, info['rows']))

    final_orgs.sort(key=lambda x: x[1], reverse=True)
    return final_orgs[:10]

# === UI ===
st.markdown("### Find Funders For Your Development Project")

amount_range = st.slider("Select grant amount range (USD)", 0, 100_000_000, (50_000, 100_000_000))
st.markdown(f"**Selected range:** ${amount_range[0]:,} – ${amount_range[1]:,}")
country_input = st.text_input("Enter recipient country/region(s) or leave blank to see all (comma separated)")
sector_input = st.text_input("Enter sector(s) or leave blank to see all (comma separated)")
description = st.text_area("Describe your project")

all_types = sorted(df['reporting-org-type'].dropna().unique())
if 'select_all_orgs' not in st.session_state:
    st.session_state.select_all_orgs = True
if 'selected_org_types' not in st.session_state:
    st.session_state.selected_org_types = all_types

select_all = st.checkbox("Select all organization types", value=st.session_state.select_all_orgs)
selected_types = st.multiselect("Filter by organization type", all_types, default=all_types if select_all else st.session_state.selected_org_types)

st.session_state.select_all_orgs = select_all
st.session_state.selected_org_types = selected_types

if st.button("Find Matches"):
    with st.spinner("Matching your project..."):
        matches = match_projects(description, amount_range[0], amount_range[1], country_input, sector_input, selected_types)

    if not matches:
        st.warning("⚠️ No matches found. Try adjusting your filters.")
    else:
        st.subheader("The organization that funds the most projects similar to yours is:")
        for org_name, score, rows in matches:
            filtered_rows = [r for r in rows if pd.notna(r['reporting-org']) and pd.notna(r['Description Append'])]
            if not filtered_rows:
                continue
            org_row = filtered_rows[0]
            org_display = f"[{org_name}]({org_row['publisher_url']})" if pd.notna(org_row.get('publisher_url')) else org_name

            st.markdown(f"### Organization: {org_display}")
            st.markdown(f"**Focus Area:**\n{summarize_focus_area(df[df['reporting-org'] == org_name])}")
            st.markdown(f"**Example of a similar project:**\n{org_row['Description Append']}")
            st.markdown(f"**Country:** {org_row['recipient-country'].title()} | **Sector:** {org_row['sector_list'].title()}")
            st.markdown(f"**Funding Amount:** ${org_row['total-Commitment-USD']:,.0f}")
            st.markdown("---")
            

# Responsive footer banner
banner_path = "banner.png"
banner_html = f"""
<div style='text-align:center; margin-top: 50px;'>
    <a href='https://www.econbook.biz' target='_blank'><img src='data:image/png;base64,{base64.b64encode(open(banner_path, "rb").read()).decode()}' 
         style='width: 100%; max-width: 800px; height: auto;'>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

# Footnote explanation
st.markdown("""
---
<small>
**Technical Disclaimer: Data & Model Use**

This tool was created in response to the defunding of the United States Agency for International Development (USAID) to help NGOs identify alternative funders that have historically supported similar types of projects. By analyzing historical grantmaking patterns, it aims to guide organizations toward relevant funding sources based on project similarity.

The tool leverages a Large Language Model (LLM) enhanced with vector similarity search, powered by Facebook AI Similarity Search (FAISS). The underlying dataset is compiled from publicly available records published through the International Aid Transparency Initiative (IATI), available at <https://iatistandard.org/en/>.

Funders are ranked based on the number of past projects in the dataset that closely match the user's project description. The more relevant historical projects a funder has supported, the higher they are ranked in the results. This approach is designed to surface organizations with a demonstrated history of funding work similar to yours.

While every effort has been made to ensure the reliability of the data, completeness and accuracy are ultimately dependent on the original information published by reporting organizations. Some records may be outdated, incomplete, or inconsistently formatted.

If your NGO is not currently represented and you would like to be included in future versions of this tool—or if you need support preparing your data for inclusion—please complete our submission form [submission form](https://www.virunga.ai/submission).
Contact us as well if you would like assistance being included in the IATI dataset itself.

**Disclaimer:** This tool is provided for informational purposes only. It does not constitute legal, financial, or strategic advice, nor does it guarantee access to funding. Users are encouraged to validate all outputs and consult directly with funding organizations before acting on any recommendations.
</small>
""", unsafe_allow_html=True)