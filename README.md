# Grant Matcher - Professional Final Version

This application helps users find the best matching funders and historical grants based on:

- Project description (semantic search using FAISS)
- Sector (fuzzy matching)
- Recipient countries (fuzzy matching)
- Grant amount range

---

## 🚀 Features

✅ FAISS fast vector search for massive datasets (>50M grants)  
✅ Detailed "Focus Area" summaries for each organization (sectors, countries, keywords, amounts)  
✅ Detailed "Why Match" explanations based on sectors, country, amount, and description similarity  
✅ Flexible matching even if sector or country slightly differ  
✅ Automatic fallback to most similar historical projects if no exact matches  
✅ Optimized for CPU usage (no GPU required)

---

## 📦 How to Run Locally

1. Clone or download the project files:
    - `app.py`
    - `requirements.txt`
    - `grants_data.csv` (your full grant dataset)

2. Install required packages:

```bash
pip install -r requirements.txt
streamlit run app.py
#   A p p G r a n t s  
 #   A p p G r a n t s  
 