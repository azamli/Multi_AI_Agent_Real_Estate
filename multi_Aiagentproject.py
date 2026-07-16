import os
import streamlit as st
import json
import time
import re
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import Groq

# Load environment variables
load_dotenv()

# Keys loaded directly from .env - no UI input fields
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# ---------------------------------------------
# Pydantic Models
# ---------------------------------------------

class PropertyDetails(BaseModel):
    address: str = Field(description="Full property address")
    price: Optional[str] = Field(default=None, description="Property price")
    bedrooms: Optional[str] = Field(default=None, description="Number of bedrooms")
    bathrooms: Optional[str] = Field(default=None, description="Number of bathrooms")
    square_feet: Optional[str] = Field(default=None, description="Square footage")
    property_type: Optional[str] = Field(default=None, description="Type of property")
    description: Optional[str] = Field(default=None, description="Property description")
    listing_url: Optional[str] = Field(default=None, description="Original listing URL")

class PropertyListing(BaseModel):
    properties: List[PropertyDetails] = Field(description="List of properties found")
    total_count: int = Field(description="Total number of properties found")
    source_website: str = Field(description="Website where properties were found")

# ---------------------------------------------
# Groq Client
# ---------------------------------------------

def get_groq_client(api_key: str):
    return Groq(api_key=api_key)

def groq_chat(client, system_prompt: str, user_message: str, model: str = "llama-3.3-70b-versatile") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=2000,
        temperature=0.7
    )
    return response.choices[0].message.content

# ---------------------------------------------
# Direct Firecrawl Agent
# ---------------------------------------------

class DirectFirecrawlAgent:
    def __init__(self, firecrawl_api_key: str, groq_client, model_name: str):
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
        self.client = groq_client
        self.model_name = model_name

    def find_properties_direct(self, city: str, state: str, user_criteria: dict, selected_websites: list) -> dict:
        city_formatted = city.replace(" ", "-").lower()
        state_upper = state.upper() if state else ""
        state_lower = state.lower() if state else ""
        city_trulia = city.replace(" ", "_").lower()

        search_urls = {
            "Zillow": f"https://www.zillow.com/homes/for_sale/{city_formatted}-{state_upper}/",
            "Realtor.com": f"https://www.realtor.com/realestateandhomes-search/{city_formatted}_{state_upper}/pg-1",
            "Trulia": f"https://www.trulia.com/{state_upper}/{city_trulia}/",
            "Homes.com": f"https://www.homes.com/homes-for-sale/{city_formatted}-{state_lower}/"
        }

        urls_to_search = {site: url for site, url in search_urls.items() if site in selected_websites}
        all_properties = []

        for site_name, url in urls_to_search.items():
            try:
                result = self.firecrawl.scrape(
                    url,
                    formats=["markdown"],
                    only_main_content=True,
                    timeout=30000
                )

                markdown_content_raw = ""
                if isinstance(result, dict):
                    markdown_content_raw = result.get("markdown", "")
                elif hasattr(result, "markdown"):
                    markdown_content_raw = result.markdown

                if markdown_content_raw:
                    markdown_content = markdown_content_raw[:8000]

                    system_prompt = """You are a real estate data extraction expert.
Extract ALL property listings from the provided web content.
Return ONLY valid JSON in this exact format, nothing else:
{
  "properties": [],
  "total_count": 0,
  "source_website": ""
}"""

                    user_msg = f"""Extract property listings from {site_name} for {city}, {state}

Content:
{markdown_content}
"""

                    response_text = groq_chat(self.client, system_prompt, user_msg, self.model_name)

                    # clean JSON
                    response_text = response_text.strip()
                    if response_text.startswith("```"):
                        response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
                        response_text = re.sub(r'\n?```$', '', response_text)

                    site_data = json.loads(response_text)

                    props = site_data.get("properties", [])
                    for p in props:
                        p["source"] = site_name

                    all_properties.extend(props)

            except Exception as e:
                st.warning(f"Could not scrape {site_name}: {str(e)[:120]}")
                continue

        return {
            "properties": all_properties,
            "total_count": len(all_properties),
            "sources": list(urls_to_search.keys())
        }

# ---------------------------------------------
# Sequential Analysis Agents
# ---------------------------------------------

def run_market_analysis(client, model_name: str, properties_data: dict, city: str, state: str) -> str:
    system_prompt = """You are a real estate market analysis expert.
Provide CONCISE market insights based on the property data provided.
Use bullet points and keep each section under 100 words.
Cover: Market Condition (buyer's/seller's), Key Neighborhoods, and Investment Outlook."""

    user_msg = f"""Analyze the real estate market for {city}, {state} based on these {properties_data['total_count']} properties found:

{json.dumps(properties_data['properties'][:10], indent=2)}

Provide:
1. Market Condition: Buyer's/seller's market, price trends
2. Key Neighborhoods: Brief overview of areas where properties are located
3. Investment Outlook: 2-3 key points about investment potential"""

    return groq_chat(client, system_prompt, user_msg, model_name)

def run_property_valuation(client, model_name: str, properties_data: dict, user_criteria: dict) -> str:
    system_prompt = """You are a property valuation expert.
Provide CONCISE property assessments.
For each property provide:
1. Value Assessment: Fair/Over/Under-priced
2. Investment Potential: High/Medium/Low with brief reason
3. Key Recommendation: One actionable insight
Keep each property under 60 words. Use bullet points."""

    props_json = json.dumps(properties_data['properties'][:8], indent=2)

    user_msg = f"""Evaluate these properties for a buyer with budget ${user_criteria.get('min_price',0):,}-${user_criteria.get('max_price',9999999):,}, looking for {user_criteria.get('bedrooms','Any')} bedrooms:

{props_json}

Provide valuation for each property."""

    return groq_chat(client, system_prompt, user_msg, model_name)

# ---------------------------------------------
# Utility Functions
# ---------------------------------------------

def calculate_average_price(properties: list) -> str:
    prices = []
    for p in properties:
        price_str = p.get("price", "")
        if price_str:
            cleaned = re.sub(r'[^\d]', '', str(price_str))
            if cleaned:
                try:
                    prices.append(int(cleaned))
                except:
                    pass
    if prices:
        avg = sum(prices) / len(prices)
        return f"${avg:,.0f}"
    return "N/A"

def get_most_common_type(properties: list) -> str:
    types = [p.get("property_type", "") for p in properties if p.get("property_type")]
    if not types:
        return "Mixed"
    return max(set(types), key=types.count)


# ---------------------------------------------
# Icons (inline SVG, line-style, no emoji)
# ---------------------------------------------

ICONS = {
    "home": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v9a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-9"/><path d="M10 20v-6h4v6"/></svg>',
    "settings": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15 1.65 1.65 0 0 0 3.17 14H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/></svg>',
    "cpu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="7" width="10" height="10" rx="1"/><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2"/></svg>',
    "globe": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14.5 14.5 0 0 1 0 18 14.5 14.5 0 0 1 0-18Z"/></svg>',
    "pin": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12Z"/><circle cx="12" cy="10" r="2.3"/></svg>',
    "doc": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h8M8 9h2"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/></svg>',
    "dollar": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1v22M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
    "search": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>',
    "link": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.07 0l2.83-2.83a5 5 0 0 0-7.07-7.07L11.5 4.5"/><path d="M14 11a5 5 0 0 0-7.07 0L4.1 13.83a5 5 0 0 0 7.07 7.07L12.5 19.5"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    "bed": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 18v-6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v6"/><path d="M2 18v2M22 18v2M2 12V8a2 2 0 0 1 2-2h6v6"/></svg>',
    "bath": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h16v3a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5v-3Z"/><path d="M4 12V6a2 2 0 0 1 2-2h1v3"/><path d="M2 20h20"/></svg>',
    "ruler": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 16 16 3l5 5L8 21z"/><path d="M14 5l2 2M10 9l2 2M6 13l2 2"/></svg>',
    "building": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="1"/><path d="M9 6h.01M15 6h.01M9 10h.01M15 10h.01M9 14h.01M15 14h.01M9 18h6"/></svg>',
    "tag": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 12.9 12.6 4.9a2 2 0 0 0-1.4-.6H5a2 2 0 0 0-2 2v6.2a2 2 0 0 0 .6 1.4l8 8a2 2 0 0 0 2.8 0l6.2-6.2a2 2 0 0 0 0-2.8Z"/><circle cx="8" cy="9" r="1.3"/></svg>',
}

def icon(name: str, size: int = 16) -> str:
    svg = ICONS.get(name, "")
    return f'<span class="icon-wrap" style="width:{size}px;height:{size}px">{svg}</span>'

# ---------------------------------------------
# Streamlit App
# ---------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Real Estate Agent Team",
        page_icon=":house:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #10141a;
        --bg-card: #171c24;
        --bg-elevated: #1d2430;
        --border: #262e3b;
        --border-strong: #3a4356;
        --gold: #c8a355;
        --gold-soft: rgba(200,163,85,0.14);
        --gold-dim: rgba(200,163,85,0.35);
        --green: #5aa387;
        --green-soft: rgba(90,163,135,0.14);
        --text: #edf0f3;
        --text-dim: #98a2b3;
        --text-faint: #5f6b7d;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: var(--bg); color: var(--text); }
    .main .block-container { padding-top: 2.5rem; max-width: 1120px; }

    h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
    h1 { font-size: 2.6rem !important; font-weight: 600 !important; color: #ffffff !important; }
    h2 { font-size: 1.5rem !important; font-weight: 600 !important; color: #e8e8e8 !important; }
    h3 { font-size: 1.15rem !important; font-weight: 600 !important; color: #e0e0e0 !important; }
    p, label, .stMarkdown { color: var(--text-dim) !important; }

    .icon-wrap { display: inline-flex; align-items: center; justify-content: center; vertical-align: middle; }
    .icon-wrap svg { width: 100%; height: 100%; }

    /* Eyebrow / kicker */
    .eyebrow {
        font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.14em; text-transform: uppercase; color: var(--gold);
        margin-bottom: 0.4rem; display: block;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0d1116 !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #dfe3e9 !important; }

    .sidebar-heading {
        display: flex; align-items: center; gap: 9px; font-family: 'Inter', sans-serif;
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--text-dim); margin: 0.4rem 0 0.9rem 0;
    }
    .sidebar-heading .icon-wrap { color: var(--gold); }

    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: var(--bg-elevated) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px var(--gold-soft) !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background-color: var(--bg-elevated) !important; border-color: var(--border) !important;
        border-radius: 6px !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #d4b06a, #b8863a) !important;
        color: #1a1206 !important; border: none !important; border-radius: 6px !important;
        font-weight: 700 !important; padding: 0.7rem 2rem !important;
        font-size: 0.98rem !important; width: 100% !important; transition: all 0.2s ease !important;
        letter-spacing: 0.01em;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(200,163,85,0.3) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: var(--bg-elevated) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
    }
    [data-testid="stLinkButton"] a {
        border-color: var(--border) !important; color: var(--text) !important;
        background: var(--bg-elevated) !important;
    }
    [data-testid="stLinkButton"] a:hover { border-color: var(--gold) !important; color: var(--gold) !important; }

    [data-testid="metric-container"] {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 10px !important; padding: 1.1rem 1.3rem !important;
    }
    [data-testid="metric-container"] label { color: var(--text-faint) !important; font-size: 0.75rem !important; letter-spacing: 0.04em; text-transform: uppercase; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 1.7rem !important; font-weight: 700 !important;
        font-family: 'Fraunces', serif !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: var(--text-faint) !important;
        border-bottom: 2px solid transparent !important; padding: 0.7rem 1.2rem !important; font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--gold) !important; border-bottom-color: var(--gold) !important; background: transparent !important;
    }

    .streamlit-expanderHeader {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; color: var(--text) !important;
    }
    .streamlit-expanderContent {
        background: #141922 !important; border: 1px solid var(--border) !important; border-top: none !important;
    }

    .stCheckbox label { color: #dfe3e9 !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, var(--gold), #e6c589) !important; }

    .stInfo { background: rgba(59,130,246,0.08) !important; border-color: #3b82f6 !important; color: #93c5fd !important; border-radius: 8px !important; }
    .stWarning { background: rgba(245,158,11,0.08) !important; border-color: #f59e0b !important; border-radius: 8px !important; }
    .stSuccess { background: var(--green-soft) !important; border-color: var(--green) !important; border-radius: 8px !important; }
    .stError { border-radius: 8px !important; }

    hr { border-color: var(--border) !important; }

    /* Hero */
    .hero-row { display: flex; align-items: center; gap: 16px; margin-bottom: 0.2rem; }
    .hero-mark {
        width: 52px; height: 52px; border-radius: 50%; border: 1px solid var(--gold-dim);
        background: var(--gold-soft); display: flex; align-items: center; justify-content: center;
        color: var(--gold); flex-shrink: 0;
    }
    .hero-mark .icon-wrap { width: 26px; height: 26px; }
    .hero-title { font-family: 'Fraunces', serif; font-size: 2.5rem; font-weight: 600; color: #fff; line-height: 1.05; margin: 0; }
    .hero-sub { color: var(--text-dim); font-size: 1.02rem; margin: 0.6rem 0 0 68px; }
    .hero-rule { height: 1px; background: linear-gradient(90deg, var(--gold-dim), transparent); margin: 1.6rem 0 2rem 0; }

    /* Section header */
    .section-header {
        display: flex; align-items: center; gap: 10px; margin: 0.2rem 0 1.1rem 0;
    }
    .section-header .icon-wrap { color: var(--gold); }
    .section-header .label {
        font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600; color: #fff;
    }

    /* Property card */
    .property-card {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
        padding: 1.5rem 1.6rem; margin-bottom: 0.9rem; transition: border-color 0.2s;
        position: relative;
    }
    .property-card:hover { border-color: var(--gold-dim); }
    .property-index {
        font-family: 'Fraunces', serif; color: var(--text-faint); font-size: 0.85rem; font-weight: 600;
    }
    .property-address { font-size: 1.12rem; font-weight: 600; color: #fff; margin-top: 2px; }
    .property-tag {
        display: inline-flex; align-items: center; gap: 5px; background: var(--bg-elevated);
        border: 1px solid var(--border); color: var(--text-dim);
        padding: 4px 11px; border-radius: 20px; font-size: 0.78rem; margin: 3px 6px 0 0;
    }
    .property-tag .icon-wrap { color: var(--gold); width: 12px; height: 12px; }
    .price-badge {
        background: var(--green-soft); color: var(--green);
        padding: 6px 16px; border-radius: 20px; font-weight: 700;
        font-size: 1.05rem; border: 1px solid rgba(90,163,135,0.3);
        font-family: 'Fraunces', serif;
    }

    /* Activity log */
    .activity-box {
        background: var(--bg-card); border: 1px solid var(--border); border-left: 2px solid var(--gold);
        border-radius: 6px; padding: 0.7rem 1.1rem; margin-bottom: 0.6rem; color: #dfe3e9; font-size: 0.87rem;
        display: flex; align-items: center; gap: 9px;
    }
    .activity-box .icon-wrap { color: var(--green); flex-shrink: 0; }

    /* Info cards (idle state) */
    .info-card {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
        padding: 1.8rem 1.5rem; text-align: left; height: 100%;
    }
    .info-card .icon-wrap { color: var(--gold); width: 26px; height: 26px; margin-bottom: 0.9rem; }
    .info-card .info-title { font-family: 'Fraunces', serif; font-weight: 600; color: #fff; font-size: 1.05rem; margin-bottom: 0.4rem; }
    .info-card .info-body { color: var(--text-dim); font-size: 0.86rem; line-height: 1.5; }

    .content-box {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
        padding: 1.6rem 1.7rem; line-height: 1.7; color: #d8dce2;
    }

    .status-line { display:flex; align-items:center; gap:8px; color: var(--text-dim); font-size: 0.92rem; }
    </style>
    """, unsafe_allow_html=True)

    # -- Hero --------------------------------------------------------------
    st.markdown(f"""
    <div class="hero-row">
        <div class="hero-mark">{icon('home', 26)}</div>
        <h1 class="hero-title">AI Real Estate Agent Team</h1>
    </div>
    <p class="hero-sub">Specialized AI agents that search listings, read the market, and price every property &mdash; powered by Groq.</p>
    <div class="hero-rule"></div>
    """, unsafe_allow_html=True)

    # -- Sidebar -------------------------------------------------------------
    with st.sidebar:
        st.markdown(f'<div class="sidebar-heading">{icon("settings", 15)} Configuration</div>', unsafe_allow_html=True)

        if GROQ_API_KEY and FIRECRAWL_API_KEY:
            st.success("API keys loaded from .env")
        else:
            missing = []
            if not GROQ_API_KEY:
                missing.append("GROQ_API_KEY")
            if not FIRECRAWL_API_KEY:
                missing.append("FIRECRAWL_API_KEY")
            st.error(f"Missing in .env: {', '.join(missing)}")

        st.markdown("---")
        st.markdown(f'<div class="sidebar-heading">{icon("cpu", 15)} Model</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
            index=0,
            help="llama-3.3-70b-versatile is best, llama-3.1-8b-instant is fastest",
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown(f'<div class="sidebar-heading">{icon("globe", 15)} Search Sources</div>', unsafe_allow_html=True)
        st.markdown("<small style='color:var(--text-faint)'>Select real estate websites to search:</small>", unsafe_allow_html=True)
        use_zillow = st.checkbox("Zillow", value=True)
        use_realtor = st.checkbox("Realtor.com", value=True)
        use_trulia = st.checkbox("Trulia", value=False)
        use_homes = st.checkbox("Homes.com", value=False)

        selected_websites = []
        if use_zillow: selected_websites.append("Zillow")
        if use_realtor: selected_websites.append("Realtor.com")
        if use_trulia: selected_websites.append("Trulia")
        if use_homes: selected_websites.append("Homes.com")

        if selected_websites:
            st.success(f"{len(selected_websites)} source(s) selected")
        else:
            st.warning("Select at least one source")

    # -- Main Form -----------------------------------------------------------
    st.markdown(f'<div class="section-header">{icon("doc", 20)}<span class="label">Your Property Requirements</span></div>', unsafe_allow_html=True)
    st.info("Please provide the location, budget, and property details to help us find your ideal home.")

    with st.form("property_preferences"):
        st.markdown("<span class='eyebrow'>Location &amp; Budget</span>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            city = st.text_input("City", placeholder="e.g., New York")
            state = st.text_input("State/Province (optional)", placeholder="e.g., NY")
        with col2:
            min_price = st.number_input("Minimum Price ($)", min_value=0, value=500000, step=50000)
            max_price = st.number_input("Maximum Price ($)", min_value=0, value=1500000, step=50000)

        st.markdown("<span class='eyebrow'>Property Details</span>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            property_type = st.selectbox("Property Type", ["Any", "House", "Condo", "Townhouse", "Apartment"])
            bedrooms = st.selectbox("Bedrooms", ["Any", "1", "2", "3", "4", "5+"])
        with col2:
            bathrooms = st.selectbox("Bathrooms", ["Any", "1", "2", "3", "4+"])
            min_sqft = st.number_input("Minimum Square Feet", min_value=0, value=1000, step=100)
        with col3:
            timeline = st.selectbox("Timeline", ["Flexible", "ASAP", "1-3 months", "3-6 months", "6+ months"])
            urgency = st.selectbox("Urgency", ["Not urgent", "Moderate", "Urgent"])

        st.markdown("<span class='eyebrow'>Additional Preferences</span>", unsafe_allow_html=True)
        special_requirements = st.text_area(
            "Special Requirements",
            placeholder="e.g., near good schools, garage, backyard, pool, pet-friendly...",
            height=80,
            label_visibility="collapsed"
        )

        submitted = st.form_submit_button("Start Property Analysis", type="primary")

    # -- Run Analysis ----------------------------------------------------------
    if submitted:
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found in .env file. Please add it and restart the app.")
            return
        if not GROQ_API_KEY.startswith("gsk_"):
            st.error("Invalid Groq API key in .env! Must start with 'gsk_'. Get FREE from https://console.groq.com/keys")
            return
        if not FIRECRAWL_API_KEY:
            st.error("FIRECRAWL_API_KEY not found in .env file. Please add it and restart the app.")
            return
        if not city:
            st.error("Please enter a city.")
            return
        if not selected_websites:
            st.error("Please select at least one website to search.")
            return

        user_criteria = {
            "city": city, "state": state,
            "min_price": min_price, "max_price": max_price,
            "property_type": property_type, "bedrooms": bedrooms,
            "bathrooms": bathrooms, "min_sqft": min_sqft,
            "timeline": timeline, "urgency": urgency,
            "special_requirements": special_requirements
        }

        st.markdown("---")
        st.markdown(f'<div class="section-header">{icon("chart", 20)}<span class="label">Property Analysis in Progress</span></div>', unsafe_allow_html=True)

        status_box = st.empty()
        progress_bar = st.progress(0)
        activity_container = st.empty()
        activity_log = []

        def update_status(progress: float, status: str, activity: str):
            status_box.markdown(f'<div class="status-line">{icon("cpu", 15)} {status}</div>', unsafe_allow_html=True)
            progress_bar.progress(progress)
            activity_log.append(activity)
            activity_html = "".join([f'<div class="activity-box">{icon("check", 14)}<span>{a}</span></div>' for a in activity_log])
            activity_container.markdown(
                f'<div style="margin-top:1rem"><b style="color:#fff;font-family:Fraunces,serif">Current Activity</b><br><br>{activity_html}</div>',
                unsafe_allow_html=True
            )

        try:
            groq_client = get_groq_client(GROQ_API_KEY)

            update_status(0.15, "Initializing AI agents...", f"Agents initialized with {model_choice}")
            time.sleep(0.5)

            update_status(0.25, "Searching properties across platforms...", f"Property Search Agent: Scraping {', '.join(selected_websites)}...")

            firecrawl_agent = DirectFirecrawlAgent(FIRECRAWL_API_KEY, groq_client, model_choice)
            properties_data = firecrawl_agent.find_properties_direct(
                city=city, state=state,
                user_criteria=user_criteria,
                selected_websites=selected_websites
            )

            update_status(0.55, "Running market analysis...", f"Market Analysis Agent: Analyzing {properties_data['total_count']} properties found")

            market_analysis = run_market_analysis(groq_client, model_choice, properties_data, city, state)
            update_status(0.75, "Running property valuations...", "Property Valuation Agent: Evaluating individual properties...")

            property_valuations = run_property_valuation(groq_client, model_choice, properties_data, user_criteria)
            update_status(1.0, "Complete!", "Complete analysis ready!")

            time.sleep(0.5)
            status_box.markdown(f'<div class="status-line">{icon("check", 15)} <span style="color:var(--green)">Analysis complete</span></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return

        # -- Results --------------------------------------------------------
        st.markdown("---")
        properties = properties_data.get("properties", [])
        total = properties_data.get("total_count", 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Properties Found", total)
        with col2:
            st.metric("Average Price", calculate_average_price(properties))
        with col3:
            st.metric("Most Common Type", get_most_common_type(properties))

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Properties", "Market Analysis", "Valuations"])

        with tab1:
            if not properties:
                st.warning("No properties found. Try different search criteria or websites.")
            else:
                st.markdown(f"**Found {total} properties matching your criteria**")
                st.markdown("<br>", unsafe_allow_html=True)

                for i, prop in enumerate(properties, 1):
                    addr = prop.get("address", "Unknown Address")
                    price = prop.get("price", "Price not available")
                    beds = prop.get("bedrooms", "?")
                    baths = prop.get("bathrooms", "?")
                    sqft = prop.get("square_feet", "?")
                    ptype = prop.get("property_type", "Property")
                    listing_url = prop.get("listing_url", "")
                    source = prop.get("source", "")

                    st.markdown(f"""
                    <div class="property-card">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">
                            <div>
                                <div class="property-index">LISTING #{i:02d}</div>
                                <div class="property-address">{addr}</div>
                                <div style="margin-top:10px">
                                    <span class="property-tag">{icon('building', 12)}{ptype}</span>
                                    <span class="property-tag">{icon('bed', 12)}{beds} beds</span>
                                    <span class="property-tag">{icon('bath', 12)}{baths} baths</span>
                                    <span class="property-tag">{icon('ruler', 12)}{sqft} sq ft</span>
                                    {f'<span class="property-tag">{icon("tag", 12)}{source}</span>' if source else ''}
                                </div>
                            </div>
                            <div style="text-align:right">
                                <div class="price-badge">{price}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        with st.expander("Investment Analysis"):
                            val_text = property_valuations if property_valuations else "Analysis not available."
                            st.markdown(val_text[:500] + "..." if len(val_text) > 500 else val_text)
                    with col_b:
                        if listing_url and listing_url.startswith("http"):
                            st.link_button("Property Link", listing_url, use_container_width=True)

                    st.markdown("<br>", unsafe_allow_html=True)

        with tab2:
            st.markdown(f'<div class="section-header">{icon("chart", 18)}<span class="label" style="font-size:1.1rem">Market Analysis</span></div>', unsafe_allow_html=True)
            if market_analysis:
                st.markdown(f"""
                <div class='content-box'>
                {market_analysis.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Market analysis not available.")

        with tab3:
            st.markdown(f'<div class="section-header">{icon("dollar", 18)}<span class="label" style="font-size:1.1rem">Property Valuations</span></div>', unsafe_allow_html=True)
            if property_valuations:
                st.markdown(f"""
                <div class='content-box'>
                {property_valuations.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Property valuations not available.")

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="info-card">
                {icon('search', 26)}
                <div class="info-title">Property Search Agent</div>
                <div class="info-body">Scrapes Zillow, Realtor.com, Trulia and Homes.com using Firecrawl.</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-card">
                {icon('chart', 26)}
                <div class="info-title">Market Analysis Agent</div>
                <div class="info-body">Reads trends, neighborhoods, and investment outlook.</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="info-card">
                {icon('dollar', 26)}
                <div class="info-title">Valuation Agent</div>
                <div class="info-body">Evaluates fair pricing and investment potential per property.</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
