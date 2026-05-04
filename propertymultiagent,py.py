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

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# Groq Client
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# Direct Firecrawl Agent
# ─────────────────────────────────────────────

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
                st.warning(f"⚠️ Could not scrape {site_name}: {str(e)[:120]}")
                continue

        return {
            "properties": all_properties,
            "total_count": len(all_properties),
            "sources": list(urls_to_search.keys())
        }

# ─────────────────────────────────────────────
# Sequential Analysis Agents
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Real Estate Agent Team",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0e0e0e; color: #f0f0f0; }
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    h1 { font-size: 2.4rem !important; font-weight: 700 !important; color: #ffffff !important; }
    h2 { font-size: 1.5rem !important; font-weight: 600 !important; color: #e8e8e8 !important; }
    h3 { font-size: 1.2rem !important; font-weight: 600 !important; color: #e0e0e0 !important; }
    p, label, .stMarkdown { color: #c0c0c0 !important; }
    [data-testid="stSidebar"] { background-color: #161616 !important; border-right: 1px solid #2a2a2a; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label { color: #e0e0e0 !important; }
    .stTextInput input, .stNumberInput input {
        background-color: #1e1e1e !important; color: #f0f0f0 !important;
        border: 1px solid #333 !important; border-radius: 6px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #e63946 !important;
        box-shadow: 0 0 0 2px rgba(230,57,70,0.2) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e63946, #c1121f) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; padding: 0.6rem 2rem !important;
        font-size: 1rem !important; width: 100% !important; transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(230,57,70,0.4) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: #1e1e1e !important; color: #f0f0f0 !important;
        border: 1px solid #333 !important; border-radius: 6px !important;
    }
    [data-testid="metric-container"] {
        background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
        border-radius: 10px !important; padding: 1rem !important;
    }
    [data-testid="metric-container"] label { color: #888 !important; font-size: 0.8rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important; border-bottom: 1px solid #2a2a2a !important; gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #888 !important;
        border-bottom: 2px solid transparent !important; padding: 0.7rem 1.5rem !important; font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #e63946 !important; border-bottom-color: #e63946 !important; background: transparent !important;
    }
    .streamlit-expanderHeader {
        background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
        border-radius: 8px !important; color: #f0f0f0 !important;
    }
    .streamlit-expanderContent {
        background: #161616 !important; border: 1px solid #2a2a2a !important; border-top: none !important;
    }
    .stCheckbox label { color: #e0e0e0 !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #e63946, #ff6b6b) !important; }
    .stInfo { background: #1a2035 !important; border-color: #3b82f6 !important; color: #93c5fd !important; }
    .stWarning { background: #201a10 !important; border-color: #f59e0b !important; }
    .stSuccess { background: #102010 !important; border-color: #22c55e !important; }
    .property-card {
        background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px;
        padding: 1.4rem; margin-bottom: 1rem; transition: border-color 0.2s;
    }
    .property-card:hover { border-color: #e63946; }
    .property-tag {
        display: inline-block; background: #2a2a2a; color: #c0c0c0;
        padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; margin: 2px;
    }
    .price-badge {
        background: linear-gradient(135deg, #1e3a1e, #2d5a2d); color: #4ade80;
        padding: 4px 14px; border-radius: 20px; font-weight: 700;
        font-size: 1rem; border: 1px solid #22c55e44;
    }
    .activity-box {
        background: #1a2a1a; border: 1px solid #2a4a2a; border-radius: 8px;
        padding: 0.8rem 1.2rem; margin-bottom: 0.8rem; color: #86efac; font-size: 0.88rem;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem;
        padding-bottom: 0.5rem; border-bottom: 2px solid #e63946; display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────────────────
    st.markdown("# 🏠 AI Real Estate Agent Team")
    st.markdown("<p style='color:#888;margin-top:-10px;'>Find Your Dream Home with Specialized AI Agents powered by Groq (Free!)</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        with st.expander("🔑 API Keys", expanded=True):
            groq_key = st.text_input(
                "Groq API Key",
                value=os.getenv("GROQ_API_KEY", ""),
                type="password",
                help="Get FREE from https://console.groq.com/keys — starts with 'gsk_'"
            )
            if groq_key and not groq_key.startswith("gsk_"):
                st.error("⚠️ Groq key must start with 'gsk_'. Get it FREE from https://console.groq.com/keys")

            firecrawl_key = st.text_input(
                "Firecrawl API Key",
                value=os.getenv("FIRECRAWL_API_KEY", ""),
                type="password",
                help="Get from https://firecrawl.dev"
            )

        st.markdown("---")
        st.markdown("### 🤖 Model")
        model_choice = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
            index=0,
            help="llama-3.3-70b-versatile is best, llama-3.1-8b-instant is fastest"
        )

        st.markdown("---")
        st.markdown("### 🌐 Search Sources")
        st.markdown("<small style='color:#888'>Select real estate websites to search:</small>", unsafe_allow_html=True)
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
            st.success(f"✅ {len(selected_websites)} source(s) selected")
        else:
            st.warning("⚠️ Select at least one source")

    # ── Main Form ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Your Property Requirements</div>', unsafe_allow_html=True)
    st.info("Please provide the location, budget, and property details to help us find your ideal home.")

    with st.form("property_preferences"):
        st.markdown("### 📍 Location & Budget")
        col1, col2 = st.columns(2)
        with col1:
            city = st.text_input("🏙️ City", placeholder="e.g., New York")
            state = st.text_input("🗺️ State/Province (optional)", placeholder="e.g., NY")
        with col2:
            min_price = st.number_input("💰 Minimum Price ($)", min_value=0, value=500000, step=50000)
            max_price = st.number_input("💰 Maximum Price ($)", min_value=0, value=1500000, step=50000)

        st.markdown("### 🏡 Property Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            property_type = st.selectbox("🏠 Property Type", ["Any", "House", "Condo", "Townhouse", "Apartment"])
            bedrooms = st.selectbox("🛏️ Bedrooms", ["Any", "1", "2", "3", "4", "5+"])
        with col2:
            bathrooms = st.selectbox("🚿 Bathrooms", ["Any", "1", "2", "3", "4+"])
            min_sqft = st.number_input("📐 Minimum Square Feet", min_value=0, value=1000, step=100)
        with col3:
            timeline = st.selectbox("📅 Timeline", ["Flexible", "ASAP", "1-3 months", "3-6 months", "6+ months"])
            urgency = st.selectbox("⚡ Urgency", ["Not urgent", "Moderate", "Urgent"])

        st.markdown("### 🎯 Additional Preferences")
        special_requirements = st.text_area(
            "Special Requirements",
            placeholder="e.g., near good schools, garage, backyard, pool, pet-friendly...",
            height=80
        )

        submitted = st.form_submit_button("🚀 Start Property Analysis", type="primary")

    # ── Run Analysis ──────────────────────────────────────────────────────
    if submitted:
        if not groq_key:
            st.error("❌ Please enter your Groq API key in the sidebar.")
            return
        if not groq_key.startswith("gsk_"):
            st.error("❌ Invalid Groq API key! Must start with 'gsk_'. Get FREE from https://console.groq.com/keys")
            return
        if not firecrawl_key:
            st.error("❌ Please enter your Firecrawl API key in the sidebar.")
            return
        if not city:
            st.error("❌ Please enter a city.")
            return
        if not selected_websites:
            st.error("❌ Please select at least one website to search.")
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
        st.markdown('<div class="section-header">🔄 Property Analysis in Progress</div>', unsafe_allow_html=True)

        status_box = st.empty()
        progress_bar = st.progress(0)
        activity_container = st.empty()
        activity_log = []

        def update_status(progress: float, status: str, activity: str):
            status_box.info(f"⏳ {status}")
            progress_bar.progress(progress)
            activity_log.append(activity)
            activity_html = "".join([f'<div class="activity-box">✅ {a}</div>' for a in activity_log])
            activity_container.markdown(
                f'<div style="margin-top:1rem"><b style="color:#fff">📋 Current Activity</b><br><br>{activity_html}</div>',
                unsafe_allow_html=True
            )

        try:
            groq_client = get_groq_client(groq_key)

            update_status(0.15, "Initializing AI agents...", f"🤖 Agents initialized with {model_choice}")
            time.sleep(0.5)

            update_status(0.25, "Searching properties across platforms...", f"🔍 Property Search Agent: Scraping {', '.join(selected_websites)}...")

            firecrawl_agent = DirectFirecrawlAgent(firecrawl_key, groq_client, model_choice)
            properties_data = firecrawl_agent.find_properties_direct(
                city=city, state=state,
                user_criteria=user_criteria,
                selected_websites=selected_websites
            )

            update_status(0.55, "Running market analysis...", f"📊 Market Analysis Agent: Analyzing {properties_data['total_count']} properties found")

            market_analysis = run_market_analysis(groq_client, model_choice, properties_data, city, state)
            update_status(0.75, "Running property valuations...", "💰 Property Valuation Agent: Evaluating individual properties...")

            property_valuations = run_property_valuation(groq_client, model_choice, properties_data, user_criteria)
            update_status(1.0, "Complete!", "🎉 Complete analysis ready!")

            time.sleep(0.5)
            status_box.success("✅ Analysis Complete!")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            return

        # ── Results ──────────────────────────────────────────────────────
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

        tab1, tab2, tab3 = st.tabs(["🏠 Properties", "📊 Market Analysis", "💰 Valuations"])

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
                                <div style="font-size:1.1rem;font-weight:700;color:#fff">#{i} 🏠 {addr}</div>
                                <div style="margin-top:6px">
                                    <span class="property-tag">🏡 {ptype}</span>
                                    <span class="property-tag">🛏️ {beds} beds</span>
                                    <span class="property-tag">🚿 {baths} baths</span>
                                    <span class="property-tag">📐 {sqft} sq ft</span>
                                    {f'<span class="property-tag">📌 {source}</span>' if source else ''}
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
                        with st.expander("💰 Investment Analysis"):
                            val_text = property_valuations if property_valuations else "Analysis not available."
                            st.markdown(val_text[:500] + "..." if len(val_text) > 500 else val_text)
                    with col_b:
                        if listing_url and listing_url.startswith("http"):
                            st.link_button("🔗 Property Link", listing_url, use_container_width=True)

                    st.markdown("<br>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="section-header">📊 Market Analysis</div>', unsafe_allow_html=True)
            if market_analysis:
                st.markdown(f"""
                <div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:1.5rem;line-height:1.7;color:#d0d0d0;'>
                {market_analysis.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Market analysis not available.")

        with tab3:
            st.markdown('<div class="section-header">💰 Property Valuations</div>', unsafe_allow_html=True)
            if property_valuations:
                st.markdown(f"""
                <div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:1.5rem;line-height:1.7;color:#d0d0d0;'>
                {property_valuations.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Property valuations not available.")

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:1.5rem;text-align:center'>
                <div style='font-size:2rem'>🔍</div>
                <div style='font-weight:700;color:#fff;margin:8px 0'>Property Search Agent</div>
                <div style='color:#888;font-size:0.85rem'>Scrapes Zillow, Realtor.com, Trulia & Homes.com using Firecrawl</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:1.5rem;text-align:center'>
                <div style='font-size:2rem'>📊</div>
                <div style='font-weight:700;color:#fff;margin:8px 0'>Market Analysis Agent</div>
                <div style='color:#888;font-size:0.85rem'>Analyzes trends, neighborhoods & investment outlook</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:1.5rem;text-align:center'>
                <div style='font-size:2rem'>💰</div>
                <div style='font-weight:700;color:#fff;margin:8px 0'>Valuation Agent</div>
                <div style='color:#888;font-size:0.85rem'>Evaluates fair pricing & investment potential per property</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()