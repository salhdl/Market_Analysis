"""
Streamlit Multi-Agent app for Real Estate market analysis (Gemini + Agno)

Agents:
  1) NeighborhoodAnalyzerAgent (LocalLogic-like)
  2) InvestmentCalculatorBot (Mashvisor-like)
  3) MarketReporterAgent (Real Estate Webmasters-like)

Design goals:
- Single-file, readable, and production-ish structure
- Uses Agno Agent + custom @tool functions
- Streamlit UI with three tabs

Run locally:
  pip install -U agno streamlit pydantic pandas numpy requests python-dateutil google-genai
  # optional for PDF export in tabs 1 & 3
  pip install reportlab

  # Gemini API key (Agno supports Google Gemini via google-genai)
  export GEMINI_API_KEY=your_key   # or: export GOOGLE_API_KEY=your_key

  streamlit run streamlit_market_analysis_agents.py
"""

from __future__ import annotations

import io
import json
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser
from pydantic import BaseModel, Field

# Agno imports
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.decorator import tool

# --------------------------------------------------------------------------------------
# Utilities & constants
# --------------------------------------------------------------------------------------
APP_TITLE = "Real Estate Market Analysis – Multi-Agent (Agno + Gemini + Streamlit)"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# If you want to enforce key presence explicitly, uncomment:
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY.")

# Helpful defaults for demo
DEFAULT_LATLON = (45.5017, -73.5673)  # Montréal downtown
DEFAULT_AMENITIES = ["schools", "groceries", "parks", "transit"]

# --------------------------------------------------------------------------------------
# 1) NeighborhoodAnalyzerAgent – models & functions
# --------------------------------------------------------------------------------------
class NeighborhoodInput(BaseModel):
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    categories: List[str] = Field(default_factory=list)
    commute_minutes: int = Field(20, ge=1, le=120)

def compute_locallogic_scores(payload: NeighborhoodInput) -> dict:
    lat, lon = float(payload.lat), float(payload.lon)
    cats = payload.categories or DEFAULT_AMENITIES

    def score_for(cat: str) -> int:
        base = abs(math.sin(lat) * math.cos(lon)) * 70 + 20
        jitter = (sum(ord(c) for c in cat) % 15) - 7
        return int(max(0, min(100, round(base + jitter))))

    scores = {cat: score_for(cat) for cat in cats}
    tiles = [
        {"id": "amenities_heat", "url": "https://tiles.example/amenities/{z}/{x}/{y}.png"},
        {"id": "transit_lines", "url": "https://tiles.example/transit/{z}/{x}/{y}.png"},
    ]
    commute = int(payload.commute_minutes)
    buyer_profile = (
        f"Profil acheteur: Acheteur urbain cherchant un accès <{commute} min> aux pôles d'emploi, "
        f"avec priorités {', '.join(cats)}."
    )
    return {"scores": scores, "tiles": tiles, "buyer_profile": buyer_profile}

@tool(name="locallogic_scores")
def locallogic_scores_tool(payload: NeighborhoodInput) -> dict:
    return compute_locallogic_scores(payload)

# --------------------------------------------------------------------------------------
# 2) InvestmentCalculatorBot – models & functions
# --------------------------------------------------------------------------------------
class FinancingInput(BaseModel):
    rate: float = Field(..., description="Annual interest rate %, e.g. 6.5")
    ltv: float = Field(..., description="Loan-to-value %, e.g. 75")
    years: int = Field(..., description="Amortization term in years")

class InvestmentInput(BaseModel):
    rent_comps_csv: str
    price: float
    opex_monthly: float
    hoa_monthly: float
    taxes_yearly: float
    financing: FinancingInput
    vacancy_rate: float = Field(5.0)

def compute_rental_investment_metrics(payload: InvestmentInput) -> dict:
    df = pd.read_csv(io.StringIO(payload.rent_comps_csv))
    if "rent" not in df.columns:
        raise ValueError("CSV must have a 'rent' column")
    if "type" not in df.columns:
        df["type"] = "long"

    avg_rents = df.groupby("type")["rent"].mean().to_dict()
    long_rent = float(avg_rents.get("long", df["rent"].mean()))
    short_rent = float(avg_rents.get("short", long_rent * 1.15))

    price = float(payload.price)
    opex = float(payload.opex_monthly) + float(payload.hoa_monthly)
    taxes_m = float(payload.taxes_yearly) / 12.0
    vac = float(payload.vacancy_rate) / 100.0

    def mortgage_pmt(principal: float, annual_rate: float, years: int) -> float:
        r = (annual_rate / 100.0) / 12.0
        n = years * 12
        if r == 0:
            return principal / n
        return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    rate = float(payload.financing.rate)
    ltv = float(payload.financing.ltv) / 100.0
    years = int(payload.financing.years)
    loan = price * ltv
    down = price - loan
    debt_service = mortgage_pmt(loan, rate, years)

    def scenario(name: str, rent_m: float) -> dict:
        eff_rent = rent_m * (1 - vac)
        noi = eff_rent - opex - taxes_m
        cap_rate = (noi * 12.0) / price * 100.0
        cashflow = noi - debt_service
        coc = (cashflow * 12.0) / max(1.0, down) * 100.0
        return {
            "scenario": name,
            "rent_monthly": round(rent_m, 2),
            "effective_rent": round(eff_rent, 2),
            "NOI_monthly": round(noi, 2),
            "cap_rate_%": round(cap_rate, 2),
            "debt_service_m": round(debt_service, 2),
            "cashflow_m": round(cashflow, 2),
            "cash_on_cash_%": round(coc, 2),
        }

    scenarios = [scenario("long_term", long_rent), scenario("short_term", short_rent)]

    rents = np.round(np.linspace(long_rent * 0.9, long_rent * 1.1, 5), 2)
    rate_grid = np.round(np.linspace(max(0.1, rate - 1.0), rate + 1.0, 5), 2)

    rows = []
    for r_ in rate_grid:
        ds = mortgage_pmt(loan, r_, years)
        for rent_ in rents:
            eff_rent = rent_ * (1 - vac)
            noi = eff_rent - opex - taxes_m
            cap = (noi * 12.0) / price * 100.0
            cf = noi - ds
            coc_ = (cf * 12.0) / max(1.0, down) * 100.0
            rows.append({"rate_%": r_, "rent_m": rent_, "cap_rate_%": round(cap, 2), "cash_on_cash_%": round(coc_, 2)})
    sens_df = pd.DataFrame(rows)
    csv_bytes = sens_df.to_csv(index=False).encode()

    return {"scenarios": scenarios, "sensitivity_csv": csv_bytes.decode()}

@tool(name="rental_investment_metrics")
def rental_investment_metrics_tool(payload: InvestmentInput) -> dict:
    return compute_rental_investment_metrics(payload)

# --------------------------------------------------------------------------------------
# 3) MarketReporterAgent – models & functions
# --------------------------------------------------------------------------------------
class MarketReportInput(BaseModel):
    kpis: List[str] = Field(default_factory=lambda: ["new_listings", "avg_dom", "median_price"])
    mls_ids: List[str] = Field(default_factory=list)
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    brand: Optional[str] = Field("Your Brand")

def compute_market_report(payload: MarketReportInput) -> dict:
    start = dtparser.parse(payload.start_date).date()
    end = dtparser.parse(payload.end_date).date()
    days = max(1, (end - start).days)

    rng = np.random.default_rng(abs(hash(tuple(payload.kpis))) % 2**32)
    kpi_values = {k: int(rng.uniform(10, 200)) for k in payload.kpis}

    agents = ["A. Martin", "B. Chen", "C. Dubois", "D. Singh", "E. Lopez"]
    deals = rng.integers(3, 25, size=len(agents))
    volume = np.round(rng.uniform(0.5, 8.0, size=len(agents)), 2)
    ranking = [{"agent": a, "deals": int(d), "volume_musd": float(v)} for a, d, v in zip(agents, deals, volume)]
    ranking = sorted(ranking, key=lambda r: (r["volume_musd"], r["deals"]), reverse=True)

    brand = payload.brand or "Your Brand"
    html = f"""
    <div style='font-family:Inter,system-ui,Arial;margin:0;padding:16px'>
      <h2 style='margin:0 0 8px'>{brand} • Market Report</h2>
      <p style='margin:0 0 12px;color:#555'>Period: {start} → {end} ({days} days)</p>
      <ul style='padding-left:18px;color:#333'>
        {''.join(f"<li><b>{k}</b>: {v}</li>" for k, v in kpi_values.items())}
      </ul>
      <h3 style='margin:16px 0 8px'>Agent Ranking</h3>
      <ol>
        {''.join(f"<li>{r['agent']} — {r['deals']} deals • ${r['volume_musd']}M</li>" for r in ranking)}
      </ol>
    </div>
    """

    pdf_bytes = None
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        w, h = LETTER
        c.setFont("Helvetica-Bold", 16); c.drawString(72, h - 72, f"{brand} • Market Report")
        c.setFont("Helvetica", 10);      c.drawString(72, h - 92, f"Period: {start} → {end} ({days} days)")
        y = h - 120; c.setFont("Helvetica-Bold", 12); c.drawString(72, y, "KPIs:"); y -= 16; c.setFont("Helvetica", 10)
        for k, v in kpi_values.items():
            c.drawString(86, y, f"• {k}: {v}"); y -= 14
        y -= 8; c.setFont("Helvetica-Bold", 12); c.drawString(72, y, "Agent Ranking:"); y -= 16; c.setFont("Helvetica", 10)
        for r in ranking[:10]:
            c.drawString(86, y, f"• {r['agent']} — {r['deals']} deals • ${r['volume_musd']}M"); y -= 14
            if y < 72: break
        c.showPage(); c.save()
        pdf_bytes = buf.getvalue()
    except Exception:
        pdf_bytes = None

    return {"html": html, "agent_ranking": ranking, "pdf_bytes": pdf_bytes}

@tool(name="compose_market_report")
def compose_market_report_tool(payload: MarketReportInput) -> dict:
    return compute_market_report(payload)

# --------------------------------------------------------------------------------------
# Build Agents (Gemini backend)
# --------------------------------------------------------------------------------------
def make_neighborhood_agent() -> Agent:
    return Agent(
        name="NeighborhoodAnalyzerAgent",
        role="Analyze neighborhoods and produce amenity scores and buyer profile.",
        model = Gemini(id=DEFAULT_MODEL, api_key="AIzaSyAlMVCqCy9dURLZpHa4xTvAdGUPyYi_5qQ"),
        tools=[locallogic_scores_tool],
        instructions=[
            "Use the locallogic_scores tool for scoring and insights.",
            "Return concise, structured guidance.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

def make_investment_agent() -> Agent:
    return Agent(
        name="InvestmentCalculatorBot",
        role="Compute rental investment metrics and sensitivities.",
        model = Gemini(id=DEFAULT_MODEL, api_key="AIzaSyAlMVCqCy9dURLZpHa4xTvAdGUPyYi_5qQ"),
        tools=[rental_investment_metrics_tool],
        instructions=[
            "Always compute cap rate using NOI/Price (NOI excludes debt service).",
            "Report cash-on-cash using (Annual Cashflow / Downpayment).",
            "Return a short explanation of assumptions.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

def make_market_agent() -> Agent:
    return Agent(
        name="MarketReporterAgent",
        role="Assemble branded market report HTML and agent rankings.",
        model = Gemini(id=DEFAULT_MODEL, api_key="AIzaSyAlMVCqCy9dURLZpHa4xTvAdGUPyYi_5qQ"),
        tools=[compose_market_report_tool],
        instructions=[
            "Use compose_market_report to generate newsletter-ready HTML.",
            "Summarize highlights in bullet points.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.expander("About this demo"):
    st.markdown(
        """
        This single-file app shows how to wire up **Agno agents** to tools and expose
        them in a simple **Streamlit UI**. External APIs are mocked but the structure
        mirrors real integrations:
        - LocalLogic: amenity/location scores & tiles
        - Mashvisor: cap rate, cash-on-cash, sensitivity
        - REW: branded HTML report & agent rankings
        - Models: **Google Gemini** via Agno (set `GEMINI_API_KEY` or `GOOGLE_API_KEY`)
        """
    )

@st.cache_resource(show_spinner=False)
def get_agents() -> Tuple[Agent, Agent, Agent]:
    return make_neighborhood_agent(), make_investment_agent(), make_market_agent()

neigh_agent, invest_agent, market_agent = get_agents()

# Tabs
T1, T2, T3 = st.tabs(["🏙️ Neighborhood Analyzer", "📈 Investment Calculator", "📰 Market Reporter"])

# ---------------------------------- TAB 1 -------------------------------------------
with T1:
    st.subheader("NeighborhoodAnalyzerAgent")
    with st.form("neigh_form"):
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=float(DEFAULT_LATLON[0]), format="%.6f")
            cats = st.multiselect("Amenity categories", DEFAULT_AMENITIES, default=DEFAULT_AMENITIES)
        with col2:
            lon = st.number_input("Longitude", value=float(DEFAULT_LATLON[1]), format="%.6f")
            commute = st.slider("Commute preference (minutes)", 5, 90, 20)
        prompt = st.text_area(
            "Goal (optional)",
            "Analyse ce quartier et propose un profil acheteur type avec points forts/faiblesses.",
        )
        submitted = st.form_submit_button("Run analysis")

    if submitted:
        payload = NeighborhoodInput(lat=lat, lon=lon, categories=cats, commute_minutes=commute)
        raw = compute_locallogic_scores(payload)
        st.success("Scores calculés")

        # Scores & tiles nicely
        st.json(raw["scores"])
        st.caption("Mock tile layers (replace with real tile URLs if available)")
        st.dataframe(pd.DataFrame(raw["tiles"]), use_container_width=True)

        # Optional: bar chart of scores
        try:
            import altair as alt
            scores_df = pd.DataFrame([{"category": k, "score": v} for k, v in raw["scores"].items()])
            chart = alt.Chart(scores_df).mark_bar().encode(x="category:N", y="score:Q")
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            pass

        msg = (
            f"Voici les scores: {json.dumps(raw['scores'])}. "
            f"Rédige un bref résumé, un profil d'acheteur et 3 recommandations."
        )
        with st.spinner("Agent is writing summary..."):
            run = neigh_agent.run(msg)
        resp_text = getattr(run, "content", str(run))
        st.markdown(resp_text)

        # PDF export (optional)
        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.pdfgen import canvas
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=LETTER)
            c.setFont("Helvetica-Bold", 14); c.drawString(72, 750, "Neighborhood Analysis Summary")
            c.setFont("Helvetica", 10); y = 730
            for line in resp_text.splitlines():
                if not line.strip(): y -= 8; continue
                c.drawString(72, y, line[:95]); y -= 14
                if y < 72: break
            c.showPage(); c.save()
            st.download_button("📄 Télécharger le rapport PDF", data=buf.getvalue(), file_name="neighborhood_report.pdf")
        except Exception:
            st.info("Installez 'reportlab' pour exporter en PDF (optionnel).")

# ---------------------------------- TAB 2 -------------------------------------------
with T2:
    st.subheader("InvestmentCalculatorBot")

    st.markdown("Upload or paste rent comps as CSV with columns **rent,type**. Type ∈ {short,long}.")
    upload = st.file_uploader("CSV des loyers (comps)", type=["csv"], accept_multiple_files=False)
    default_csv = """rent,type
2200,long
2350,long
130,short
150,short
"""
    if upload is not None:
        csv_text = upload.read().decode("utf-8")
    else:
        csv_text = st.text_area("...ou collez le CSV ici", value=default_csv, height=120)

    c1, c2, c3 = st.columns(3)
    with c1:
        price = st.number_input("Prix d'achat", min_value=10000.0, value=450000.0, step=1000.0)
        opex = st.number_input("Dépenses mensuelles (hors HOA & taxes)", min_value=0.0, value=500.0, step=50.0)
    with c2:
        hoa = st.number_input("HOA mensuel", min_value=0.0, value=75.0, step=5.0)
        taxes = st.number_input("Taxes annuelles", min_value=0.0, value=3600.0, step=100.0)
    with c3:
        rate = st.number_input("Taux %", min_value=0.0, value=6.5, step=0.1)
        ltv = st.number_input("LTV %", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        years = st.number_input("Durée (années)", min_value=1, value=30, step=1)

    vac = st.slider("Vacance %", 0, 30, 5)

    if st.button("Calculer"):
        payload = InvestmentInput(
            rent_comps_csv=csv_text,
            price=price,
            opex_monthly=opex,
            hoa_monthly=hoa,
            taxes_yearly=taxes,
            financing=FinancingInput(rate=rate, ltv=ltv, years=int(years)),
            vacancy_rate=vac,
        )
        results = compute_rental_investment_metrics(payload)

        st.markdown("### Scénarios")
        st.dataframe(pd.DataFrame(results["scenarios"]), use_container_width=True)

        st.markdown("### Table de sensibilité (CSV)")
        st.download_button(
            "⬇️ Télécharger sensitivity.csv",
            data=results["sensitivity_csv"].encode(),
            file_name="sensitivity.csv",
            mime="text/csv",
        )

        explanation_prompt = (
            "En 4–6 lignes, explique les hypothèses, compare long vs court terme, et cite cap rate & CoC."
        )
        with st.spinner("Agent explanation..."):
            run = invest_agent.run(explanation_prompt)
        resp_text = getattr(run, "content", str(run))
        st.markdown(resp_text)

# ---------------------------------- TAB 3 -------------------------------------------
with T3:
    st.subheader("MarketReporterAgent")

    kpis = st.multiselect(
        "KPIs",
        ["new_listings", "avg_dom", "median_price", "sold", "price_sf"],
        default=["new_listings", "avg_dom", "median_price"],
    )
    mls_ids = st.text_input("MLS feed identifiers (comma-sep)", value="MLS-01,MLS-02")

    col1, col2 = st.columns(2)
    with col1:
        start_s = st.date_input("Start date")
    with col2:
        end_s = st.date_input("End date")

    brand = st.text_input("Brand name", value="Acme Realty")

    if st.button("Générer le rapport"):
        start_str = start_s.strftime("%Y-%m-%d")
        end_str = end_s.strftime("%Y-%m-%d")
        payload = MarketReportInput(
            kpis=kpis,
            mls_ids=[s.strip() for s in mls_ids.split(",") if s.strip()],
            start_date=start_str,
            end_date=end_str,
            brand=brand,
        )
        data = compute_market_report(payload)

        st.markdown("### Aperçu newsletter (HTML)")
        st.components.v1.html(data["html"], height=400, scrolling=True)

        st.markdown("### Classement des agents")
        st.dataframe(pd.DataFrame(data["agent_ranking"]), use_container_width=True)

        if data.get("pdf_bytes"):
            st.download_button("📄 Télécharger le PDF brandé", data=data["pdf_bytes"], file_name="market_report.pdf")
        else:
            st.info("Installez 'reportlab' pour exporter un PDF brandé (optionnel).")

st.caption("Built with Agno Agents + Streamlit + Google Gemini. Replace the mocked tool internals with real API calls when you have credentials.")
