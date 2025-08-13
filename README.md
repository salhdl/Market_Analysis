# Market_Analysis
Real Estate Market Analysis – Multi-Agent (Agno + Gemini + Streamlit)
App Streamlit tout-en-un pour analyser un marché immobilier avec 3 agents IA (Agno + Google Gemini) :

NeighborhoodAnalyzerAgent (type LocalLogic)

InvestmentCalculatorBot (type Mashvisor)

MarketReporterAgent (type Real Estate Webmasters)

⚠️ Les appels externes (LocalLogic/Mashvisor/MLS) sont mockés dans ce repo pour rester exécutable sans comptes tiers. Remplacez facilement les fonctions compute_* par vos intégrations réelles.

✨ Fonctionnalités
UI Streamlit claire avec 3 onglets (quartier, investissement, reporting)

Agno Agents + jeux d’outils @tool Pydantic typés

Gemini via google-genai (config par variable d’environnement)

Calculs cap rate, cash-on-cash, sensibilité taux × loyers (CSV)

Rapport HTML “newsletter-ready” + export PDF (optionnel)

Code monofichier : facile à lire, modifier et déployer

🧱 Architecture (vue rapide)
java
Copier
Modifier
streamlit_market_analysis_agents.py
├── Agents Agno
│   ├── NeighborhoodAnalyzerAgent → tool: locallogic_scores
│   ├── InvestmentCalculatorBot    → tool: rental_investment_metrics
│   └── MarketReporterAgent        → tool: compose_market_report
├── Modèles Pydantic (inputs)
├── Outils "compute_*" (mocked, remplaçables par API réelles)
└── UI Streamlit (3 onglets)
Les 3 agents
NeighborhoodAnalyzerAgent
Analyse un secteur (lat/lon), renvoie des scores d’aménités, tuiles de carte mockées et un profil d’acheteur.

InvestmentCalculatorBot
À partir d’un CSV de loyers comparables, calcule NOI, cap rate, cash-on-cash, flux et table de sensibilité (taux vs loyers).

MarketReporterAgent
Génère un HTML brandé (KPIs, période, ranking d’agents) et un PDF (si reportlab installé).

🚀 Démarrage rapide
Prérequis
Python 3.10+

Clé API Google Gemini

(Optionnel) reportlab pour export PDF

Installation
bash
Copier
Modifier
git clone <votre-repo>.git
cd <votre-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -U agno streamlit pydantic pandas numpy requests python-dateutil google-genai
# PDF (optionnel)
pip install reportlab
Variables d’environnement
bash
Copier
Modifier
# Au choix (Agno lit les deux)
export GEMINI_API_KEY="votre_clef"      # recommandé
# ou
export GOOGLE_API_KEY="votre_clef"

# Modèle (optionnel)
export GEMINI_MODEL="gemini-2.5-flash"  # défaut dans le code
💡 Ne commitez jamais votre clé en dur. Préférez des .env ou variables système.

Lancer l’app
bash
Copier
Modifier
streamlit run streamlit_market_analysis_agents.py
Ouvrez le lien local proposé par Streamlit (par défaut http://localhost:8501).
