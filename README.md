## K2 Risk – Multi-Asset Reasoning Copilot

K2 Risk wraps portfolio sensitivities with an interpretable reasoning layer powered by the (mocked) **K2 Risk** engine. Upload a multi-asset portfolio and the app turns raw delta/ DV01/ beta vectors into:

- Narrative explanations that answer *why* risk exists.
- Counterfactual stress tests (rates, credit, FX, equity shocks).
- Visual causal graphs linking factor exposures to P&L.

The goal is to demonstrate how institutional PMs can interrogate cross-asset risk with transparent, multi-step logic instead of opaque dashboards.

### Features
- CSV uploader or bundled `data/sample_portfolio.csv` for quick demos.
- Aggregated metrics across rates, credit, FX, and equity.
- Rule-based reasoning chain plus downloadable narrative report.
- Cloud LLM integrations (Gemini or OpenRouter) with automatic fallback to the rule engine.
- Predefined counterfactual scenarios translating sensitivities into P&L impacts.
- Plotly causal diagram showing how shocks propagate through the book.
- Optional historical context upload with line chart + key day callouts.
- Scenario tab that ingests macro research PDFs (sample included) and maps inferred shocks into portfolio impacts, with optional Gemini/OpenRouter support for magnitudes and Landing AI ADE-powered document parsing.

### Getting Started
1. (Recommended) Create a virtual environment with Homebrew’s Python 3.10:
   ```bash
   brew install python@3.10  # skip if already installed
   /opt/homebrew/opt/python@3.10/bin/python3.10 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the **Upload** tab inside the app to provide your CSVs or toggle the bundled samples (portfolio + `data/sample_history.csv` for historical context), then move to **Analysis**.
5. (Optional) Visit the **Scenario** tab to parse macro PDFs; a sample file lives at `data/Macro_Research_Sample.pdf`.

#### Optional: Enable Cloud Reasoning (Gemini or OpenRouter)
- Copy `.env.example` to `.env` (or export env vars directly) and provide the credentials for whichever provider you prefer. The app automatically routes reasoning + scenario parsing through the chosen engine and falls back to the deterministic rules when keys are missing.

**Gemini**
1. `export GEMINI_API_KEY="your-key-here"`
2. (Optional) `export GEMINI_MODEL="gemini-1.5-pro"` to override the default `models/gemini-1.0-pro`.

**OpenRouter**
1. `export OPENROUTER_API_KEY="your-key-here"`
2. (Optional) `export OPENROUTER_MODEL="openrouter/llama-3.1-70b-instruct"` (or any model OpenRouter supports).
3. (Optional) `export OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME` so OpenRouter can attribute calls to your app.

### Portfolio Schema
| Column | Description |
| --- | --- |
| instrument | Identifier for the risk position |
| asset_class | Rates / Credit / Equity / FX / Commodity |
| notional | Exposure in millions |
| duration | Duration in years |
| dv01 | DV01 in $ per 1bp move |
| convexity | Second-order rate sensitivity |
| credit_spread_dv01 | P&L change for 1bp credit move |
| fx_delta | FX delta in base currency |
| beta | Equity beta vs. MSCI World |
| commentary | Optional analyst note |

### Historical Schema (optional upload)
| Column | Description |
| --- | --- |
| date | ISO date (YYYY-MM-DD) |
| pnl | Daily P&L in base currency |
| rates_change_bps | Rate move in basis points |
| credit_change_bps | Credit spread change in basis points |
| fx_change_pct | FX percentage move |
| equity_return_pct | Equity index percentage move |

### Macro Scenario Parser
- Drop a macro PDF into the Scenario tab or use `data/Macro_Research_Sample.pdf`.
- The parser (via `pypdf`) extracts bullet points, infers rate/credit/FX/equity shocks, and pipes them through the portfolio sensitivities.
- When scenarios are purely qualitative, set `GEMINI_API_KEY` or `OPENROUTER_API_KEY` so the LLM can infer reasonable shock magnitudes (defaults stay deterministic if no LLM is configured). Provide `ADE_API_KEY` to have Landing AI ADE handle PDF parsing.
- Customize the PDF structure for better extraction; the parser is heuristic and meant for demos.

### Notes
- The reasoning engine can call Gemini or OpenRouter for richer language output or fall back to the deterministic rules in `k2_reasoner/reasoner.py`.
- Extend `run_counterfactuals` with custom scenarios or connect to real stress-test libraries.
**Landing AI ADE (PDF parsing)**
1. `export ADE_API_KEY="your-ade-api-key"`
2. (Optional) override `ADE_MODEL` (default `dpt-2-latest`) or `ADE_ENDPOINT` if your org uses the EU endpoint.
3. (Optional) `export ADE_SPLIT=page` to receive page-level splits.
4. When ADE credentials are present, the Scenario tab uploads PDFs to ADE instead of using the local `pypdf` extractor; failures automatically fall back to the local parser.
