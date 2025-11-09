from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from k2_reasoner import (
    HISTORY_SCHEMA,
    SAMPLE_COLUMNS,
    build_causal_graph,
    answer_simple_question,
    build_scenario_insights,
    breakdown_sentences,
    load_history,
    generate_reasoning_outputs,
    load_portfolio_dataframe,
    run_counterfactuals,
    summarize_exposures,
    summarize_history,
)

APP_ROOT = Path(__file__).parent
SAMPLE_FILE = APP_ROOT / "data" / "sample_portfolio.csv"
SAMPLE_HISTORY_FILE = APP_ROOT / "data" / "sample_history.csv"
SAMPLE_SCENARIO_PDF = APP_ROOT / "data" / "Macro_Research_Sample.pdf"
load_dotenv(APP_ROOT / ".env")


def _render_portfolio_schema() -> None:
    with st.expander("Portfolio schema"):
        schema_df = pd.DataFrame(
            {"Column": SAMPLE_COLUMNS.keys(), "Description": SAMPLE_COLUMNS.values()}
        )
        st.dataframe(schema_df, use_container_width=True, hide_index=True)


def _render_history_schema() -> None:
    with st.expander("Historical schema"):
        history_df = pd.DataFrame(
            {"Column": HISTORY_SCHEMA.keys(), "Description": HISTORY_SCHEMA.values()}
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="K2 Risk Reasoner", layout="wide")
    st.title("K2 Risk — Multi-Asset Reasoning Copilot")
    st.caption("Translate raw sensitivities into causal explanations using K2 Risk.")

    if "llm_choice" not in st.session_state:
        st.session_state["llm_choice"] = "Rule engine (offline)"
    if "user_question" not in st.session_state:
        st.session_state["user_question"] = ""

    if "analysis_ready" not in st.session_state:
        st.session_state["analysis_ready"] = False
    if "analysis_snapshot" not in st.session_state:
        st.session_state["analysis_snapshot"] = ""
    if "analysis_trigger_id" not in st.session_state:
        st.session_state["analysis_trigger_id"] = 0
    if "scenario_results" not in st.session_state:
        st.session_state["scenario_results"] = []
    if "scenario_auto_ran" not in st.session_state:
        st.session_state["scenario_auto_ran"] = False
    if "portfolio_sample_toggle" not in st.session_state:
        st.session_state["portfolio_sample_toggle"] = True
    if "history_sample_toggle" not in st.session_state:
        st.session_state["history_sample_toggle"] = True
    if "scenario_sample_toggle" not in st.session_state:
        st.session_state["scenario_sample_toggle"] = True

    (
        about_tab,
        upload_tab,
        positions_tab,
        analysis_tab,
        scenario_tab,
        history_tab,
        counter_tab,
        causal_tab,
        settings_tab,
    ) = st.tabs(
        [
            "About",
            "Upload",
            "Positions",
            "Analysis",
            "Scenario",
            "History",
            "Counterfactual",
            "Causal",
            "Settings",
        ]
    )

    with about_tab:
        st.header("About K2 Risk")
        st.write(
            "K2 Risk combines cross-asset sensitivities with explainable AI so portfolio managers can see **why** risk exists, not just how much."
        )
        st.subheader("Key Features")
        st.markdown(
            """
- 🔍 **Causal Risk Explanations:** Understand the drivers of P&L.
- 📈 **Multi-Asset Coverage:** Rates, credit, FX, commodities, and equities.
- 🤔 **Counterfactual Reasoning:** Stress-test “what if” scenarios.
- 🎯 **Cross-Asset Interactions:** Trace how shocks propagate across factors.
- ⚡ **Real-Time Insights:** Toggle Gemini to generate richer narratives.
"""
        )
        st.subheader("Example Questions")
        st.markdown(
            """
- “Why did my P&L drop today?”
- “What happens if yields rally 50 bps?”
- “How do credit exposures interact with my equity hedges?”
"""
        )

    with upload_tab:
        st.subheader("Portfolio Upload")
        st.info(
            "K2 Risk links rates, credit, FX, and equity exposures into a transparent reasoning chain. "
            "Use this tab to load your portfolio and optional historical context; schema references sit below each uploader."
        )
        portfolio_upload = st.file_uploader("Portfolio CSV", type=["csv"], key="portfolio_upload")
        use_sample = st.toggle(
            "Use bundled sample portfolio",
            value=st.session_state["portfolio_sample_toggle"],
            key="portfolio_sample_toggle",
        )
        st.caption("Sample file: data/sample_portfolio.csv")
        _render_portfolio_schema()

        st.subheader("Historical Context (optional)")
        history_upload = st.file_uploader(
            "Historical P&L CSV", type=["csv"], key="history_upload"
        )
        use_history_sample = st.toggle(
            "Use sample history",
            value=st.session_state["history_sample_toggle"],
            key="history_sample_toggle",
        )
        st.caption("Sample file: data/sample_history.csv")
        _render_history_schema()
        st.info("Proceed to the Analysis tab after uploading or enabling samples.")

    portfolio = None
    try:
        if portfolio_upload is not None:
            portfolio = load_portfolio_dataframe(uploaded_file=portfolio_upload)
        elif use_sample and SAMPLE_FILE.exists():
            portfolio = load_portfolio_dataframe(sample_path=SAMPLE_FILE)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    history = None
    try:
        if history_upload is not None:
            history = load_history(uploaded_file=history_upload)
        elif use_history_sample and SAMPLE_HISTORY_FILE.exists():
            history = load_history(sample_path=SAMPLE_HISTORY_FILE)
    except ValueError as exc:
        st.warning(f"Historical data issue: {exc}")

    summary = summarize_exposures(portfolio) if portfolio is not None else None
    signature = repr(summary.totals) if summary is not None else "none"
    prev_signature = st.session_state.get("portfolio_signature")
    if signature != prev_signature:
        st.session_state["portfolio_signature"] = signature
        st.session_state["analysis_ready"] = False
        st.session_state["analysis_snapshot"] = ""
        st.session_state["scenario_results"] = []

    with settings_tab:
        st.subheader("Settings")
        engine_options = [
            "Rule engine (offline)",
            "Gemini (cloud)",
            "OpenAI (coming soon)",
            "Anthropic (coming soon)",
            "Free LLM API (coming soon)",
        ]
        selection = st.selectbox(
            "Reasoning engine",
            options=engine_options,
            key="llm_choice",
            help="Gemini is available today; other providers are placeholders for upcoming integrations.",
        )
        if selection != st.session_state.get("analysis_engine_prev"):
            st.session_state["analysis_ready"] = False
            st.session_state["analysis_snapshot"] = ""
            st.session_state["analysis_engine_prev"] = selection
        st.caption("The selected model is used once you submit a question.")

    user_question = st.session_state.get("user_question")
    llm_choice = st.session_state.get("llm_choice", "Rule engine (offline)")
    current_snapshot = f"{user_question}|{llm_choice}"
    question_submitted = (
        st.session_state.get("analysis_ready")
        and st.session_state.get("analysis_snapshot") == current_snapshot
        and user_question
        and user_question.strip()
    )
    gemini_selected = "Gemini" in llm_choice
    use_gemini = question_submitted and gemini_selected
    unsupported_choice = question_submitted and llm_choice not in (
        "Rule engine (offline)",
        "Gemini (cloud)",
    )

    with positions_tab:
        if portfolio is None or summary is None:
            st.info("Upload a portfolio CSV or enable the sample book to review positions.")
        else:
            df = portfolio.df
            st.subheader("Positions")
            st.dataframe(df, use_container_width=True)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Notional (mm)", f"{summary.totals['notional']:.0f}")
            metric_cols[1].metric("DV01", f"{summary.totals['dv01']:.0f}")
            metric_cols[2].metric("Credit Spread DV01", f"{summary.totals['credit_spread_dv01']:.0f}")
            metric_cols[3].metric("FX Delta", f"{summary.totals['fx_delta']:.0f}")

    with analysis_tab:
        if portfolio is None or summary is None:
            st.info("Upload a portfolio CSV or enable the sample book to unlock analytics.")
        else:
            st.subheader("Question")
            with st.form("analysis_form"):
                st.subheader("Question")
                question_input = st.text_input(
                    "What would you like to understand?",
                    placeholder="Why did my P&L drop today?",
                    key="user_question",
                )
                submitted = st.form_submit_button("Generate")
                if submitted:
                    st.session_state["analysis_snapshot"] = f"{st.session_state.get('user_question','')}|{st.session_state.get('llm_choice')}"
                    st.session_state["analysis_ready"] = True
                    st.session_state["analysis_trigger_id"] += 1

            current_snapshot = f"{st.session_state.get('user_question','')}|{st.session_state.get('llm_choice')}"
            if st.session_state["analysis_snapshot"] != current_snapshot:
                st.session_state["analysis_ready"] = False
                question_submitted = False
            if not st.session_state["analysis_ready"]:
                st.info("Enter a question and click Generate reasoning to run the selected engine.")
            else:
                user_question = st.session_state.get("user_question")
                llm_choice = st.session_state.get("llm_choice", "Rule engine (offline)")
                use_gemini = "Gemini" in llm_choice
                unsupported_choice = llm_choice not in (
                    "Rule engine (offline)",
                    "Gemini (cloud)",
                )
                chosen_label = llm_choice if user_question else "Rule engine (offline)"
                engine_label = "Gemini (cloud)" if "Gemini" in chosen_label else "Local rule engine"
                if unsupported_choice:
                    engine_label = f"{llm_choice} (using rule engine until integration ships)"
                engine_icon = "⚡️" if "Gemini" in chosen_label else "🛡️"
                st.caption(f"{engine_icon} Reasoner source: {engine_label}. Change this in the Settings tab.")
                if unsupported_choice:
                    st.info(
                        f"{llm_choice} routing is not wired yet; responses fall back to the offline rule engine."
                    )
                quick_answer = answer_simple_question(summary, user_question)
                if quick_answer:
                    st.success(quick_answer)

                st.subheader("Reasoning Chain")
                reasoning = generate_reasoning_outputs(
                    portfolio,
                    user_question,
                    use_gemini=use_gemini,
                )
                if reasoning.note:
                    st.info(reasoning.note)
                st.caption(f"Source: {reasoning.source}")
                for step in reasoning.chain:
                    st.markdown(f"- {step}")

                st.subheader("Narrative Explanation")
                st.write(reasoning.narrative)
                st.download_button(
                    "Download narrative",
                    data=reasoning.narrative,
                    file_name="k2_risk_narrative.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    with scenario_tab:
        if portfolio is None or summary is None:
            st.info("Upload a portfolio to evaluate macro scenarios.")
        else:
            scenario_upload = st.file_uploader(
                "Macro research PDF", type=["pdf"], key="scenario_pdf"
            )
            scenario_use_gemini = gemini_selected
            use_sample_scenario = st.toggle(
                "Use sample macro PDF",
                value=st.session_state["scenario_sample_toggle"],
                key="scenario_sample_toggle",
            )
            auto_should_run = (
                summary is not None
                and not st.session_state["scenario_results"]
                and scenario_upload is None
                and use_sample_scenario
                and SAMPLE_SCENARIO_PDF.exists()
            )
            if auto_should_run:
                try:
                    with st.spinner("Extracting sample scenarios..."):
                        insights = build_scenario_insights(
                            SAMPLE_SCENARIO_PDF, summary, use_gemini=scenario_use_gemini
                        )
                    st.session_state["scenario_results"] = insights
                except Exception as exc:  # pragma: no cover
                    st.warning(f"Sample scenario extraction failed: {exc}")

            if st.button("Extract scenarios", key="extract_scenarios"):
                pdf_path: Path | None = None
                temp_path: Path | None = None
                if scenario_upload is not None:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(scenario_upload.getbuffer())
                    tmp.flush()
                    tmp.close()
                    temp_path = Path(tmp.name)
                    pdf_path = temp_path
                elif use_sample_scenario and SAMPLE_SCENARIO_PDF.exists():
                    pdf_path = SAMPLE_SCENARIO_PDF
                if pdf_path is None:
                    st.warning("Provide a macro PDF or enable the sample file.")
                else:
                    try:
                        insights = build_scenario_insights(
                            pdf_path, summary, use_gemini=scenario_use_gemini
                        )
                        st.session_state["scenario_results"] = insights
                        st.success(f"Parsed {len(insights)} scenarios from the document.")
                    except Exception as exc:  # pragma: no cover - best effort
                        st.error(f"Unable to parse scenarios: {exc}")
                    finally:
                        if temp_path:
                            temp_path.unlink(missing_ok=True)

            insights = st.session_state.get("scenario_results", [])
            if insights:
                for idx, insight in enumerate(insights, start=1):
                    st.markdown(f"**Scenario {idx}**")
                    sentences = breakdown_sentences(insight.text)
                    if sentences:
                        for sentence in sentences:
                            st.markdown(f"- {sentence}")
                    else:
                        st.markdown(f"- {insight.text}")
                    st.caption(
                        f"Rates {insight.shocks['rate_bps']:+.0f}bp | "
                        f"Credit {insight.shocks['credit_bps']:+.0f}bp | "
                        f"FX {insight.shocks['fx_move']:+.2%} | "
                        f"Equity {insight.shocks['equity_move']:+.2%}"
                    )
                    if insight.note:
                        st.caption(insight.note)
                    st.markdown(
                        f"💥 **Impact:** <span style='font-size:1.1rem; font-weight:600;'>{insight.impact}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Upload a macro PDF or use the sample to generate scenario bullets.")

    with history_tab:
        if history is None:
            st.info("Upload history data or enable the sample in the Upload tab to view context.")
        else:
            st.subheader("Historical Context")
            df = history.df
            if df.empty:
                st.warning("Historical dataset is empty; upload a file with at least one row to view context.")
            else:
                line_df = df.set_index("date")[["pnl"]]
                st.line_chart(line_df, use_container_width=True)
                try:
                    history_points = summarize_history(history)
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    worst = history_points["worst"]
                    best = history_points["best"]
                    latest = history_points["latest"]
                    st.markdown(
                        f"- **Latest ({latest['date'].date()}):** P&L {latest['pnl']:.2f}, rates move {latest['rates_change_bps']:+.0f}bp, credit {latest['credit_change_bps']:+.0f}bp."
                    )
                    st.markdown(
                        f"- **Worst day ({worst['date'].date()}):** {worst['pnl']:.2f} with rates {worst['rates_change_bps']:+.0f}bp and credit {worst['credit_change_bps']:+.0f}bp."
                    )
                    st.markdown(
                        f"- **Best day ({best['date'].date()}):** {best['pnl']:.2f} alongside rates {best['rates_change_bps']:+.0f}bp and equities {best['equity_return_pct']:+.1f}%."
                    )

    with counter_tab:
        if portfolio is None:
            st.info("Upload a portfolio CSV or enable the sample book to explore counterfactuals.")
        else:
            st.subheader("Counterfactual Explorer")
            scenarios = pd.DataFrame(run_counterfactuals(portfolio))
            st.dataframe(scenarios, use_container_width=True)

    with causal_tab:
        if portfolio is None:
            st.info("Upload a portfolio CSV or enable the sample book to view the causal graph.")
        else:
            st.subheader("Causal Graph")
            summary = summarize_exposures(portfolio)
            fig = build_causal_graph(summary)
            st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()
