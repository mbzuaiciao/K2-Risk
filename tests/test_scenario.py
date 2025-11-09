from __future__ import annotations

import unittest

from unittest import mock

from k2_reasoner import scenario
from k2_reasoner.scenario import derive_scenarios, interpret_shocks


class ScenarioParsingTests(unittest.TestCase):
    def test_interpret_shocks_assigns_fx_and_equity_percentages(self) -> None:
        text = "USD weakens 2% while global equities drop 4% and credit spreads widen 25bp."
        shocks = interpret_shocks(text)
        self.assertAlmostEqual(shocks["fx_move"], 0.02)
        self.assertAlmostEqual(shocks["equity_move"], 0.04)
        self.assertEqual(shocks["credit_bps"], 25.0)

    def test_interpret_shocks_handles_multiple_percentage_mentions(self) -> None:
        text = "Equities sell off 3% and the dollar strengthens 1% amid a dovish rate rally."
        shocks = interpret_shocks(text)
        self.assertAlmostEqual(shocks["equity_move"], 0.03)
        self.assertAlmostEqual(shocks["fx_move"], 0.01)
        # No explicit bp figure, but rates keyword should trigger default +/-25bp.
        self.assertEqual(abs(shocks["rate_bps"]), 25.0)

    def test_interpret_shocks_map_bp_values_to_correct_assets(self) -> None:
        text = "Credit spreads widen 25bp while rates climb 50bp and the dollar holds steady."
        shocks = interpret_shocks(text)
        self.assertEqual(shocks["credit_bps"], 25.0)
        self.assertEqual(shocks["rate_bps"], 50.0)

    def test_derive_scenarios_respects_bullet_markers(self) -> None:
        raw = """• Rates sell-off quickens on sticky inflation
        • Credit sentiment cracks as defaults tick up
        Macro overview continues here without marker"""
        bullets = derive_scenarios(raw)
        self.assertEqual(len(bullets), 2)

    def test_maybe_llm_shocks_uses_gemini_payload(self) -> None:
        with mock.patch.object(scenario, "gemini_available", return_value=True), mock.patch.object(
            scenario,
            "generate_gemini_scenario_shocks",
            return_value={"rate_bps": 30.0, "credit_bps": 10.0, "fx_move": 0.01, "equity_move": -0.02},
        ):
            payload, note = scenario._maybe_llm_shocks("Test scenario", "gemini")
            self.assertIsNotNone(payload)
            self.assertEqual(payload["rate_bps"], 30.0)
            self.assertIn("Gemini-derived", note or "")

    def test_maybe_llm_shocks_handles_missing_sdk(self) -> None:
        with mock.patch.object(scenario, "gemini_available", return_value=False):
            payload, note = scenario._maybe_llm_shocks("Scenario text", "gemini")
            self.assertIsNone(payload)
            self.assertIn("unavailable", note or "")

    def test_maybe_llm_shocks_openrouter_success(self) -> None:
        with mock.patch.object(scenario, "openrouter_available", return_value=True), mock.patch.object(
            scenario,
            "generate_openrouter_scenario_shocks",
            return_value={"rate_bps": -15.0, "credit_bps": 5.0, "fx_move": -0.02, "equity_move": 0.01},
        ):
            payload, note = scenario._maybe_llm_shocks("Scenario text", "openrouter")
            self.assertEqual(payload["credit_bps"], 5.0)
            self.assertIn("OpenRouter-derived", note or "")


if __name__ == "__main__":
    unittest.main()
