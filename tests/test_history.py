from __future__ import annotations

import io
import unittest

import pandas as pd

from k2_reasoner.history import HISTORY_SCHEMA, HistoricalSeries, load_history, summarize_history


class HistoryValidationTests(unittest.TestCase):
    def test_load_history_rejects_empty_file(self) -> None:
        header = ",".join(HISTORY_SCHEMA.keys())
        csv_data = header + "\n"
        with self.assertRaises(ValueError):
            load_history(uploaded_file=io.StringIO(csv_data))

    def test_summarize_history_rejects_empty_dataset(self) -> None:
        empty_df = pd.DataFrame(columns=HISTORY_SCHEMA.keys())
        history = HistoricalSeries(raw=empty_df)
        with self.assertRaises(ValueError):
            summarize_history(history)


if __name__ == "__main__":
    unittest.main()
