import os
import tempfile
import unittest

import pandas as pd

from nhl_model3 import compute_bet_performance_summary, grade_bets_log


class BetGradingTests(unittest.TestCase):
    def _write_log(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.close()
        pd.DataFrame(rows).to_csv(tmp.name, index=False)
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.remove(tmp.name))
        return tmp.name

    def test_grade_bets_log_grades_bet_and_pick_rows_when_directional(self):
        log_path = self._write_log(
            [
                {
                    "date": "2025-12-22 19:00:00",
                    "game_id": "2025020575",
                    "matchup": "PIT@TOR",
                    "result": "",
                    "action": "BET",
                    "side": "UNDER",
                    "line": 6.5,
                },
                {
                    "date": "2025-12-22 19:00:00",
                    "game_id": "2025020575",
                    "matchup": "PIT@TOR",
                    "result": "",
                    "action": "PICK",
                    "side": "OVER",
                    "line": 6.5,
                },
            ]
        )
        historical = pd.DataFrame(
            [
                {
                    "game_id": "2025020575",
                    "date": "2025-12-22",
                    "home_team": "TOR",
                    "away_team": "PIT",
                    "total_goals": 5,
                }
            ]
        )

        summary = grade_bets_log(log_path=log_path, historical_days=7, historical_frame=historical)
        graded = pd.read_csv(log_path)

        self.assertEqual(summary.get("graded"), 2)

        bet_row = graded[graded["action"].astype(str).str.upper() == "BET"].iloc[0]
        pick_row = graded[graded["action"].astype(str).str.upper() == "PICK"].iloc[0]
        self.assertEqual(str(bet_row["result"]).upper(), "WIN")
        self.assertEqual(str(pick_row["result"]).upper(), "LOSS")

    def test_grade_bets_log_keeps_legacy_blank_action_rows_gradable(self):
        log_path = self._write_log(
            [
                {
                    "date": "2025-12-22 19:00:00",
                    "game_id": "2025020576",
                    "matchup": "NYR@WSH",
                    "result": "",
                    "action": "",
                    "side": "OVER",
                    "line": 5.5,
                }
            ]
        )
        historical = pd.DataFrame(
            [
                {
                    "game_id": "2025020576",
                    "date": "2025-12-22",
                    "home_team": "WSH",
                    "away_team": "NYR",
                    "total_goals": 5,
                }
            ]
        )

        summary = grade_bets_log(log_path=log_path, historical_days=7, historical_frame=historical)
        graded = pd.read_csv(log_path)

        self.assertEqual(summary.get("graded"), 1)
        self.assertEqual(str(graded.iloc[0]["result"]).upper(), "LOSS")

    def test_performance_summary_counts_directional_picks_and_bets(self):
        log_path = self._write_log(
            [
                {
                    "date": "2025-12-23 15:48:47",
                    "game_id": "2025020580",
                    "matchup": "NJD@NYI",
                    "result": "WIN",
                    "action": "PICK",
                    "side": "OVER",
                    "line": 5.5,
                },
                {
                    "date": "2025-12-23 15:48:47",
                    "game_id": "2025020578",
                    "matchup": "BUF@OTT",
                    "result": "LOSS",
                    "action": "BET",
                    "side": "UNDER",
                    "line": 6.5,
                },
            ]
        )

        summary = compute_bet_performance_summary(log_path=log_path)
        self.assertEqual(summary.get("wins"), 1)
        self.assertEqual(summary.get("losses"), 1)
        self.assertEqual(summary.get("total"), 2)
        self.assertIn("(1/2)", str(summary.get("ytd_str", "")))


if __name__ == "__main__":
    unittest.main()
