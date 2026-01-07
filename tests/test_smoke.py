import os
import unittest


class SmokeTestNHLModel(unittest.TestCase):
    def test_train_and_predict_offline(self) -> None:
        try:
            import pandas  # noqa: F401
            import numpy  # noqa: F401
        except Exception:
            self.skipTest("Optional deps (pandas/numpy) not installed in this environment")

        # Keep it deterministic and fast
        os.environ.setdefault("TRAIN_SPEED", "fast")
        os.environ.setdefault("MAX_TRAIN_SAMPLES", "120")
        os.environ.setdefault("TRAIN_TARGET", "total")
        os.environ.setdefault("MC_SIMS_TOTALS", "2000")

        from nhl_model3 import RealDataNHLModel  # noqa: WPS433 (import inside test)

        model = RealDataNHLModel()
        hist = model.create_realistic_sample_data()
        feats = model.create_enhanced_features(hist)
        X, y, dates = model.prepare_model_data(feats)
        results = model.train_model(X, y, dates)
        self.assertIn("rmse", results)
        self.assertIn("over_under_accuracy", results)

        # Predict a single "game" using the first row's features
        row = X.iloc[0].to_numpy()
        pred = model.predict_game(row, betting_line=6.5, over_american_odds=-110, under_american_odds=-110)
        self.assertIsNotNone(pred.predicted_total)
        self.assertTrue(0.0 <= float(pred.over_probability) <= 1.0)
        self.assertTrue(0.0 <= float(pred.under_probability) <= 1.0)
        # Push prob can be 0 for half-lines; just check normalization is sane
        total_prob = float(pred.over_probability) + float(pred.under_probability) + float(pred.push_probability)
        self.assertTrue(0.98 <= total_prob <= 1.02)


if __name__ == "__main__":
    unittest.main()

