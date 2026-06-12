import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from onc_ssamba.experiments import anomaly_holdout as holdout


class AnomalyHoldoutTests(unittest.TestCase):
    def test_manifest_full_grid_has_expected_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = holdout.build_manifest_rows(
                output_root=tmp_path,
                data_path=tmp_path / "dataset.h5",
                labels=holdout.MAIN_ANOMALY_LABELS,
                seeds=[42, 43, 44],
                methods=holdout.DEFAULT_METHODS,
            )

            self.assertEqual(len(rows), 32 * 3 * 2)
            self.assertEqual({row.k for row in rows}, {0, 1, 2, 3, 4, 5})
            self.assertTrue(any(row.exclude_labels == ["Tonal"] for row in rows))

            manifest_path = holdout.write_manifest(rows, tmp_path)
            loaded = holdout.read_manifest(manifest_path)

            self.assertEqual(len(loaded), len(rows))
            self.assertEqual(loaded[0]["exclude_labels"], [])
            self.assertTrue((tmp_path / "launch_commands.sh").exists())

    def test_evaluate_predictions_uses_novel_vs_normal_subset(self):
        val_rows = [
            {"is_anomalous": False, "score": 0.01, "labels": "normal"},
            {"is_anomalous": False, "score": 0.02, "labels": "normal"},
            {"is_anomalous": True, "score": 0.90, "labels": "Engine Noise"},
        ]
        eval_rows = [
            {"is_anomalous": False, "score": 0.01, "labels": "normal"},
            {"is_anomalous": False, "score": 0.03, "labels": "normal"},
            {"is_anomalous": True, "score": 0.80, "labels": "Tonal"},
            {"is_anomalous": True, "score": 0.70, "labels": "Dropout;Tonal"},
            {"is_anomalous": True, "score": 0.20, "labels": "Engine Noise"},
        ]

        metrics = holdout.evaluate_predictions(eval_rows, val_rows, ["Tonal"])

        self.assertEqual(metrics["novel"]["positives"], 2)
        self.assertEqual(metrics["novel"]["negatives"], 2)
        self.assertEqual(metrics["novel"]["auroc"], 1.0)
        self.assertEqual(metrics["in_distribution"]["positives"], 1)
        self.assertEqual(metrics["full"]["positives"], 3)

    def test_audit_splits_rejects_held_out_labels_in_train_or_val(self):
        train = SimpleNamespace(
            sample_info=[
                {"index": 1, "labels": ["normal"], "is_anomalous": False},
                {"index": 2, "labels": ["Engine Noise"], "is_anomalous": True},
            ]
        )
        val = SimpleNamespace(
            sample_info=[
                {"index": 3, "labels": ["normal"], "is_anomalous": False},
                {"index": 4, "labels": ["Tonal"], "is_anomalous": True},
            ]
        )
        test = SimpleNamespace(
            sample_info=[
                {"index": 5, "labels": ["normal"], "is_anomalous": False},
                {"index": 6, "labels": ["Tonal"], "is_anomalous": True},
                {"index": 7, "labels": ["Engine Noise"], "is_anomalous": True},
            ]
        )

        audit = holdout.audit_splits(train, val, [test], ["Tonal"])

        self.assertFalse(audit["ok"])
        self.assertEqual(audit["train_excluded_label_hits"], 0)
        self.assertEqual(audit["val_excluded_label_hits"], 1)
        self.assertEqual(audit["novel_eval_samples"], 1)

    def test_aggregate_audit_flags_split_signature_mismatch(self):
        base = {
            "seed": 42,
            "k": 1,
            "exclusion_id": "tonal",
            "audit_train_signature": "train-a",
            "audit_val_signature": "val-a",
            "audit_eval_signature": "eval-a",
        }
        rows = [
            {"method": "ssl_finetune", **base},
            {"method": "supervised_scratch", **base, "audit_eval_signature": "eval-b"},
        ]

        mismatches = holdout.split_signature_mismatches(rows)

        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["exclusion_id"], "tonal")


if __name__ == "__main__":
    unittest.main()
