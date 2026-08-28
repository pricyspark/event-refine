import numpy as np

from flowsis.cli.eval.evaluate_base import _precision_recall_curve


def test_precision_recall_curve_ranks_predictions_by_score() -> None:
    curve = _precision_recall_curve(
        [(0.7, True), (0.8, False), (0.9, True)],
        targets=2,
    )

    np.testing.assert_allclose(curve["scores"], [0.9, 0.8, 0.7])
    np.testing.assert_allclose(curve["recall"], [0.5, 0.5, 1.0])
    np.testing.assert_allclose(curve["precision"], [1.0, 0.5, 2.0 / 3.0])
