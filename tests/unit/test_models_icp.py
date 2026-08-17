"""Unit tests for the Invariant Contamination Predictor (ICP) model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aquacontam.models.icp import (  # noqa: E402
    GradientReversalFunction,
    GradientReversalLayer,
    ICPClassifier,
    ICPRegressor,
    _ICPNetwork,
    _sigmoid_schedule,
)


@pytest.fixture()
def synthetic_data() -> dict:
    """Generate synthetic classification data with monitoring confounders."""
    rng = np.random.RandomState(42)
    n = 200
    n_features = 20

    X = rng.randn(n, n_features)
    # Target depends on first 5 features
    logits = X[:, :5].sum(axis=1) + rng.randn(n) * 0.5
    y = (logits > 0).astype(float)

    # Monitoring confounders: correlated with some features
    n_samples = np.exp(X[:, 0] + rng.randn(n) * 0.3 + 2)
    mean_dl = np.abs(X[:, 1] + rng.randn(n) * 0.2 + 0.01)

    confounders = np.column_stack([np.log1p(n_samples), mean_dl])
    groups = rng.choice(10, size=n)

    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y_series = pd.Series(y, name="target")

    return {
        "X": X_df,
        "y": y_series,
        "confounders": confounders,
        "groups": groups,
        "n_samples": n_samples,
    }


@pytest.fixture()
def synthetic_regression_data() -> dict:
    """Generate synthetic regression data."""
    rng = np.random.RandomState(42)
    n = 200
    n_features = 20

    X = rng.randn(n, n_features)
    y = np.maximum(0, X[:, :3].sum(axis=1) + rng.randn(n) * 0.5)
    censored = (y < 0.5).astype(float)
    y[censored > 0.5] = 0.0
    detection_limits = np.full(n, 0.5)

    confounders = np.column_stack(
        [
            np.log1p(rng.exponential(5, n)),
            np.abs(rng.randn(n) * 0.01 + 0.004),
        ]
    )

    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y_series = pd.Series(y, name="target")

    return {
        "X": X_df,
        "y": y_series,
        "censored": censored,
        "detection_limits": detection_limits,
        "confounders": confounders,
    }


class TestGradientReversalLayer:
    def test_forward_is_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(5, 10)
        y = grl(x)
        assert torch.allclose(x, y)

    def test_backward_negates_gradients(self):
        """Verify gradients are negated and scaled by lambda."""
        x = torch.randn(5, 10, requires_grad=True)
        lambda_val = 2.0
        y = GradientReversalFunction.apply(x, lambda_val)
        loss = y.sum()
        loss.backward()
        # Gradient of sum(x) = 1 for each element
        # After reversal: -lambda * 1 = -2
        expected = torch.full_like(x, -lambda_val)
        assert torch.allclose(x.grad, expected)

    def test_set_lambda(self):
        grl = GradientReversalLayer(lambda_=0.0)
        assert grl.lambda_ == 0.0
        grl.set_lambda(1.5)
        assert grl.lambda_ == 1.5


class TestSigmoidSchedule:
    def test_zero_during_warmup(self):
        for epoch in range(20):
            assert _sigmoid_schedule(epoch, warmup=20) == 0.0

    def test_increases_after_warmup(self):
        values = [_sigmoid_schedule(e, warmup=20) for e in range(20, 100)]
        # Should be monotonically non-decreasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-6

    def test_approaches_one(self):
        val = _sigmoid_schedule(200, warmup=20)
        assert val > 0.95


class TestICPNetwork:
    def test_forward_shapes(self):
        net = _ICPNetwork(n_features=20, encoder_sizes=[32, 16], head_size=8)
        x = torch.randn(10, 20)
        task_out, conf_out, z = net(x)
        assert task_out.shape == (10,)
        assert conf_out.shape == (10, 2)
        assert z.shape == (10, 16)  # last encoder size

    def test_contamination_head_independent(self):
        """Verify contamination head output doesn't require monitoring head."""
        net = _ICPNetwork(n_features=10, encoder_sizes=[16], head_size=8)
        x = torch.randn(5, 10)
        z = net.encoder(x)
        task_out = net.contamination_head(z)
        assert task_out.shape == (5, 1)


class TestICPClassifier:
    def test_fit_predict_shapes(self, synthetic_data):
        model = ICPClassifier(
            config={
                "encoder_sizes": [32, 16],
                "head_size": 8,
                "epochs": 5,
                "warmup_epochs": 2,
                "batch_size": 64,
                "patience": 3,
            }
        )
        model.fit(
            synthetic_data["X"],
            synthetic_data["y"],
            confounder_targets=synthetic_data["confounders"],
            groups=synthetic_data["groups"],
        )

        preds = model.predict(synthetic_data["X"])
        assert preds.shape == (200,)
        assert set(np.unique(preds)).issubset({0, 1})

        probs = model.predict_proba(synthetic_data["X"])
        assert probs.shape == (200, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_fit_with_validation(self, synthetic_data):
        model = ICPClassifier(
            config={
                "encoder_sizes": [16],
                "head_size": 8,
                "epochs": 10,
                "warmup_epochs": 2,
                "batch_size": 64,
                "patience": 3,
            }
        )
        X = synthetic_data["X"]
        y = synthetic_data["y"]
        n = len(X)
        split = int(0.8 * n)

        model.fit(
            X.iloc[:split],
            y.iloc[:split],
            X_val=X.iloc[split:],
            y_val=y.iloc[split:],
            confounder_targets=synthetic_data["confounders"][:split],
            confounder_targets_val=synthetic_data["confounders"][split:],
            groups=synthetic_data["groups"][:split],
            groups_val=synthetic_data["groups"][split:],
        )

        probs = model.predict_proba(X)
        assert probs.shape == (n, 2)

    def test_adversary_r2(self, synthetic_data):
        """After training, adversary R² should be available."""
        model = ICPClassifier(
            config={
                "encoder_sizes": [16],
                "head_size": 8,
                "epochs": 5,
                "warmup_epochs": 1,
                "batch_size": 64,
            }
        )
        model.fit(
            synthetic_data["X"],
            synthetic_data["y"],
            confounder_targets=synthetic_data["confounders"],
        )
        r2 = model.get_adversary_r2()
        assert isinstance(r2, float)
        assert not np.isnan(r2)

    def test_training_history(self, synthetic_data):
        """After training, training history should be available."""
        model = ICPClassifier(
            config={
                "encoder_sizes": [16],
                "head_size": 8,
                "epochs": 5,
                "warmup_epochs": 1,
                "batch_size": 64,
            }
        )
        model.fit(
            synthetic_data["X"],
            synthetic_data["y"],
            confounder_targets=synthetic_data["confounders"],
        )
        history = model.get_training_history()
        assert isinstance(history, list)
        assert len(history) == 5
        assert "epoch" in history[0]
        assert "task_loss" in history[0]
        assert "adv_loss" in history[0]
        assert "lambda_adv" in history[0]
        # Warmup epoch should have lambda_adv == 0
        assert history[0]["lambda_adv"] == 0.0

    def test_fit_without_confounders(self, synthetic_data):
        """Model should work even without confounder targets (fallback to zeros)."""
        model = ICPClassifier(
            config={
                "encoder_sizes": [16],
                "head_size": 8,
                "epochs": 3,
                "batch_size": 64,
            }
        )
        model.fit(synthetic_data["X"], synthetic_data["y"])
        probs = model.predict_proba(synthetic_data["X"])
        assert probs.shape == (200, 2)

    def test_model_name(self):
        model = ICPClassifier()
        assert model.name == "icp_classifier"


class TestICPRegressor:
    def test_fit_predict_shapes(self, synthetic_regression_data):
        model = ICPRegressor(
            config={
                "encoder_sizes": [32, 16],
                "head_size": 8,
                "epochs": 5,
                "warmup_epochs": 2,
                "batch_size": 64,
                "patience": 3,
            }
        )
        d = synthetic_regression_data
        model.fit(
            d["X"],
            d["y"],
            censored=d["censored"],
            detection_limits=d["detection_limits"],
            confounder_targets=d["confounders"],
        )

        preds = model.predict(d["X"])
        assert preds.shape == (200,)
        assert (preds >= 0).all()

    def test_predict_proba_raises(self, synthetic_regression_data):
        model = ICPRegressor(
            config={
                "encoder_sizes": [16],
                "head_size": 8,
                "epochs": 3,
                "batch_size": 64,
            }
        )
        d = synthetic_regression_data
        model.fit(d["X"], d["y"])

        with pytest.raises(NotImplementedError):
            model.predict_proba(d["X"])

    def test_training_history_regressor(self, synthetic_regression_data):
        """ICPRegressor should record training history."""
        model = ICPRegressor(
            config={
                "encoder_sizes": [16],
                "head_size": 8,
                "epochs": 4,
                "warmup_epochs": 1,
                "batch_size": 64,
            }
        )
        d = synthetic_regression_data
        model.fit(
            d["X"],
            d["y"],
            censored=d["censored"],
            detection_limits=d["detection_limits"],
        )
        history = model.get_training_history()
        assert isinstance(history, list)
        assert len(history) == 4
        assert "task_loss" in history[0]
        assert "adv_loss" in history[0]

    def test_model_name(self):
        model = ICPRegressor()
        assert model.name == "icp_regressor"


class TestIPWWeights:
    def test_import_and_compute(self, synthetic_data):
        from aquacontam.models._propensity import (
            compute_ipw_weights,
            compute_propensity_scores,
        )

        propensity = compute_propensity_scores(synthetic_data["X"], synthetic_data["n_samples"])
        assert len(propensity) == 200
        assert propensity.min() >= 0
        assert propensity.max() <= 1

        weights = compute_ipw_weights(propensity, synthetic_data["n_samples"])
        assert len(weights) == 200
        assert weights.min() >= 0.1
        assert weights.max() <= 10.0
        # Stabilized: mean should be ~1
        assert abs(weights.mean() - 1.0) < 0.1


class TestLazyImport:
    def test_icp_in_lazy_models(self):
        from aquacontam.models import ICPClassifier, ICPRegressor

        assert ICPClassifier is not None
        assert ICPRegressor is not None
