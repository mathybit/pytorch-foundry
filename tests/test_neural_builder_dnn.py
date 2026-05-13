"""Tests for neural.builders.dnn_builder."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural.builders.dnn_builder import DNNModel, build_dnn_model, _ACTIVATIONS


# -- helpers --

_BASE_CONFIG = {
    "d_input": 10,
    "d_hidden": [64, 32],
    "d_output": 5,
    "activation": "relu",
    "batch_norm": False,
    "bn_affine": True,
    "bn_track_running_stats": True,
    "dropout": 0.0,
}


def _make_config(**overrides):
    cfg = dict(_BASE_CONFIG)
    cfg.update(overrides)
    return cfg


# -- DNNModel basic --

class TestDNNModelBasic:
    def test_returns_logits(self):
        cfg = _make_config()
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)
        assert out.dtype == torch.float32

    def test_logits_not_sigmoid(self):
        """Logits can be > 1 or < 0 — not probabilities."""
        cfg = _make_config()
        model = DNNModel(cfg)
        x = torch.randn(1000, 10)  # large batch to get extreme values
        out = model(x)
        assert out.max() > 1.0 or out.min() < 0.0, "Logits should not be bounded to [0,1]"

    def test_no_output_activation(self):
        cfg = _make_config()
        model = DNNModel(cfg)
        # Verify output layer is nn.Linear, not nn.Softmax/etc.
        assert isinstance(model.output, nn.Linear)

    def test_parameter_count(self):
        cfg = _make_config()
        model = DNNModel(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        # 10*64+64 + 64*32+32 + 32*5+5 = 704 + 2080 + 165 = 2949
        assert n_params == 2949

    def test_embedding_returns_pre_logits(self):
        cfg = _make_config(d_hidden=[64, 32])
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        logits = model(x, embedding=False)
        emb = model(x, embedding=True)
        assert logits.shape == (4, 5)
        assert emb.shape == (4, 32)

    def test_embedding_same_hidden(self):
        cfg = _make_config()
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        emb1 = model(x, embedding=True)
        emb2 = model(x, embedding=True)
        assert torch.allclose(emb1, emb2)


# -- No hidden layers --

class TestNoHiddenLayers:
    def test_empty_hidden(self):
        cfg = _make_config(d_hidden=[])
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)
        assert isinstance(model.hidden, nn.Identity)
        assert isinstance(model.output, nn.Linear)
        assert model.output.in_features == 10
        assert model.output.out_features == 5

    def test_embedding_no_hidden(self):
        cfg = _make_config(d_hidden=[])
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        logits1 = model(x, embedding=False)
        logits2 = model(x, embedding=True)
        assert torch.equal(logits1, logits2)


# -- Single hidden layer --

class TestSingleHiddenLayer:
    def test_single_hidden(self):
        cfg = _make_config(d_hidden=[64])
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_embedding_single_hidden(self):
        cfg = _make_config(d_hidden=[64])
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        emb = model(x, embedding=True)
        assert emb.shape == (4, 64)


# -- Activation --

class TestActivation:
    def test_single_activation_string(self):
        cfg = _make_config(activation="relu")
        model = DNNModel(cfg)
        assert len([m for m in model.modules() if isinstance(m, nn.ReLU)]) == 2

    def test_activation_list(self):
        cfg = _make_config(activation=["relu", "elu"])
        model = DNNModel(cfg)
        modules = list(model.hidden.modules())
        activations = [m for m in modules if isinstance(m, (nn.ReLU, nn.ELU))]
        assert len(activations) == 2

    def test_invalid_activation(self):
        cfg = _make_config(activation="invalid_act")
        with pytest.raises(ValueError, match="Unknown activation"):
            DNNModel(cfg)

    def test_mismatched_activation_length(self):
        cfg = _make_config(activation=["relu"])
        with pytest.raises(ValueError, match="activation list length"):
            DNNModel(cfg)

    def test_leakyrelu(self):
        cfg = _make_config(activation="leakyrelu")
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_gelu(self):
        cfg = _make_config(activation="gelu")
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)


# -- Batch normalization --

class TestBatchNorm:
    def test_batch_norm_all(self):
        cfg = _make_config(batch_norm=True)
        model = DNNModel(cfg)
        bn_count = len([m for m in model.modules() if isinstance(m, nn.BatchNorm1d)])
        assert bn_count == 2

    def test_batch_norm_none(self):
        cfg = _make_config(batch_norm=False)
        model = DNNModel(cfg)
        bn_count = len([m for m in model.modules() if isinstance(m, nn.BatchNorm1d)])
        assert bn_count == 0

    def test_batch_norm_list(self):
        cfg = _make_config(batch_norm=[True, False])
        model = DNNModel(cfg)
        bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_modules) == 1

    def test_batch_norm_mismatched_length(self):
        cfg = _make_config(batch_norm=[True])
        with pytest.raises(ValueError, match="batch_norm list length"):
            DNNModel(cfg)

    def test_bn_affine_false(self):
        cfg = _make_config(batch_norm=True, bn_affine=False)
        model = DNNModel(cfg)
        bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert all(not m.affine for m in bn_modules)

    def test_bn_track_running_stats_false(self):
        cfg = _make_config(batch_norm=True, bn_track_running_stats=False)
        model = DNNModel(cfg)
        bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert all(not m.track_running_stats for m in bn_modules)

    def test_bn_forward_infer(self):
        """BatchNorm should behave differently in train vs eval mode."""
        cfg = _make_config(batch_norm=True)
        model = DNNModel(cfg)
        x = torch.randn(100, 10)
        model.train()
        out_train = model(x)
        model.eval()
        out_eval = model(x)
        # Outputs should differ due to different batch statistics
        assert not torch.allclose(out_train, out_eval, atol=1e-3)


# -- Dropout --

class TestDropout:
    def test_dropout_enabled(self):
        cfg = _make_config(dropout=0.5)
        model = DNNModel(cfg)
        dp_count = len([m for m in model.modules() if isinstance(m, nn.Dropout)])
        assert dp_count == 2

    def test_dropout_disabled(self):
        cfg = _make_config(dropout=0.0)
        model = DNNModel(cfg)
        dp_count = len([m for m in model.modules() if isinstance(m, nn.Dropout)])
        assert dp_count == 0

    def test_dropout_train_vs_eval(self):
        cfg = _make_config(d_hidden=[64, 32], dropout=0.5)
        model = DNNModel(cfg)
        x = torch.randn(100, 10)
        model.train()
        out_train = model(x)
        model.eval()
        out_eval = model(x)
        assert not torch.allclose(out_train, out_eval, atol=1e-3)


# -- build_dnn_model function --

class TestBuildDNNModel:
    def test_returns_module(self):
        cfg = _make_config()
        model = build_dnn_model(cfg)
        assert isinstance(model, DNNModel)
        assert isinstance(model, torch.nn.Module)

    def test_forward(self):
        cfg = _make_config()
        model = build_dnn_model(cfg)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)


# -- Config validation --

class TestConfigValidation:
    def test_missing_d_input(self):
        cfg = _BASE_CONFIG.copy()
        del cfg["d_input"]
        with pytest.raises(KeyError):
            DNNModel(cfg)

    def test_missing_d_output(self):
        cfg = _BASE_CONFIG.copy()
        del cfg["d_output"]
        with pytest.raises(KeyError):
            DNNModel(cfg)

    def test_none_hidden(self):
        cfg = _make_config(d_hidden=None)
        model = DNNModel(cfg)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)


# -- Forward with batch dimensions --

class TestBatchDims:
    def test_single_sample(self):
        cfg = _make_config()
        model = DNNModel(cfg)
        x = torch.randn(1, 10)
        assert model(x).shape == (1, 5)

    def test_large_batch(self):
        cfg = _make_config()
        model = DNNModel(cfg)
        x = torch.randn(256, 10)
        assert model(x).shape == (256, 5)

    def test_multidim_input(self):
        cfg = _make_config()
        model = DNNModel(cfg)
        x = torch.randn(2, 8, 10)
        assert model(x).shape == (2, 8, 5)
