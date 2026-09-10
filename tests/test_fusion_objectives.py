import torch
from train.fusion_objectives import pooled_task_term
from train.evaluate_fusion import paired_bootstrap
import numpy as np
import pytest
from train.fusion_preprocessing import transform_descriptors
from train.evaluate_fusion import routing_probe
from model.fusion_factory import build_model, ROUTED_MODES, BIT_MODES, parameter_digest
from train.train_fusion import atomic_checkpoint
from train.fusion_label_budget import apply_cav_budget


def test_label_budgets_are_nested_train_only_and_leave_other_tasks_unchanged():
    labels = np.arange(700, dtype=float).reshape(100,7)
    labels[::3,5] = np.nan
    train = np.arange(60)
    low, rows_low, meta = apply_cav_budget(labels, train, .25)
    high, rows_high, _ = apply_cav_budget(labels, train, .5)
    full, rows_full, _ = apply_cav_budget(labels, train, 1.)
    assert set(rows_low) < set(rows_high) < set(rows_full)
    np.testing.assert_array_equal(low[60:], labels[60:])
    np.testing.assert_array_equal(low[:,:5], labels[:,:5])
    np.testing.assert_array_equal(full, labels)
    assert meta['retained_training_molecules'] == 15
    assert np.isfinite(low[np.ix_(train,[5,6])]).sum(0).tolist() == meta['retained_label_counts']


def test_failed_checkpoint_write_preserves_previous_best(tmp_path, monkeypatch):
    path = tmp_path/'model.pt'
    atomic_checkpoint({'epoch': 3}, path)
    def interrupted_save(checkpoint, file):
        file.write(b'partial write')
        raise OSError('simulated interruption')
    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError):
        atomic_checkpoint({'epoch': 4}, path)
    assert torch.load(path, weights_only=True) == {'epoch': 3}
    assert sorted(p.name for p in tmp_path.iterdir()) == ['model.pt']


@pytest.mark.parametrize('mode', [*ROUTED_MODES, *BIT_MODES])
def test_routing_probe_supports_each_architecture(mode):
    torch.set_num_threads(1)
    model = build_model(mode).eval()
    x = torch.randn(12, 6548)
    x[:, :6144] = (x[:, :6144] > 1).float()
    probe = routing_probe(model, x)
    assert abs(probe['per_molecule_sd']-model.attention_stats(x)['per_molecule_sd']) < 1e-6
    assert all(np.isfinite(v) for changes in probe['prediction_rms_change_z_or_logit'].values() for v in changes.values())


def test_initial_parameter_digest_detects_mismatch_even_when_routes_are_uniform():
    torch.manual_seed(42)
    a = build_model('bit_attention')
    torch.manual_seed(42)
    b = build_model('bit_attention')
    assert parameter_digest(a) == parameter_digest(b)
    with torch.no_grad():
        b.router.keys[0,0,0] += .1
    assert parameter_digest(a) != parameter_digest(b)


def test_pooled_sparse_loss_matches_full_update_gradient():
    p = torch.tensor([.2, .5, .8, 1.1, .4, -.2], requires_grad=True)
    y = torch.tensor([1., float('nan'), 0., 1., float('nan'), 0.])
    observed = y.isfinite()
    full = ((p[observed]-y[observed])**2).mean()
    expected, = torch.autograd.grad(full, p)
    pooled = 0
    for indices in (slice(0, 2), slice(2, 6)):
        mask = y[indices].isfinite()
        loss = ((p[indices][mask] - y[indices][mask])**2).mean()
        pooled = pooled + pooled_task_term(loss, int(mask.sum()), int(observed.sum()))
    actual, = torch.autograd.grad(pooled, p)
    torch.testing.assert_close(pooled, full)
    torch.testing.assert_close(actual, expected)


def test_paired_bootstrap_identical_predictions_zero_width():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(100, 7))
    for c in (1, 2, 4, 6):
        y[:, c] = y[:, c] > 0
    y[:30, 3] = np.nan
    p = rng.normal(size=y.shape)
    result = paired_bootstrap(y, p, p, 30)
    assert result['delta_wrmse'] == 0
    assert result['ci95'] == [0, 0]
    for head in result['per_regression_head'].values():
        assert head['delta_rmse'] == 0
        assert head['ci95'] == [0, 0]


def test_signed_log_retains_order_and_is_invertible():
    z = np.array([-706., -2., -1., 0., 1., 2., 706.], dtype=np.float32)
    actual = transform_descriptors(z, 'signed_log')
    assert np.all(np.diff(actual) > 0)
    np.testing.assert_allclose(np.sign(actual)*np.expm1(np.abs(actual)), z, rtol=1e-6)
