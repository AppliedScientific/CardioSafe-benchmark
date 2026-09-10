import torch
from model.revision_baselines import build_baseline_model
from model.revision import REVISION_HEADS,build_revision_model


def test_concat_baseline_factory_preserves_locked_model_exactly():
    torch.manual_seed(42);a=build_revision_model()
    torch.manual_seed(42);b=build_baseline_model()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)


def test_attention_comparator_routes_four_channels_and_backpropagates():
    torch.set_num_threads(1)
    for mode in ['static','trunk','centered']:
        m=build_baseline_model(mode=mode);out=m(torch.randn(8,6747))
        assert set(out)==set(REVISION_HEADS)
        sum(v.square().mean() for v in out.values()).backward()
        assert m.query_embed.grad.shape==(4,128) and torch.isfinite(m.query_embed.grad).all()
        m.eval();stats=m.attention_stats(torch.randn(8,6747))
        assert stats['per_molecule_sd']>=0
