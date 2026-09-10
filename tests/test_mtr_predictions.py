import torch
from scripts.build_mtr_predictions import predict


def test_pretrained_prediction_path_matches_eval_head():
    torch.manual_seed(42)
    head = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(384, 384),
                               torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(384, 199)).eval()
    weights = {'regression.dense.weight':head[1].weight, 'regression.dense.bias':head[1].bias,
               'regression.out_proj.weight':head[4].weight, 'regression.out_proj.bias':head[4].bias,
               'norm_mean':torch.randn(199), 'norm_std':torch.rand(199)+.1}
    x = torch.randn(11, 384)
    torch.testing.assert_close(predict(x, weights), head(x)*weights['norm_std']+weights['norm_mean'], rtol=0, atol=0)
