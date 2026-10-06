# Wesley wrote this
"""The learning-rate schedule has to reach every optimizer in the form it expects."""
import torch

from wesleyqwen.train import set_lr


def test_a_tensor_learning_rate_stays_a_tensor():
    # torchao's 8-bit Adam keeps lr as a tensor and refuses to step if it is replaced by a float.
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=torch.tensor(1e-3))
    set_lr([opt], 5e-4)
    lr = opt.param_groups[0]["lr"]
    assert isinstance(lr, torch.Tensor) and lr.item() == torch.tensor(5e-4).item()


def test_a_float_learning_rate_is_replaced():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=1e-3)
    set_lr([opt], 5e-4)
    assert opt.param_groups[0]["lr"] == 5e-4
