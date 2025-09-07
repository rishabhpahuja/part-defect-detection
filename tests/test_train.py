import types
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train import train

class TinyDataset(Dataset):
    def __init__(self, n=4, h=32, w=32):
        self.x = torch.rand(n, 3, h, w)        # images in [0,1]
        self.y = torch.zeros(n, h, w)          # simple mask: left half = 1
        self.y[:, :, : w // 2] = 1

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    def forward(self, x): return self.conv(x)

class SimpleCriterion(nn.Module):
    """Binary loss compatible with your (logits, masks) signature."""
    def forward(self, logits, masks):
        return nn.functional.binary_cross_entropy_with_logits(
            logits, masks.unsqueeze(1).float()
        )


def test_train_updates_params_and_returns_float():
    ds = TinyDataset(n=2)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = TinyModel()
    crit = SimpleCriterion()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # snapshot a param
    p0_before = model.conv.weight.detach().clone()

    loss = train(
        data_loader=loader,
        model=model,
        criterion=crit,
        optimizer=opt,
        device="cpu",
        epoch=1,
        cfg={"use_wandb": False},
    )

    assert isinstance(loss, float)
    # weights should change
    p0_after = model.conv.weight.detach()
    assert not torch.allclose(p0_before, p0_after)