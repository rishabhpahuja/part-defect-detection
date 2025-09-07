import torch
from loss import DefectSegmentationLoss
import pytest
import yaml

cfg = yaml.safe_load(open("config.yaml"))

def make_binary_batch(B = 2, H = 32, W = 32):
    """
    Half-left foreground mask, logits_good aligns with targets, logits_bad inverts them minus a scalar.
    """
    targets = torch.zeros(B, cfg['data']['num_classes'], H, W)
    targets[:, :, : W // 2] = 1.0

    logits_good = torch.full((B, cfg['data']['num_classes'], H, W), -20.0)
    logits_good[:, :, :, : W // 2] = 20.0

    logits_bad = -logits_good - 5.0
    return logits_good, logits_bad, targets

@pytest.mark.parametrize("loss_type", cfg['loss']['loss_types'])
def test_binary_perfect_vs_wrong(loss_type):
    logits_good, logits_bad, targets = make_binary_batch(B = 1, H = 64, W = 64)

    crit = DefectSegmentationLoss(loss_directory = cfg['loss']['loss_types'], loss_type = loss_type)
    good = crit(logits_good, targets)
    bad = crit(logits_bad, targets)

    assert good < bad, f"{loss_type}: loss for good preds must be lower than for bad preds"

@pytest.mark.parametrize("loss_type", cfg['loss']['loss_types'])
def test_binary_scalar_and_backward(loss_type):
    B, H, W = 2, 32, 32
    logits = torch.randn(B, cfg['data']['num_classes'], H, W, requires_grad = True)
    targets = torch.randint(0, 2, (B, cfg['data']['num_classes'], H, W))

    crit = DefectSegmentationLoss(loss_directory = cfg['loss']['loss_types'],loss_type = loss_type)
    loss = crit(logits, targets)

    assert loss.ndim == 0, f"{loss_type} should return a scalar"
    assert torch.isfinite(loss), f"{loss_type} produced non-finite loss"
    loss.backward()
    assert logits.grad is not None