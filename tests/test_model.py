import torch
from src.model import MLP


def test_output_shape():
    model = MLP()
    x = torch.randn(5, 64)
    assert model(x).shape == (5, 10)


def test_deterministic_in_eval():
    model = MLP()
    model.eval()
    x = torch.randn(3, 64)
    assert torch.equal(model(x), model(x))
