import copy

import pytest
import torch
from torch import nn

from vision_projects.models import (
    ConvAutoencoder,
    EmbeddingCNN,
    FrozenResNetClassifier,
    InceptionCNN,
    LeNet,
    MLP,
    PatchTransformer,
    SuperResolution,
    TinyUNet,
)
from vision_projects.training import TrainConfig, Trainer


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(1031)
    torch.set_num_threads(2)


@pytest.mark.parametrize("constructor", [MLP, LeNet, PatchTransformer])
def test_classifier_backward_and_restore(constructor):
    model = constructor()
    images = torch.rand(3, 1, 16, 16)
    labels = torch.tensor([0, 1, 9])
    logits = model(images)
    assert logits.shape == (3, 10)
    nn.functional.cross_entropy(logits, labels).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    restored = constructor()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    model.eval()
    restored.eval()
    assert torch.equal(model(images), restored(images))


def test_multiscale_and_reconstruction_shapes():
    assert InceptionCNN()(torch.rand(2, 3, 64, 64)).shape == (2, 6)
    images = torch.rand(2, 1, 16, 16)
    assert ConvAutoencoder()(images).shape == images.shape
    assert TinyUNet()(torch.rand(2, 1, 31, 33)).shape == (2, 1, 31, 33)
    assert SuperResolution()(images).shape == (2, 1, 32, 32)


def test_embedding_normalization():
    embeddings = EmbeddingCNN()(torch.rand(4, 1, 16, 16))
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(4), atol=1e-6)


def test_frozen_resnet_keeps_batchnorm_statistics():
    from torchvision.models import resnet18

    model = FrozenResNetClassifier(resnet18(weights=None))
    model.train()
    batch_norm = next(layer for layer in model.encoder.modules() if isinstance(layer, nn.BatchNorm2d))
    running_mean = batch_norm.running_mean.clone()
    model(torch.rand(2, 3, 64, 64)).sum().backward()
    assert torch.equal(batch_norm.running_mean, running_mean)
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert model.classifier.weight.grad.abs().sum() > 0


def test_trainer_restores_best_validation_checkpoint():
    model = nn.Linear(1, 1)

    class ScriptedTrainer(Trainer):
        def train_epoch(self, loader):
            with torch.no_grad():
                self.model.weight.fill_(float(loader[0]))
            loader[0] += 1
            return 0.0

        def evaluate_loss(self, loader):
            return abs(float(self.model.weight.detach()) - 2)

    trainer = ScriptedTrainer(model, nn.MSELoss(), TrainConfig(epochs=4))
    trainer.fit([1], None)
    assert model.weight.item() == 2
    assert not model.training


def test_empty_loader_fails_clearly():
    trainer = Trainer(nn.Linear(1, 1), nn.MSELoss(), TrainConfig())
    with pytest.raises(ValueError, match="empty"):
        trainer.evaluate_loss([])
