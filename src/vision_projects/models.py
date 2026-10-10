import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, size=16, classes=10, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size * size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, images):
        return self.layers(images)


class LeNet(nn.Module):
    def __init__(self, channels=1, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(24 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, classes)
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class InceptionBlock(nn.Module):
    def __init__(self, channels, width=8):
        super().__init__()
        self.pointwise = nn.Sequential(nn.Conv2d(channels, width, 1), nn.ReLU())
        self.local = nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU()
        )
        self.wide = nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.ReLU(), nn.Conv2d(width, width, 5, padding=2), nn.ReLU()
        )
        self.pooled = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1), nn.Conv2d(channels, width, 1), nn.ReLU()
        )

    def forward(self, features):
        return torch.cat(
            [self.pointwise(features), self.local(features), self.wide(features), self.pooled(features)],
            dim=1,
        )


class InceptionCNN(nn.Module):
    def __init__(self, channels=3, classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            InceptionBlock(16),
            nn.MaxPool2d(2),
            InceptionBlock(32),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, images):
        return self.classifier(self.features(images))


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.decoder(self.encoder(images))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(nn.Conv2d(24, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, images):
        skip = self.encoder(images)
        coarse = self.bottleneck(F.max_pool2d(skip, 2))
        upsampled = F.interpolate(coarse, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.decoder(torch.cat([skip, upsampled], dim=1))


class EmbeddingCNN(nn.Module):
    def __init__(self, dimensions=16):
        super().__init__()
        self.encoder = LeNet(classes=dimensions)

    def forward(self, images):
        return F.normalize(self.encoder(images), dim=-1)


class SuperResolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 1, 3, padding=1),
        )

    def forward(self, low_resolution):
        enlarged = F.interpolate(low_resolution, scale_factor=2, mode="bicubic", align_corners=False)
        return enlarged + self.residual(enlarged)


class PatchTransformer(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.patches = nn.Conv2d(1, 32, kernel_size=4, stride=4)
        self.positions = nn.Parameter(torch.randn(1, 16, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.classifier = nn.Linear(32, classes)

    def forward(self, images):
        tokens = self.patches(images).flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens + self.positions)
        return self.classifier(encoded.mean(dim=1))


class FrozenResNetClassifier(nn.Module):
    def __init__(self, backbone, classes=10):
        super().__init__()
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.classifier = nn.Linear(backbone.fc.in_features, classes)

    def forward(self, images):
        # Frozen BatchNorm statistics must stay fixed even inside a training epoch.
        self.encoder.eval()
        return self.classifier(self.encoder(images).flatten(1))
