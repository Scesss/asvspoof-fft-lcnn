import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential


class AngularLinear(nn.Module):
    def __init__(self, in_features=80, out_features=2, m=4):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.m = int(m)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, embeddings):
        embedding_norm = torch.linalg.vector_norm(
            embeddings, ord=2, dim=1, keepdim=True
        ).clamp_min(1e-7)
        normalized_embeddings = embeddings / embedding_norm
        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        cos_theta = F.linear(normalized_embeddings, normalized_weight)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)

        k = torch.floor(self.m * theta / torch.pi).detach()
        sign = torch.where(
            k.to(torch.int64) % 2 == 0,
            torch.ones_like(k),
            -torch.ones_like(k),
        )
        phi_theta = sign * torch.cos(self.m * theta) - 2.0 * k

        logits = embedding_norm * cos_theta
        margin_logits = embedding_norm * phi_theta
        return logits, margin_logits


class MFM(nn.Module):
    def forward(self, x):
        first, second = torch.chunk(x, chunks=2, dim=1)
        result = torch.maximum(first, second)
        return result


class LCNNModel(nn.Module):
    def __init__(self, angular_margin=4, dropout=0.75):
        super().__init__()

        self.net = Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MFM(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MFM(),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MFM(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MFM(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.embedding = nn.Sequential(
            nn.Linear(in_features=32 * 53 * 37, out_features=160),
            MFM(),
            nn.BatchNorm1d(num_features=80),
            nn.Dropout(p=dropout),
        )
        self.angular_head = AngularLinear(
            in_features=80,
            out_features=2,
            m=angular_margin,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Kaiming-normal initialization to the LCNN layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, audio, **batch):
        """
        Model forward method.

        Args:
            audio: input audio
        Returns:
            output (dict): output dict containing logits.
        """
        features = self.net(audio)
        features = torch.flatten(features, start_dim=1)
        embeddings = self.embedding(features)
        logits, margin_logits = self.angular_head(embeddings)
        return {
            "embeddings": embeddings,
            "logits": logits,
            "margin_logits": margin_logits,
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
