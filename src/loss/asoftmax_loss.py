import torch
import torch.nn.functional as F
from torch import nn


class ASoftmaxLoss(nn.Module):
    """
    Angular softmax loss for logits
    """

    def __init__(
        self,
        lambda_max=1000.0,
        lambda_min=5.0,
        lambda_decay=0.1,
    ):
        super().__init__()
        if not 0 <= lambda_min <= lambda_max:
            raise ValueError("Expected 0 <= lambda_min <= lambda_max")
        if lambda_decay < 0:
            raise ValueError("lambda_decay must be non-negative")
        self.lambda_max = float(lambda_max)
        self.lambda_min = float(lambda_min)
        self.lambda_decay = float(lambda_decay)
        self.register_buffer("iteration", torch.zeros((), dtype=torch.long))

    def forward(
        self,
        logits: torch.Tensor,
        margin_logits: torch.Tensor,
        labels: torch.Tensor,
        **batch,
    ):
        if logits.shape != margin_logits.shape:
            raise ValueError("logits and margin_logits must have the same shape")

        labels = labels.long()
        target_mask = F.one_hot(labels, num_classes=logits.shape[1]).to(
            dtype=torch.bool
        )

        if self.training:
            self.iteration.add_(1)
        angular_lambda = max(
            self.lambda_min,
            self.lambda_max / (1.0 + self.lambda_decay * float(self.iteration.item())),
        )
        annealed_margin_logits = (angular_lambda * logits + margin_logits) / (
            1.0 + angular_lambda
        )
        training_logits = torch.where(
            target_mask,
            annealed_margin_logits,
            logits,
        )

        return {"loss": F.cross_entropy(training_logits, labels)}
