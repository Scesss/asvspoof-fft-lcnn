import torch
import torch.nn.functional as F
from torch import nn


class ASoftmaxLoss(nn.Module):
    """
    Angular softmax loss for logits
    """

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
        training_logits = torch.where(target_mask, margin_logits, logits)

        return {"loss": F.cross_entropy(training_logits, labels)}
