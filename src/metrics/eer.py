import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.calculate_eer import compute_eer


class EERMetric(BaseMetric):
    """Equal error rate accumulated over a complete dataset partition."""

    def __init__(self, positive_label=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_label = positive_label
        self.reset()

    def reset(self):
        self._scores = []
        self._labels = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        scores = logits[:, 1] - logits[:, 0]
        self._scores.append(scores.detach().cpu())
        self._labels.append(labels.detach().cpu())
        return None

    def compute(self):
        scores = torch.cat(self._scores).numpy()
        labels = torch.cat(self._labels).numpy()
        bonafide_scores = scores[labels == self.positive_label]
        spoof_scores = scores[labels != self.positive_label]

        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        return float(eer * 100.0)
