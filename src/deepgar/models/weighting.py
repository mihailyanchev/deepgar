"""Loss-weighting schemes for DeepGaR, shared by the TensorFlow and PyTorch models.

See ``deepgar.models.deepgar_skew`` for why ``EQUAL_WEIGHTING`` is the default.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeighting:
    """Weights of the (tilted loss, distribution loss) pair during training.

    The weights switch from ``initial`` to ``final`` at the end of the epoch
    that passes ``switch_at`` of the total epochs. ``final=None`` keeps the
    initial weights throughout.
    """

    initial: tuple[float, float]
    final: tuple[float, float] | None = None
    switch_at: float = 0.9

    def switches_after(self, epoch: int, total_epochs: int) -> bool:
        """Whether the weights switch to ``final`` at the end of ``epoch`` (0-based)."""
        return self.final is not None and epoch > int(total_epochs * self.switch_at)


# Used to produce the published results (see module docstring). Weights of 1.0
# reproduce the original unweighted sum exactly; under Adam this is equivalent
# to 0.5 / 0.5, since rescaling the loss by a constant leaves the updates unchanged.
EQUAL_WEIGHTING = LossWeighting(initial=(1.0, 1.0))

# The dynamic scheme described in the paper.
PAPER_DYNAMIC_WEIGHTING = LossWeighting(initial=(0.95, 0.05), final=(0.05, 0.95), switch_at=0.9)
