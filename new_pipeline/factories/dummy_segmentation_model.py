import torch
import torch.nn as nn


class DummySegmentationModel(nn.Module):
    """
    Dummy segmentation model used for testing pipeline architecture.

    Input:
        x: Tensor of shape [B, C, H, W]

    Output:
        Tensor of shape [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        constant_value: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.constant_value = constant_value

        # Important:
        # Gives the model at least one parameter so optimizers work.
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        output = torch.full(
            size=(batch_size, self.num_classes, height, width),
            fill_value=self.constant_value,
            device=x.device,
            dtype=x.dtype,
        )

        # Connect output to dummy_param so loss.backward() works.
        return output + self.dummy_param