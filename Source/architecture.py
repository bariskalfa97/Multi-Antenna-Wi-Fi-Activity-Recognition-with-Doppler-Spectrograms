# PyTorch implementation of the supplied activity-classification diagram.
#
# Input: (batch, 1, 340, 100), using channel-first (NCHW) layout.
# The diagram does not specify padding or the number of activity classes.
# We use SAME padding (extra padding on bottom/right for even kernels) to
# align the branches, and require num_classes explicitly. Convolutions and
# the dense layer include biases. forward() returns logits for CrossEntropyLoss;
# use logits.softmax(dim=1) for probabilities or logits.argmax(dim=1) for labels.

import torch
from torch import nn


class ActivityCNN(nn.Module):
    # Three parallel branches, 1x1 fusion, flatten, dropout, dense output.
    #
    #     Example:
    #         model = ActivityCNN(num_classes=4)
    #         logits = model(torch.randn(2, 1, 340, 100))
    #     

    def __init__(self, num_classes: int):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")

        # Top branch: 1 x 340 x 100 -> 1 x 170 x 50.
        self.pool_branch = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle branch: 1 x 340 x 100 -> 5 x 170 x 50.
        self.conv_branch = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        # Bottom branch: 1 -> 3 -> 6 -> 9 channels.
        # SAME 2x2/S1 needs one zero on bottom/right; SAME 4x4/S2
        # on the even-sized input needs one zero on every side.
        self.deep_branch = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(3, 6, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(6, 9, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Channel concatenation yields (1 + 5 + 9) x 170 x 50.
        self.fusion = nn.Sequential(
            nn.Conv2d(15, 3, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(3 * 170 * 50, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Return flattened features before dropout and classification.
        #
        #         Contrastive pretraining uses this representation with a temporary
        #         projection head. The ordinary classifier still uses :meth:`forward`.
        #         
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 340, 100):
            raise ValueError(
                f"Expected input shape (batch, 1, 340, 100), got {tuple(x.shape)}"
            )
        features = torch.cat(
            (self.pool_branch(x), self.conv_branch(x), self.deep_branch(x)),
            dim=1,
        )
        features = self.fusion(features)
        return self.flatten(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return (batch, num_classes) unnormalized activity scores.
        return self.dense(self.dropout(self.forward_features(x)))
