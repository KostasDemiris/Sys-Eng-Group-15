import torch.nn as nn


# A really basic prototype model so that we can work on the rest of the pipeline.
class BasicCrosswalkDetector(nn.Module):
    def __init__(self, class_size=2, proposal_number=4):
        super().__init__()
        self.class_size = class_size  # In this basic example it's yes or no (1, 0)
        self.proposal_number = proposal_number
        # This refers to the max. num of potential crosswalks we accept
        # - not the perfect solution, but we will improve on it. We need to apply NMS to extract the actual predictions

        # RGB images have 3 input channels, the rest of the parameters are relatively default - we can finetune later
        self.first_convolutional_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.second_convolutional_layer = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.third_convolutional_layer = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fully_connected_layer = nn.Linear(64 * 32 * 32, 1024)

        self.bounding_box_head = nn.Linear(1024, 4 * self.proposal_number)
        self.class_label_head = nn.Linear(1024, self.class_size * self.proposal_number)

    def forward(self, x):
        x = self.first_convolutional_layer(x)
        x = nn.LeakyReLU(x)

        x = self.second_convolutional_layer(x)
        x = nn.MaxPool2d(x)

        x = self.third_convolutional_layer(x)
        x = nn.MaxPool2d(x)

        x = x.view(x.size(0), -1)
        # Flattens the feature maps to pass into the fully connected layer - from 4D to 2D (batch, lin_tensor)

        x = self.fully_connected_layer(x)
        x = nn.LeakyReLU(x)

        bounding_box_predictions = self.bounding_box_head(x)
        class_label_predictions = self.class_label_head(x)

        return bounding_box_predictions, class_label_predictions

