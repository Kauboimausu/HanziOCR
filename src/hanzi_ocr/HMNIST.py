from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Flatten, Dropout, Linear, BatchNorm2d, BatchNorm1d
from torch.nn.init import kaiming_normal_, zeros_
import torch

class HMNISTModel(Module):
    def __init__(self, n_classes):
        super().__init__()

        # Conv
        self.conv1 = Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.norm1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.norm2 = BatchNorm2d(32)
        self.pool1 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = ReLU()
        self.norm3 = BatchNorm2d(64)
        self.pool2 = MaxPool2d(2)
        self.conv4 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = ReLU()
        self.norm4 = BatchNorm2d(128)
        self.pool3 = MaxPool2d(2)
        self.conv5 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = ReLU()
        self.norm5 = BatchNorm2d(256)

        # MLP 
        self.flatten = Flatten()
        self.linear1 = Linear(256 * 8 * 8, 128)
        self.relu6 = ReLU()
        self.dropout1 = Dropout(0.2)
        self.norm6 = BatchNorm1d(num_features=128)
        self.linear2 = Linear(128, 128)
        self.relu7 = ReLU()
        self.dropout2 = Dropout(0.2)
        self.norm7 = BatchNorm1d(num_features=128)
        self.dropout3 = Dropout(0.5)
        self.linear3 = Linear(128, n_classes)

        # Weight initialization
        for mod in self.modules():
            if isinstance(mod, Conv2d) or isinstance(mod, Linear):
                kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    zeros_(mod.bias)

    def forward(self, X):
        for _, mod in self.named_children():
            X = mod(X)
        return X

    def test_shape(self):
        with torch.no_grad():
            X = torch.randn(size=(2, 3, 64, 64))
            print(f"Initial shape: {X.shape}")
            for mod_name, mod in self.named_children():
                X = mod(X)
                print(f"{mod_name}: {X.shape}")