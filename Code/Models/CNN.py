
import torch
import torch.nn as nn


# -----------------------------
class Baseline3DCNNwithMLP(nn.Module):
    def __init__(self, input_channels, output_channels, time_steps, height, width, new_height, new_width, batch_first=True):
        """
        A baseline 3D CNN architecture followed by an MLP for spatiotemporal regression tasks.

        Parameters:
        - input_channels: number of input feature channels (e.g. number of atmospheric variables)
        - output_channels: number of channels in output (currently unused, can be used for multi-channel output)
        - time_steps: number of input time steps
        - height, width: spatial dimensions of input grid
        - new_height, new_width: target spatial size of output grid
        - batch_first: if True, input is in the form [batch, channels, time, height, width]
        """
        super(Baseline3DCNNwithMLP, self).__init__()
        
        # 1st 3D convolution block
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # only downscale spatial dims

        # 2nd 3D convolution block
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # downscale both time and space

        # 3rd 3D convolution block
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # downscale further

        # compute flattened size after convolutions and pooling
        flat_size = (time_steps // 2 // 2) * (height // 8) * (width // 8) * 256  # final 3D feature volume flattened

        self.flatten = nn.Flatten()

        # fully connected layers (MLP)
        self.fc1 = nn.Linear(flat_size, 1024)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, new_height * new_width)  # final output reshaped into spatial grid

        # store the final output shape
        self.output_reshape = (new_height, new_width)

    def forward(self, x):
        # pass through 1st convolution block
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        # pass through 2nd convolution block
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        # pass through 3rd convolution block
        x = torch.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)

        # flatten the 3D feature volume to 1D vector
        x = self.flatten(x)

        # pass through MLP
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        # reshape final output into (batch, 1, height, width) grid format
        x = x.view(-1, 1, *self.output_reshape)  
        return x
