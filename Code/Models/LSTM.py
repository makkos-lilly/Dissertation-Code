import torch
import torch.nn as nn

"""Parameters:
    input_size (int): Number of input features per grid point per time step.
    hidden_size (int): Number of hidden units in each LSTM layer.
    num_layers (int): Number of stacked LSTM layers.
    output_size (int): Number of output features per grid point (e.g., 1 for scalar regression).
    grid_width (int): Width of the spatial grid.
    grid_height (int): Height of the spatial grid.

Expected Input Shape:
    x: Tensor of shape (batch_size, sequence_length, input_size)
       - Note: 'input_size' should be the total number of variables per time step across all grid points.
       - If you have `V` variables and `G = grid_width * grid_height` grid points, 'input_size = V * G'.

Output Shape:
    out: Tensor of shape (batch_size, output_size * grid_width * grid_height)
         - Typically reshaped externally to (batch_size, output_size, grid_height, grid_width)

Example:
    model = LSTM(input_size=1260, hidden_size=128, num_layers=2, output_size=1, grid_width=81, grid_height=97)
    x = torch.randn(16, 12, 1260)  # batch of 16, 12 time steps, 1260 features (e.g. 15 variables over 81x97 grid)
    y = model(x)
    y.shape  # torch.Size([16, 7857]) for 81*97 = 7857 grid points
"""

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, grid_width, grid_height):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.grid_points = grid_width * grid_height  # total number of grid cells

        # define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # fully connected layer to map hidden output to output for each grid cell
        self.fc = nn.Linear(hidden_size, output_size * self.grid_points)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        batch_size = x.size(0)

        # run the LSTM over the input sequence
        out, _ = self.lstm(x)  # out shape: (batch_size, sequence_length, hidden_size)

        # select only the output at the final time step
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # project the hidden state to the full output grid
        out = self.fc(out)  # shape: (batch_size, output_size * grid_points)

        # reshape the output to match expected flattened spatial grid format
        out = out.view(batch_size, -1)  # shape: (batch_size, grid_points) for output_size = 1
        return out