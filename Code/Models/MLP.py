import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize an MLP model for precipitation prediction.

        Parameters:
        - input_size: Flattened input size (grid_height * grid_width * input_variables)
        - hidden_size: Number of hidden units in the hidden layers
        - output_size: Number of outputs (default: 1, for total precipitation)
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the MLP.
        - x: Input tensor with shape (batch_size, input_size)
        """
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x) 
        return x

