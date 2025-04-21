import torch.nn as nn
import torch

"""
A periodicity-aware variant of ConvLSTM that uses a custom activation function
inside the memory cell to better model periodic temporal dynamics (e.g., daily or seasonal cycles).

Each ConvLSTMCell includes a sinusoidal modulation in its candidate memory update gate,
which helps encode cyclic patterns in the data.

Parameters:
    input_dim: Number of channels in the input tensor.
    hidden_dim: Number of hidden channels (can be an int or list of ints for multiple layers).
    kernel_size: Size of the convolutional kernel (tuple or list of tuples).
    num_layers: Number of ConvLSTM layers to stack.
    batch_first: Whether input tensors have batch dimension first (default: False).
    bias: Whether or not to include bias in the convolutional operations (default: True).
    return_all_layers: If True, returns outputs for all layers. If False, returns only the last layer.

Input:
    A 5D tensor of shape (B, T, C, H, W) if batch_first is True,
    otherwise (T, B, C, H, W), where:
        B = batch size,
        T = sequence length (timesteps),
        C = number of input channels,
        H, W = spatial dimensions.

Output:
    A tuple with two elements:
        - layer_output_list: list of output tensors (one per layer), each of shape (B, T, hidden_dim, H, W)
        - last_state_list: list of (h, c) tuples (one per layer), where:
            h: final hidden state, shape (B, hidden_dim, H, W)
            c: final cell state, shape (B, hidden_dim, H, W)
"""


# ------------------------------
# Periodicity-aware activation
# ------------------------------
class PeriodicityActivation(nn.Module):
    def __init__(self):
        super(PeriodicityActivation, self).__init__()
        # initialize 'a' as a learnable parameter in range [-2pi, 2pi]
        self.a = nn.Parameter(torch.empty(1).uniform_(-2 * torch.pi, 2 * torch.pi), requires_grad=True)

    def forward(self, x):
        # clamp a within bounds to ensure stability during training
        a_clamped = torch.clamp(self.a, min=-2 * torch.pi, max=2 * torch.pi)
        # apply sinusoidal modulation to input
        return x + a_clamped * torch.sin(x / a_clamped) ** 2


# ------------------------------
# ConvLSTM cell with periodic activation
# ------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]) // 2, (kernel_size[1]) // 2  # use same padding
        self.bias = bias

        # single convolution layer outputs all 4 gates (input, forget, output, candidate)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # add periodicity-aware activation for candidate gate
        self.periodicity_activation = PeriodicityActivation()

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # unpack hidden and cell state

        # concatenate input and previous hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # apply convolution to compute all 4 gates
        combined_conv = self.conv(combined)

        # split into input, forget, output, and candidate gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)                           # input gate
        f = torch.sigmoid(cc_f)                           # forget gate
        o = torch.sigmoid(cc_o)                           # output gate
        g = self.periodicity_activation(torch.tanh(cc_g)) # periodicity-aware candidate gate

        # update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # return zero-initialized hidden and cell states on the correct device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# ------------------------------
# ConvLSTMwith periodic activation
# ------------------------------
class ConvLSTM_PERIODIC(nn.Module):
    """
    Modified version of ConvLSTM with periodicity-aware ConvLSTMCell.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM_PERIODIC, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # ensure kernel_size and hidden_dim are lists with length equal to num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # create list of ConvLSTMCells (stacked layers)
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        # permute dimensions if input is not batch-first
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)  # (t, b, c, h, w) -> (b, t, c, h, w)

        b, _, _, h, w = input_tensor.size()

        # initialize hidden state if not provided
        if hidden_state is not None:
            raise NotImplementedError("Stateful mode not implemented.")
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []  # list of outputs from each layer
        last_state_list = []    # list of (h, c) tuples from each layer

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor  # input to the first layer

        # iterate over stacked layers
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # get hidden state for this layer
            output_inner = []

            # iterate over all time steps
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)  # save hidden state at each timestep

            layer_output = torch.stack(output_inner, dim=1)  # stack across time
            cur_layer_input = layer_output  # feed output to next layer

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # optionally return only the last layer
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        # initialize hidden and cell states for all layers
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        # check that kernel size is a tuple or list of tuples
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        # repeat scalar param into a list if needed
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
