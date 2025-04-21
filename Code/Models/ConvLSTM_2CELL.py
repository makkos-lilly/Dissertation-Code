import torch.nn as nn
import torch
import torch.nn.init as init

"""
MultiPeriodConvLSTM is a stacked ConvLSTM architecture that explicitly models multiple temporal periodicities—such as daily and yearly cycles—using two separate recurrent branches per layer.

Each layer consists of two ConvLSTM cells: one for capturing short-term periodicity (daily variations) and another for long-term periodicity (yearly trends). 
Their outputs are adaptively combined using learnable weights. 

Parameters:
    input_dim_daily: Number of channels in the daily input tensor.
    input_dim_yearly: Number of channels in the yearly input tensor.
    hidden_dim: Number of output channels in each ConvLSTM cell (can be a list for stacked layers).
    kernel_size: Size of the convolutional kernel (tuple or list of tuples).
    num_layers: Number of stacked ConvLSTM layers.
    batch_first: Whether the input tensor has batch dimension first (B, T, C, H, W) or not (T, B, C, H, W).
    bias: Whether or not to include bias in the convolutional operations.
    return_all_layers: If True, returns output from all layers. If False, returns only the last layer.

Input:
    Two 5D tensors: input_daily and input_yearly, each of shape (B, T, C, H, W) if batch_first=True,
    or (T, B, C, H, W) otherwise.

Output:
    A tuple with two elements:
        0 - layer_output_list: list of output tensors from each layer, each of shape (B, T, hidden_dim, H, W)
        1 - last_state_list: list of tuples containing the final (hidden, cell) states for both daily and yearly cells per layer.

"""

#--------------------------------------
# Custom periodicity activation
#--------------------------------------
class PeriodicityActivation(nn.Module):
    def __init__(self):
        super(PeriodicityActivation, self).__init__()
        # learnable periodicity parameter a, randomly initialized between -2pi and 2pi
        self.a = nn.Parameter(torch.empty(1).uniform_(-2 * torch.pi, 2 * torch.pi), requires_grad=True)

    def forward(self, x):
        # clamp a to ensure it stays in a meaningful range for sin(x/a)
        a_clamped = torch.clamp(self.a, min=-2 * torch.pi, max=2 * torch.pi)
        # apply sinusoidal modulation to enhance periodic features
        return x + a_clamped * torch.sin(x / a_clamped) ** 2


#--------------------------------------
# Standard cell with periodic activation
#--------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)  # same padding
        self.bias = bias

        # convolution for input, forget, output, and candidate gates
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # use periodicity-aware nonlinearity in the cell
        self.periodicity_activation = PeriodicityActivation()

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # concatenate input and previous hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # apply convolution to combined tensor
        combined_conv = self.conv(combined)

        # split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)           # input gate
        f = torch.sigmoid(cc_f)           # forget gate
        o = torch.sigmoid(cc_o)           # output gate
        g = self.periodicity_activation(torch.tanh(cc_g))  # modulated cell input

        # cell state update
        c_next = f * c_cur + i * g
        # hidden state update
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        # initialize hidden and cell states with zeros
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


#--------------------------------------
# structure for modelling both daily and yearly periodicity
#--------------------------------------
class MultiPeriodConvLSTMCell(nn.Module):
    def __init__(self, input_dim_daily, input_dim_yearly, hidden_dim, kernel_size, bias=True):
        super(MultiPeriodConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        # separate ConvLSTM cells for daily and yearly inputs
        self.daily_cell = ConvLSTMCell(input_dim=input_dim_daily, hidden_dim=hidden_dim,
                                       kernel_size=kernel_size, bias=bias)
        self.yearly_cell = ConvLSTMCell(input_dim=input_dim_yearly, hidden_dim=hidden_dim,
                                        kernel_size=kernel_size, bias=bias)

        # learnable weights to mix daily and yearly states
        self.alpha_h = nn.Parameter(torch.rand(1))  # for hidden state h
        self.beta_h = nn.Parameter(torch.rand(1))
        self.alpha_c = nn.Parameter(torch.rand(1))  # for cell state c
        self.beta_c = nn.Parameter(torch.rand(1))

    def forward(self, input_daily, input_yearly, state_daily, state_yearly):
        # run both ConvLSTM cells independently
        h_daily, c_daily = self.daily_cell(input_daily, state_daily)
        h_yearly, c_yearly = self.yearly_cell(input_yearly, state_yearly)

        # normalize learned weights using sigmoid to ensure smooth 0-1 blending
        alpha_h_norm = torch.sigmoid(self.alpha_h)
        beta_h_norm = torch.sigmoid(self.beta_h)
        alpha_c_norm = torch.sigmoid(self.alpha_c)
        beta_c_norm = torch.sigmoid(self.beta_c)

        # blend hidden and cell states from both periodic branches
        h_combined = alpha_h_norm * h_daily + beta_h_norm * h_yearly
        c_combined = alpha_c_norm * c_daily + beta_c_norm * c_yearly

        return h_combined, c_combined

    def init_hidden(self, batch_size, height, width):
        # initialize both daily and yearly branches identically
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.daily_cell.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.daily_cell.conv.weight.device))


#--------------------------------------
# Full MultiPeriodConvLSTM Model
#--------------------------------------
class MultiPeriodConvLSTM(nn.Module):
    def __init__(self, input_dim_daily, input_dim_yearly, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(MultiPeriodConvLSTM, self).__init__()
        self.input_dim_daily = input_dim_daily
        self.input_dim_yearly = input_dim_yearly
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # stack multiple multi-period cells
        self.cell_list = nn.ModuleList([
            MultiPeriodConvLSTMCell(
                input_dim_daily=self.input_dim_daily if i == 0 else self.hidden_dim[i - 1],
                input_dim_yearly=self.input_dim_yearly if i == 0 else self.hidden_dim[i - 1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size,
                bias=self.bias
            ) for i in range(self.num_layers)
        ])

    def forward(self, input_daily, input_yearly, hidden_state=None):
        # rearrange inputs if batch is not first
        if not self.batch_first:
            input_daily = input_daily.permute(1, 0, 2, 3, 4)
            input_yearly = input_yearly.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_daily.size()

        # initialize hidden states for each layer if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []  # list of outputs for each layer
        last_state_list = []   # final states of all layers

        # input to layer 0 is raw input; for others, it's the output from previous layer
        cur_layer_input_daily = input_daily
        cur_layer_input_yearly = input_yearly

        # loop over all layers
        for layer_idx in range(self.num_layers):
            # get the hidden state for both daily and yearly branches
            (h_daily, c_daily), (h_yearly, c_yearly) = hidden_state[layer_idx]
            output_inner = []  # stores outputs at each timestep for this layer

            # loop over all time steps
            for t in range(seq_len):
                # apply multi-period cell at time t
                h_combined, c_combined = self.cell_list[layer_idx](
                    cur_layer_input_daily[:, t],
                    cur_layer_input_yearly[:, t],
                    (h_daily, c_daily),
                    (h_yearly, c_yearly)
                )

                # update states and append output
                h_daily, c_daily = h_combined, c_combined
                output_inner.append(h_combined)

            # stack temporal outputs to form output for current layer
            layer_output = torch.stack(output_inner, dim=1)

            layer_output_list.append(layer_output)
            last_state_list.append(((h_combined, c_combined), (h_yearly, c_yearly)))

            # set current layer's output as input to the next layer
            cur_layer_input_daily = layer_output
            cur_layer_input_yearly = layer_output  # used for both daily/yearly in higher layers

        if not self.return_all_layers:
            return layer_output_list[-1], last_state_list[-1]
        else:
            return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        height, width = image_size
        # initialize hidden states for all layers
        return [
            (
                self.cell_list[i].init_hidden(batch_size, height, width),
                self.cell_list[i].init_hidden(batch_size, height, width)
            ) for i in range(self.num_layers)
        ]
