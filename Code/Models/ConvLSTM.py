import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        #input_dim is the number of input channels, hidden_dim is the number of hidden state channels, bias determines if a bias term is added
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]) // 2, (kernel_size[1]) // 2  #same padding to preserve spatial dimensions
        self.bias = bias

        #convolution that outputs all 4 LSTM gates at once: input, forget, output, and candidate
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        #concatenate input and hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)

        #apply convolution to get all 4 gates in a single pass
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        #apply activations to gates
        i = torch.sigmoid(cc_i)  #input gate
        f = torch.sigmoid(cc_f)  #forget gate
        o = torch.sigmoid(cc_o)  #output gate
        g = torch.tanh(cc_g)     #candidate cell state

        #update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        #initialize h and c with zeros on the same device as conv layer
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        #expand hidden_dim and kernel_size to lists of length equal to num_layers
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

        cell_list = []
        for i in range(self.num_layers):
            #for the first layer, input dim is input_dim, otherwise it is previous layer's hidden dim
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        #convert (T, B, C, H, W) to (B, T, C, H, W) if batch_first is False
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        #initialize hidden state for all layers if not provided
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []  #stores the output for each layer
        last_state_list = []    #stores the final hidden and cell states for each layer

        seq_len = input_tensor.size(1)  #length of the time sequence
        cur_layer_input = input_tensor  #initial input to the first layer

        #loop over each ConvLSTM layer
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  #get initial hidden and cell states for the current layer
            output_inner = []  #will store outputs at each time step for current layer

            #loop over each time step
            for t in range(seq_len):
                #pass one time slice of the input to the ConvLSTM cell
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)  #collect the hidden state for this time step

            #stack all hidden states to form the output sequence for this layer
            layer_output = torch.stack(output_inner, dim=1)

            #set this layer's output as the input for the next layer
            cur_layer_input = layer_output

            #append final output and last (h, c) for this layer
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        #if return_all_layers is False, keep only the output and state of the last layer
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        #initialize hidden state for each layer
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        #check that kernel_size is a tuple or list of tuples
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        #if param is not a list, repeat it for each layer
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
