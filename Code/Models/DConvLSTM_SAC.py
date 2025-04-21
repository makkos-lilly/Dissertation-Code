import numpy as np
import torch
import torch.nn as nn

"""
DConvLSTM_SAC: Deformable ConvLSTM model with Spatial Attention Control (SAC)

This architecture alternates between standard ConvLSTM and Deformable ConvLSTM cells with SAC to enhance spatiotemporal learning, especially in precipitation or weather prediction tasks where spatial dynamics are complex.

The model incorporates learnable memory attention mechanisms (`m_t`) and deformable convolutions to adaptively focus on salient spatial regions. It uses two alternating cell lists (standard and deformable) during inference and switches between them at each time step using a toggle (`self.index`).

Parameters:
    num_layers (int): Number of stacked ConvLSTM/Deformable ConvLSTM layers.
    num_hidden (list[int]): Number of hidden channels in each layer.
    configs (object): Configuration object containing:
        - img_width: Original image width.
        - patch_size: Size to divide the image into patches (used to compute spatial resolution).
        - img_channel: Number of variables/channels
        - batch_size: Number of input sequences per batch.
        - total_length: Total number of time steps (input + prediction).
        - input_length: Number of past time steps to use as input.
        - device: Torch device
        - filter_size, stride, layer_norm: Standard LSTM hyperparameters.

Expected Input:
    frames_tensor (Tensor): Shape (batch, total_length, height, width, channels)
        - Assumes last dimension contains 4 channels: [precip, atm, moran1, moran2].
        - This tensor is internally split into:
            * 'frames' with channels [0, 1] (precip + atm),
            * 'moran' with channels [2, 3] (moran spatial stats).
    mask_true (Tensor): Shape (batch, total_length - input_length, height, width, channels)
        - Binary mask indicating which frames are known during prediction phase.
        - Used for scheduled sampling: blend true frames and predictions.

Output:
    next_frames (Tensor): Shape (batch, total_length - 1, height, width, 1)
        - Predicted future precipitation maps from time `input_length` to `total_length - 1`.

Notes:
    - Internal reshaping aligns tensors into (batch, time, channel, height, width) format.
    - Two versions of every cell are maintained (standard + deformable), and `self.index` alternates which is used every step.
    - Uses custom SAC-enhanced LSTM cells with memory gate `m_t` and learnable deformable convolution kernels.

"""

class DConvLSTM_SAC(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(DConvLSTM_SAC, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel  # flattened spatial + channel info
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.width = configs.img_width // configs.patch_size  # spatial resolution after patching
        self.index = 0  # controls whether to use deformable or regular cell on each step
        cell_list = []
        cell_list1 = []

        width = configs.img_width // configs.patch_size

        self.MSE_criterion = nn.MSELoss()  # for internal debugging or auxiliary loss

        for i in range(num_layers):
            in_channel = 1 if i == 0 else num_hidden[i - 1]  # input to first layer is single-channel frame, else previous layer output
            if i % 2 == 0:
                # alternate between deformable and standard cells
                cell_list.append(
                    DeformableConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                           configs.stride, configs.layer_norm)
                )
                cell_list1.append(
                    ConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                 configs.stride, configs.layer_norm)
                )
            else:
                cell_list.append(
                    ConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                 configs.stride, configs.layer_norm)
                )
                cell_list1.append(
                    DeformableConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                           configs.stride, configs.layer_norm)
                )

        self.cell_list = nn.ModuleList(cell_list)  # deformable-starting stack
        self.cell_list1 = nn.ModuleList(cell_list1)  # regular-starting stack

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], 1, kernel_size=1, stride=1, padding=0, bias=False)  # 1x1 conv to project to output

    def forward(self, frames_tensor, mask_true):

        # separate moran index channels (2 and 3)
        moran = frames_tensor[:, :, :, :, [2,3]].detach().cpu().numpy()
        moran = np.array(moran).reshape(self.configs.batch_size, self.configs.total_length, self.configs.img_width, self.configs.img_width, self.frame_channel)
        moran = torch.FloatTensor(moran).to(self.configs.device).permute(0, 1, 4, 2, 3).contiguous()  # shape: [B, T, C, H, W]

        # separate main physical variables (e.g., TP1, TP2)
        tp = frames_tensor[:, :, :, :, [0,1]].detach().cpu().numpy()
        tp = np.array(tp).reshape(self.configs.batch_size, self.configs.total_length, self.configs.img_width, self.configs.img_width, self.frame_channel)
        frames = torch.FloatTensor(tp).to(self.configs.device).permute(0, 1, 4, 2, 3).contiguous()  # shape: [B, T, C, H, W]

        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, C, H, W]

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []  # output sequence
        h_t = []  # list of hidden states
        c_t = []  # list of cell states
        m_t = []  # list of spatial memory maps

        # initialize all hidden/cell/memory states
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros1 = torch.zeros([batch, 1, height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            m_t.append(zeros1)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]  # ground truth input
                m = moran[:, t]
            else:
                # blend ground truth and predicted frame using mask
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                       (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            if self.index == 0:
                # use regular-starting stack first
                h_t[0], c_t[0], m_t[0] = self.cell_list1[0](net, h_t[0], c_t[0], m, m_t[0])
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], m_t[i] = self.cell_list1[i](h_t[i - 1], h_t[i], c_t[i], m_t[i - 1], m_t[i])

                x_gen = self.conv_last(h_t[self.num_layers - 1])  # final projection to 1-channel output
                next_frames.append(x_gen)
                self.index = 1

            else:
                # use deformable-starting stack
                h_t[0], c_t[0], m_t[0] = self.cell_list[0](net, h_t[0], c_t[0], m, m_t[0])
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], m_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t[i - 1], m_t[i])

                x_gen = self.conv_last(h_t[self.num_layers - 1])
                next_frames.append(x_gen)
                self.index = 0

        # [T, B, C, H, W] -> [B, T, H, W, C]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames


class ConvLSTM_SAC_Cell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTM_SAC_Cell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2  # Same padding
        self._forget_bias = 1.0  # Bias term added to forget gate

        # Convolution applied to input x_t
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )

        # Convolution applied to previous hidden state h_t
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )

        # Convolution applied to Moran maps m and m_t (2 channels → 3 outputs)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([3, width, width])
        )

        # Peephole connections (learned weights interacting with c_t)
        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcg = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()

    def forward(self, x_t, h_t, c_t, m, m_t):
        # Convolve x_t and h_t to get gate activations
        x_concat = self.conv_x(x_t).cuda()
        h_concat = self.conv_h(h_t).cuda()

        # Split x and h convolutions into gates
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        # Compute input gate
        i_t = torch.sigmoid(i_x + i_h + self.Wci * c_t)
        # Compute forget gate
        f_t = torch.sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)
        # Compute cell candidate
        g_t = torch.tanh(g_x + g_h)
        # Update cell state
        c_new = f_t * c_t + i_t * g_t
        # Compute output gate
        o_t = torch.sigmoid(o_x + o_h + self.Wco * c_new)
        # Update hidden state
        h_new = o_t * torch.tanh(c_new)

        # Process spatial memory inputs m and m_t
        combined = self.conv(torch.cat([m, m_t], dim=1))  # Shape: (batch, 3, H, W)
        mo, mg, mi = torch.split(combined, 1, dim=1)

        # Update memory map
        mi = torch.sigmoid(mi)
        m_new = (1 - mi) * m + mi * torch.tanh(mg)

        # Modulate hidden state with memory map
        h_new = (1 - torch.sigmoid(m_new)) * x_t + torch.sigmoid(m_new) * h_new

        return h_new, c_new, m_new

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        # Convolution applied to input x_t
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )

        # Convolution applied to previous hidden state h_t
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )

        # Peephole connections
        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcg = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()

    def forward(self, x_t, h_t, c_t):
        # Convolve input and hidden state
        x_concat = self.conv_x(x_t).cuda()
        h_concat = self.conv_h(h_t).cuda()

        # Split into gates
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        # Compute gates
        i_t = torch.sigmoid(i_x + i_h + self.Wci * c_t)
        f_t = torch.sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h + self.Wco * c_new)

        # Final hidden state
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """ 
        Deformable convolution layer (v1 or v2 if modulation=True)
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Zero-padding to maintain spatial size consistency
        self.zero_padding = nn.ZeroPad2d(padding)

        # Final convolution applied on sampled grid
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # Predict offsets for each position in the convolutional kernel
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)  # Start with zero offsets
        self.p_conv.register_backward_hook(self._set_lr)  # Slow down learning rate for offset layer

        self.modulation = modulation
        if modulation:
            # Predict modulation mask if deformable v2 is used
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=kernel_size, padding=padding, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # Reduce learning rate of gradients for stability
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # Predict offsets from input
        offset = self.p_conv(x)

        # If modulation is enabled, compute soft weights
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2  # Number of sampling points

        if self.padding:
            x = self.zero_padding(x)

        # Compute target sampling positions (with learned offsets)
        p = self._get_p(offset, dtype)  # shape: [B, 2N, H, W]
        p = p.permute(0, 2, 3, 1)  # [B, H, W, 2N]

        # Compute integer neighbors for bilinear interpolation
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # Clip positions to remain in valid index range
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # Bilinear interpolation weights
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:] - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N] - p[..., :N])) * (1 - (q_rb[..., N:] - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N] - p[..., :N])) * (1 - (q_lb[..., N:] - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N] - p[..., :N])) * (1 + (q_rt[..., N:] - p[..., N:]))

        # Sample values from 4 neighbors
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # Weighted sum of neighbor values
        x_offset = (
            g_lt.unsqueeze(1) * x_q_lt +
            g_rb.unsqueeze(1) * x_q_rb +
            g_lb.unsqueeze(1) * x_q_lb +
            g_rt.unsqueeze(1) * x_q_rt
        )

        # Apply modulation mask if enabled
        if self.modulation:
            m = m.permute(0, 2, 3, 1).unsqueeze(1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        # Reshape offset tensor into 2D spatial layout
        x_offset = self._reshape_x_offset(x_offset, ks)

        # Final 1x1 convolution on the transformed patch
        out = self.conv(x_offset)

        return out
