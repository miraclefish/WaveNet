import torch
import math
from torch import nn
import torch.nn.functional as F
import pywt
from torch.autograd import Variable


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction

        self.pad = nn.ReflectionPad1d((0, 1))

    def forward(self, x):
        """Returns the odd and even part"""
        if x.size(2) % 2 == 1:
            x = self.pad(x)
        return x[:, :, ::2], x[:, :, 1::2]


class Haar(nn.Module):
    def __init__(self, groups, reg_details, reg_approx):
        super(Haar, self).__init__()

        self.groups = groups
        self.reg_details = reg_details
        self.reg_approx = reg_approx
        self.split = Splitting()

    def forward(self, x):

        (x_even, x_odd) = self.split(x)
        x_ds = (x_even + x_odd) / 2

        input_x = x.cpu().detach().numpy()

        L, H = pywt.dwt(input_x, 'db1')
        approx = get_variable(torch.from_numpy(L)).to(device=x.device)
        details = get_variable(torch.from_numpy(H)).to(device=x.device)
        # approx = approx.permute(0, 2, 1)
        # details = details.permute(0, 2, 1)

        rd = self.reg_details * details.abs().mean()

        # rc = self.reg_approx * torch.dist(approx.mean(), x.mean(), p=2)
        rc = self.reg_approx * torch.pow((approx - x_ds), 2).sum(dim=-1).sqrt().mean()
        # rc = self.reg_approx * torch.dist(L.mean(dim=-1), x.mean(dim=-1), p=2)
        # rc = self.reg_approx * (L.mean(dim=-1) - x.mean(dim=-1)).abs().mean()

        r = rd + rc

        return approx, r, details


def conv5x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, disable_conv=True):
        super(BottleneckBlock, self).__init__()
        self.disable_conv = disable_conv
        if not self.disable_conv:
            self.conv1 = conv1x1(in_planes, out_planes)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1)].to(x.device)
        return x


class TransEncoder(nn.Module):

    def __init__(self, d_model, num_layers, pe=False):
        super(TransEncoder, self).__init__()
        self.pe = pe
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        if pe:
            self.ps_encoding = PositionalEncoding(d_model=d_model, max_len=625)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.pe:
            x = self.ps_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        return x


class TransDecoder(nn.Module):

    def __init__(self, d_model, num_layers, pe=False):
        super(TransDecoder, self).__init__()

        self.pe = pe
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
        if pe:
            self.ps_encoding = PositionalEncoding(d_model=d_model, max_len=625)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x, x_skip):
        x = x.permute(0, 2, 1)
        x_skip = x_skip.permute(0, 2, 1)
        if self.pe:
            x = self.ps_encoding(x)
            x_skip = self.ps_encoding(x_skip)

        out = self.transformer_decoder(x, x_skip)
        out = out.permute(0, 2, 1)

        return out


class EncoderBlockOnReconstruction(nn.Module):

    def __init__(self, d_model, num_trans_layers, in_planes, stride=1, pe=False):
        super(EncoderBlockOnReconstruction, self).__init__()

        self.stride = stride
        self.patch_len = stride * 2

        if self.stride > 1:
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.linear = nn.Linear(self.patch_len, 1)
        self.conv = conv5x1(in_planes=in_planes, out_planes=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.elu = nn.ELU()
        self.trans_encoder = TransEncoder(d_model=d_model, num_layers=num_trans_layers, pe=pe)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        if self.stride > 1:
            x = self.padding_layer(x)
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [bs x c x patch_num x patch_len]
            x = self.linear(x).squeeze(-1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.trans_encoder(x)  # x: [bs x c x d_model x patch_num]

        return x


class DecoderBlockOnReconstruction(nn.Module):

    def __init__(self, d_model, num_trans_layers, out_planes, stride=1, pe=False):
        super(DecoderBlockOnReconstruction, self).__init__()
        self.d_model = d_model
        self.stride = stride
        self.patch_len = stride * 2

        if self.stride > 1:
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.linear_unfold = nn.Linear(self.patch_len, 1)
            self.linear_fold = nn.Linear(1, self.patch_len)
            self.pool_fold = nn.MaxPool1d(self.patch_len, 2, padding=self.stride)

        self.conv2 = nn.Conv1d(out_planes, d_model, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(d_model)
        self.elu2 = nn.ELU()

        self.trans_decoder = TransDecoder(d_model=d_model, num_layers=num_trans_layers, pe=pe)

        self.conv_out = nn.Conv1d(d_model, out_planes, kernel_size=3, padding='same')
        self.bn_out = nn.BatchNorm1d(out_planes)
        self.elu_out = nn.ELU()

    def forward(self, x, x_skip):

        B, C, L = x_skip.shape
        if self.stride > 1:
            '''
            x : [bs x c x d_model x patch_num_pre]
            x_skip : [bs x c x d_model x patch_num_cur]
            '''
            x_skip = self.padding_layer(x_skip)
            x_skip = x_skip.unfold(dimension=-1, size=self.patch_len,
                                   step=self.stride)  # x: [bs x c x patch_num x patch_len]
            x_skip = self.linear_unfold(x_skip).squeeze(-1)

        x_skip = self.elu2(self.bn2(self.conv2(x_skip)))

        # x = x.reshape(-1, self.C, self.d_model, x.size(-1))
        # x = x.reshape(-1, self.d_model, x.size(-1))
        # x_skip = x_skip.reshape(-1, self.C, self.d_model, x_skip.size(-1))
        # x_skip = x_skip.reshape(-1, self.d_model, x_skip.size(-1))

        # x_cat = torch.cat((x, x_skip), dim=1)
        out = self.trans_decoder(x, x_skip)

        if self.stride > 1:
            out = self.linear_fold(out.unsqueeze(-1))
            out = self.pool_fold(out.reshape(B, self.d_model, -1))
            out = F.adaptive_avg_pool1d(out, L)

        out = self.elu_out(self.bn_out(self.conv_out(out)))
        # out = out.reshape(-1 , self.d_model, out.size(-1))
        return out


class DecoderBlock(nn.Module):

    def __init__(self, in_planes, out_planes, d_model=64, num_trans_layers=1, stride=1, block_type='CNN'):
        super(DecoderBlock, self).__init__()

        self.block_type = block_type
        self.conv1 = conv5x1(in_planes=in_planes, out_planes=out_planes)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.elu1 = nn.ELU()

        if self.block_type == 'CNN':
            self.conv2 = conv5x1(in_planes=out_planes * 2, out_planes=out_planes)
            self.bn2 = nn.BatchNorm1d(out_planes)
            self.elu2 = nn.ELU()
        if self.block_type == 'Trans':
            self.encoder_layer = EncoderBlockOnReconstruction(d_model, num_trans_layers, in_planes=out_planes,
                                                              stride=stride, pe=True)
            self.decoder_layer = DecoderBlockOnReconstruction(d_model, num_trans_layers, out_planes=out_planes,
                                                              stride=stride, pe=True)

    def forward(self, x, x_skip):

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.elu1(self.bn1(self.conv1(x)))
        x = nn.AdaptiveAvgPool1d(x_skip.size(-1))(x)

        # out = x + x_skip
        output_x = None
        if self.block_type == 'CNN':
            x_cat = torch.cat((x, x_skip), dim=1)
            output_x = self.elu2(self.bn2(self.conv2(x_cat)))

        if self.block_type == 'Trans':
            x = self.encoder_layer(x)
            output_x = self.decoder_layer(x, x_skip)
        return output_x


class LiftingScheme(nn.Module):
    def __init__(self, in_planes, dropout=0, groups=1, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = True

        # kernel_size = k_size
        kernel_size = 3
        dilation = 1

        pad = dilation * (kernel_size - 1) // 2 + 1
        # pad = k_size // 2 # 2 1 0 0

        self.split = Splitting()

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, groups=groups, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, groups=groups, stride=1),
                nn.Tanh()
            ]
        else:
            size_hidden = 2

            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, groups=groups, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, groups=groups, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, groups=groups, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, groups=groups, stride=1),
                nn.Tanh()
            ]
            if self.modified:
                modules_phi += [
                    nn.ReplicationPad1d(pad),
                    nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                              kernel_size=kernel_size, dilation=dilation, groups=groups, stride=1),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv1d(in_planes * size_hidden, in_planes,
                              kernel_size=kernel_size, groups=groups, stride=1),
                    nn.Tanh()
                ]
                modules_psi += [
                    nn.ReplicationPad1d(pad),
                    nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                              kernel_size=kernel_size, dilation=dilation, groups=groups, stride=1),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv1d(in_planes * size_hidden, in_planes,
                              kernel_size=kernel_size, groups=groups, stride=1),
                    nn.Tanh()
                ]

            self.phi = nn.Sequential(*modules_phi)
            self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):

        (x_even, x_odd) = self.split(x)
        x_ds = (x_even + x_odd) / 2

        if not self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d), x_ds
        else:
            d = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            c = x_even.mul(torch.exp(self.psi(d))) + self.U(d)
            return (c, d), x_ds


class LiftingSchemeLevel(nn.Module):
    def __init__(self, in_planes, groups=1, simple_lifting=False):
        super(LiftingSchemeLevel, self).__init__()
        self.level = LiftingScheme(
            in_planes=in_planes,
            groups=groups,
            simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (L, H)'''
        (L, H), x_ds = self.level(x)  # 10 3 224 224

        return (L, H), x_ds


class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, no_bottleneck, groups,
                 simple_lifting, reg_details, reg_approx):
        super(LevelTWaveNet, self).__init__()
        self.reg_details = reg_details
        self.reg_approx = reg_approx
        self.groups = groups

        self.wavelet = LiftingSchemeLevel(in_planes, groups,
                                          simple_lifting=simple_lifting)
        if no_bottleneck:
            self.bottleneck = None
        else:
            self.bottleneck = BottleneckBlock(in_planes, in_planes, disable_conv=False)

    def forward(self, x):
        (L, H), x_ds = self.wavelet(x)  # 10 9 128
        approx = L
        details = H

        rd = self.reg_details * H.abs().mean()

        # rc = self.reg_approx * torch.dist(approx.mean(), x.mean(), p=2)
        rc = self.reg_approx * torch.pow((approx - x_ds), 2).sum(dim=-1).sqrt().mean()
        # rc = self.reg_approx * torch.dist(L.mean(dim=-1), x.mean(dim=-1), p=2)
        # rc = self.reg_approx * (L.mean(dim=-1) - x.mean(dim=-1)).abs().mean()

        r = rd + rc

        if self.bottleneck:
            return self.bottleneck(approx), r, details
        else:
            return approx, r, details


class WaveNet(nn.Module):

    def __init__(self, number_levels, class_num=6,
                 haar_wavelet=False, in_planes=22, reconstruction=None,
                 no_bottleneck=True, gate_pool=0,
                 simple_lifting=False, w_act='Sigmoid', reg_details=0.1, reg_approx=0.1,
                 w_disable_conv=False, w_position='encoder', d_model=64, num_trans_layers=1, block_type='CNN'):
        super(WaveNet, self).__init__()
        self.number_levels = number_levels
        self.reconstruction = reconstruction
        self.gate_pool = gate_pool
        self.w_position = w_position
        self.block_type = block_type
        self.d_model = d_model
        self.num_trans_layers = num_trans_layers

        self.gates = nn.ModuleList()

        for i in range(8):
            self.gates.add_module(
                'gate_' + str(i),
                GateGenerator(in_planes, kernel_size=3, activate=w_act, disable_conv=w_disable_conv,
                              gate_pool=gate_pool)
            )

        self.levels = nn.ModuleList()

        for i in range(number_levels):
            groups = 2 ** i
            if haar_wavelet:
                self.levels.add_module(
                    'level_' + str(i),
                    Haar(groups, reg_details, reg_approx)
                )
            else:
                self.levels.add_module(
                    'level_' + str(i),
                    LevelTWaveNet(in_planes,
                                  no_bottleneck,
                                  groups,
                                  simple_lifting, reg_details, reg_approx)
                )
            in_planes *= 2

        self.class_num = class_num

        self.decoder = nn.ModuleList()
        for i in range(self.number_levels - 1):
            if self.block_type == 'CNN':
                self.decoder.append(DecoderBlock(in_planes=in_planes, out_planes=in_planes // 2)
                                    )
            if self.block_type == 'Trans':
                self.decoder.append(DecoderBlock(in_planes=in_planes, out_planes=in_planes // 2,
                                                 d_model=self.d_model,
                                                 num_trans_layers=self.num_trans_layers,
                                                 stride=int(2 ** i),
                                                 block_type=self.block_type))
            in_planes //= 2
        in_planes //= 2

        if self.reconstruction is not None:
            self.reconstructor = nn.ModuleList()
            if reconstruction == 'add':
                for i in range(self.number_levels - 1):
                    self.reconstructor.append(
                        nn.Sequential(
                            nn.ConvTranspose1d(in_planes * 2 ** (self.number_levels - i),
                                               in_planes * 2 ** (self.number_levels - 1 - i),
                                               kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm1d(in_planes * 2 ** (self.number_levels - 1 - i)),
                            nn.Tanh()
                        )
                    )

        self.final_layer = nn.ConvTranspose1d(in_planes * 2, in_planes * self.class_num, kernel_size=3, stride=2,
                                              padding=1)
        # self.final_bn = nn.BatchNorm1d(in_planes*self.class_num)
        self.final_activate = nn.Sigmoid()

        #
        # self.final_layer = conv5x1(in_planes, self.class_num)
        # self.final_bn = nn.BatchNorm1d(self.class_num)
        # self.final_activate = nn.Sigmoid()

    def forward(self, x):

        spectrum = self.calculate_spectrum(x)

        gate_weight = []
        w, next = None, None
        for g in self.gates:
            if w is None:
                if self.gate_pool:
                    next, w = g(spectrum)
                else:
                    w = g(spectrum)
            else:
                if self.gate_pool:
                    next, w = g(next)
                else:
                    w = g(w)
            gate_weight.append(w)

        rs = []  # List of constrains on details and mean
        app_and_det = []  # List of approx and details

        B, C, L = x.shape

        for l, w in zip(self.levels, gate_weight[::-1]):

            approx, r, details = l(x)
            rs.append(r)

            group_num = l.groups
            l_sub_band = approx.shape[-1]
            decomposed = torch.concatenate([approx, details], dim=1)
            index = torch.arange(decomposed.shape[1], device=decomposed.device).view(-1, C)
            index = index[[torch.arange(group_num * 2).reshape(-1, group_num).T.reshape(-1)]].reshape(-1, 1)
            index = index.repeat([B, 1, l_sub_band])
            x = decomposed.gather(dim=1, index=index)

            if self.w_position == 'encoder':
                x = self.pass_tree(x, w)
            app_and_det.append(x)

        x = app_and_det[-1]
        if self.w_position == 'decoder':
            x = self.pass_tree(x, gate_weight[-self.number_levels])

        if self.reconstruction is not None:
            x_dec = [x]
            x_length = [x.size(-1)]

        for d, w, x_skip in zip(self.decoder, gate_weight[-self.number_levels + 1:],
                                app_and_det[self.number_levels - 2::-1]):
            if self.w_position == 'decoder':
                x_skip = self.pass_tree(x_skip, w)
            x = d(x, x_skip)

            if self.reconstruction is not None:
                x_dec.append(x)
                x_length.append(x.size(-1))

        if self.reconstruction is not None:
            x = self.reconstructor[0](x_dec[0])
            for i in range(1, len(self.reconstructor)):
                x = F.adaptive_avg_pool1d(x, x_length[i])
                x += x_dec[i]
                x = self.reconstructor[i](x_dec[i])

        x = self.final_activate((self.final_layer(x)))
        x = x.view(B, C, self.class_num, L)
        x = x.permute(0, 1, 3, 2)

        output = {
            'y_pred': x,
            'loss_reg': sum(rs),
            'tree': gate_weight,
            'components': app_and_det,
        }

        return output

    def pass_tree(self, x, w):
        w = self.reshape_weight(w)
        x = x * w
        return x

    def reshape_weight(self, w):
        if w.size(1) == 1:
            w = torch.permute(w, [0, 2, 1])
            w = w.repeat_interleave(22, dim=1)
        else:
            w = torch.permute(w, [0, 2, 1])
            w = w.reshape(-1, w.size(1) * w.size(2), 1)
        return w

    def calculate_spectrum(self, x):
        data_length = x.size(2)
        expanded_length = 1024 - data_length
        x = F.pad(x, (expanded_length // 2, expanded_length - expanded_length // 2), mode='circular')
        fft_result = torch.fft.fft(x).abs()
        spectrum = fft_result[:, :, :1024 // 2]
        return spectrum


class GateGenerator(nn.Module):

    def __init__(self, in_planes, kernel_size, activate, disable_conv=True, gate_pool=0):
        super(GateGenerator, self).__init__()
        self.disable_conv = disable_conv
        self.gate_pool = gate_pool
        if self.disable_conv:
            self.pad = nn.ConstantPad1d(1, 0)

        else:
            self.pad = nn.ConstantPad1d(3, 0)
            self.conv1 = nn.Conv1d(in_planes, in_planes * 2, kernel_size)
            self.relu1 = nn.LeakyReLU(0.01, inplace=True)
            self.conv2 = nn.Conv1d(in_planes * 2, in_planes, kernel_size)

        self.pool = nn.AvgPool1d(kernel_size=3, stride=2)
        if activate == 'RELU':
            self.activate = nn.ReLU()
        if activate == 'Sigmoid':
            self.activate = nn.Sigmoid()
        if activate == 'ELU':
            self.activate = nn.ELU()
        if activate == 'LeakyReLU':
            self.activate = nn.LeakyReLU()

        if self.gate_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x = F.layer_norm(x, (x.size(1), x.size(2)))
        x = (x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + 1e-6)
        if self.disable_conv:
            x = self.pad(x)
        else:
            x = self.conv2(self.relu1(self.conv1(self.pad(x))))
        next = self.pool(x)

        if self.gate_pool:
            out = self.activate(self.avgpool(next))
            return next, out
        else:
            next = self.activate(next)
            return next


if __name__ == '__main__':
    net = WaveNet(class_num=1, number_levels=4, reconstruction=None,
                  w_act='Sigmoid', gate_pool=0, reg_details=0.01, reg_approx=0.1,
                  w_position='decoder', block_type='CNN', haar_wavelet=False, d_model=32)
    batch_size = 5
    t = torch.arange(625)
    r = torch.tensor(125)
    x = torch.sin(2 * torch.pi * t / r) + torch.sin(60 * torch.pi * t / r) + torch.sin(120 * torch.pi * t / r)
    x = x.expand(5, 22, 625)

    out = net(x)
    pass
