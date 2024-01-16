import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Convolution') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, factor=2, size=(4, 4), size_transpose=(3, 3), stride=(1, 1), pad=(1, 1),
              dropout=0.):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=factor, mode='bilinear'))
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size_transpose, stride=stride, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    return block


## ------------------------ CnnEncoder * 10 + RNN + CnnDecoder---------------------- ##
class CRNN(nn.Module):
    def __init__(self, channelExponent=3, dropout=0., RNN_layers=1):
        super(CRNN, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        ##### U-Net Encoder #####
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(12, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer3 = blockUNet(channels * 2, channels * 2, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer4 = blockUNet(channels * 2, channels * 4, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(5, 5), pad=(1, 0))

        self.layer5 = blockUNet(channels * 4, channels * 8, 'layer5', transposed=False, bn=False, relu=False,
                                dropout=dropout, size=(2, 4), pad=0)

        ##### RNN layer #####
        self.CNN_embed_dim = channels * 8
        self.RNN_layers = RNN_layers
        self.RNN_output_size = channels * 8
        # self.rnn = nn.RNN(self.CNN_embed_dim, self.RNN_output_size, self.RNN_layers, batch_first=True)
        self.lstm = nn.LSTM(self.CNN_embed_dim, self.RNN_output_size, self.RNN_layers, batch_first=True)

        ##### U-Net Decoder #####
        self.dlayer5 = blockUNet(channels * 8, channels * 4, 'dlayer5', transposed=True, bn=True, relu=True,
                                 dropout=dropout, pad=(1, 2))

        self.dlayer4 = blockUNet(channels * 8, channels * 2, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout)

        self.dlayer3 = blockUNet(channels * 4, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout)

        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels * 2, 1, 4, 2, 1, bias=True))

    def forward(self, x_3d):
        # x_3d: torch.Size([20, 10, 12, 32, 64])
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            out1 = self.layer1(x_3d[:, t, :, :, :])  # torch.Size([20, 8, 16, 32])
            out2 = self.layer2(out1)  # torch.Size([20, 16, 8, 16])
            out3 = self.layer3(out2)  # torch.Size([20, 16, 4, 8])
            out4 = self.layer4(out3)  # torch.Size([20, 32, 2, 4])
            out5 = self.layer5(out4)  # torch.Size([20, 64, 1, 1])  # note that here is 1*1
            cnn_embed_seq.append(out5)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  # torch.Size([20, 10, 64, 1, 1])
        cnn_embed_seq = cnn_embed_seq.view(x_3d.size(0), x_3d.size(1), -1)
        # RNN
        # rnn_output, _ = self.rnn(cnn_embed_seq)
        # last_output = rnn_output[:, -1, :].clone()  # torch.Size([20, 64])
        # LSTM
        lstm_output, _ = self.lstm(cnn_embed_seq)
        last_output = lstm_output[:, -1, :].clone()  # torch.Size([20, 64])

        last_output_expand = last_output.unsqueeze(-1).unsqueeze(-1)  # torch.Size([20, 64, 1, 1])

        dout5 = self.dlayer5(last_output_expand)  # torch.Size([20, 32, 2, 4])
        dout5_out4 = torch.cat([dout5, out4], 1)  # torch.Size([20, 64, 2, 4])

        dout4 = self.dlayer4(dout5_out4)  # torch.Size([20, 16, 4, 8])
        dout4_out3 = torch.cat([dout4, out3], 1)  # torch.Size([20, 32, 4, 8])

        dout3 = self.dlayer3(dout4_out3)  # torch.Size([20, 16, 8, 16])
        dout3_out2 = torch.cat([dout3, out2], 1)  # torch.Size([20, 32, 8, 16])

        dout2 = self.dlayer2(dout3_out2)  # torch.Size([20, 8, 16, 32])
        dout2_out1 = torch.cat([dout2, out1], 1)  # torch.Size([20, 16, 16, 32])

        dout1 = self.dlayer1(dout2_out1)  # torch.Size([20, 1, 32, 64])

        return dout1


if __name__ == "__main__":
    model = CRNN()
    input_tensor = torch.randn(20, 10, 12, 32, 64)
    output_tensor = model(input_tensor)
    print(output_tensor.size())  # torch.Size([20, 1, 32, 64])
