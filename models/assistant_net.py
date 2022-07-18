from __future__ import absolute_import
import math
from RFC import calculate_channel_l2_norm
import torch.nn as nn
import torch



"""
preactivation resnet with bottleneck design.
"""

kwargs = {'num_workers': 1, 'pin_memory': True}

class ResNet(nn.Module):
    def __init__(self, cfg=None ,num_classes=1000,**kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()

        if cfg is None:
            # Construct config variable.
            cfg = [[64], [64, 64, 256] * 3, [128, 128, 512] * 4, [256, 256, 1024] * 6, [512, 512, 2048] * 3]
            cfg = [item for sub_list in cfg for item in sub_list]


        self.conv0 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(cfg[0])
        self.relu0 = nn.ReLU(inplace=True)
        self.P0 = []
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.C2 = nn.Identity()
        self.C3 = nn.Identity()


        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1,stride=1, bias=False)
        self.P1 = []
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3,stride=1,padding=1, bias=False)
        self.P2 = []
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], cfg[3], kernel_size=1,stride=1, bias=False)
        self.P3 = []
        self.bn3 = nn.BatchNorm2d(cfg[3])
        self.relu1 = nn.ReLU(inplace=True)
        self.C11 = nn.Identity()
        self.downsample1 = nn.Conv2d(cfg[0], cfg[3], kernel_size=1,stride=1,bias=False)
        self.downsample1_bn = nn.BatchNorm2d(cfg[3])
        self.P1_1 = []

        self.C13 = nn.Identity()
        self.conv4 = nn.Conv2d(cfg[3], cfg[4], kernel_size=1,stride=1,bias=False)
        self.P4 = []
        self.bn4 = nn.BatchNorm2d(cfg[4])
        self.conv5 = nn.Conv2d(cfg[4], cfg[5], kernel_size=3,stride=1, padding=1,bias=False)
        self.P5 = []
        self.bn5 = nn.BatchNorm2d(cfg[5])
        self.conv6 = nn.Conv2d(cfg[5], cfg[3], kernel_size=1,stride=1,bias=False)
        self.bn6 = nn.BatchNorm2d(cfg[3])
        self.P6 = []
        self.relu2 = nn.ReLU(inplace=True)
        self.C21 = nn.Identity()

        self.conv7 = nn.Conv2d(cfg[3], cfg[7], kernel_size=1,stride=1,bias=False)
        self.P7 = []
        self.bn7 = nn.BatchNorm2d(cfg[7])
        self.conv8 = nn.Conv2d(cfg[7], cfg[8], kernel_size=3,stride=1, padding=1,bias=False)
        self.P8 = []
        self.bn8 = nn.BatchNorm2d(cfg[8])
        self.conv9 = nn.Conv2d(cfg[8], cfg[3], kernel_size=1,stride=1,bias=False)
        self.bn9 = nn.BatchNorm2d(cfg[3])
        self.P9 = []
        self.relu3 = nn.ReLU(inplace=True)
        self.C29 = nn.Identity()
        self.C53 = nn.Identity()


        self.conv10 = nn.Conv2d(cfg[3], cfg[10], kernel_size=1,stride=1,bias=False)
        self.P10 = []
        self.bn10 = nn.BatchNorm2d(cfg[10])
        self.conv11 = nn.Conv2d(cfg[10], cfg[11], kernel_size=3,stride=2, padding=1,bias=False)
        self.P11 = []
        self.bn11 = nn.BatchNorm2d(cfg[11])
        self.conv12 = nn.Conv2d(cfg[11], cfg[12], kernel_size=1,stride=1,bias=False)
        self.bn12 = nn.BatchNorm2d(cfg[12])
        self.P12 = []
        self.relu4 = nn.ReLU(inplace=True)
        self.C37 = nn.Identity()
        self.downsample2 = nn.Conv2d(cfg[3], cfg[12], kernel_size=1,stride=(2,2),bias=False)
        self.downsample2_bn = nn.BatchNorm2d(cfg[12])
        self.P2_2 = []
        self.C38 = nn.Identity()


        self.conv13 = nn.Conv2d(cfg[12], cfg[13], kernel_size=1,stride=1,bias=False)
        self.P13 = []
        self.bn13 = nn.BatchNorm2d(cfg[13])
        self.conv14 = nn.Conv2d(cfg[13], cfg[14], kernel_size=3,stride=1, padding=1,bias=False)
        self.P14 = []
        self.bn14 = nn.BatchNorm2d(cfg[14])
        self.conv15 = nn.Conv2d(cfg[14], cfg[12], kernel_size=1,stride=1,bias=False)
        self.bn15 = nn.BatchNorm2d(cfg[12])
        self.P15 = []
        self.relu5 = nn.ReLU(inplace=True)
        self.C45 = nn.Identity()

        self.conv16 = nn.Conv2d(cfg[12], cfg[16], kernel_size=1,stride=1,bias=False)
        self.P16 = []
        self.bn16 = nn.BatchNorm2d(cfg[16])
        self.conv17 = nn.Conv2d(cfg[16], cfg[17], kernel_size=3,stride=1, padding=1,bias=False)
        self.P17 = []
        self.bn17 = nn.BatchNorm2d(cfg[17])
        self.conv18 = nn.Conv2d(cfg[17], cfg[12], kernel_size=1,stride=1,bias=False)
        self.bn18 = nn.BatchNorm2d(cfg[12])
        self.P18 = []
        self.relu6 = nn.ReLU(inplace=True)
        self.C46 = nn.Identity()

        self.conv19 = nn.Conv2d(cfg[12], cfg[19], kernel_size=1, stride=1, bias=False)
        self.P19 = []
        self.bn19 = nn.BatchNorm2d(cfg[19])
        self.conv20 = nn.Conv2d(cfg[19], cfg[20], kernel_size=3,stride=1, padding=1, bias=False)
        self.P20 = []
        self.bn20 = nn.BatchNorm2d(cfg[20])
        self.conv21 = nn.Conv2d(cfg[20], cfg[12], kernel_size=1, stride=1, bias=False)
        self.bn21 = nn.BatchNorm2d(cfg[12])
        self.P21 = []
        self.relu7 = nn.ReLU(inplace=True)
        self.C62 = nn.Identity()
        self.C64 = nn.Identity()


        self.conv22 = nn.Conv2d(cfg[12], cfg[22], kernel_size=1, stride=1, bias=False)
        self.P22 = []
        self.bn22 = nn.BatchNorm2d(cfg[22])
        self.conv23 = nn.Conv2d(cfg[22], cfg[23], kernel_size=3,stride=(2,2), padding=1, bias=False)
        self.P23 = []
        self.bn23 = nn.BatchNorm2d(cfg[23])
        self.conv24 = nn.Conv2d(cfg[23], cfg[24], kernel_size=1, stride=1, bias=False)
        self.bn24 = nn.BatchNorm2d(cfg[24])
        self.P24 = []
        self.relu8 = nn.ReLU(inplace=True)
        self.C71 = nn.Identity()
        self.downsample3 = nn.Conv2d(cfg[12], cfg[24], kernel_size=1,stride=(2,2),bias=False)
        self.downsample3_bn = nn.BatchNorm2d(cfg[24])
        self.P3_3 = []
        self.C72 = nn.Identity()

        self.conv25 = nn.Conv2d(cfg[24], cfg[25], kernel_size=1, stride=1, bias=False)
        self.P25 = []
        self.bn25 = nn.BatchNorm2d(cfg[25])
        self.conv26 = nn.Conv2d(cfg[25], cfg[26], kernel_size=3,stride=1, padding=1, bias=False)
        self.P26 = []
        self.bn26 = nn.BatchNorm2d(cfg[26])
        self.conv27 = nn.Conv2d(cfg[26], cfg[24], kernel_size=1, stride=1, bias=False)
        self.bn27 = nn.BatchNorm2d(cfg[24])
        self.P27 = []
        self.relu9 = nn.ReLU(inplace=True)
        self.C80 = nn.Identity()

        self.conv28 = nn.Conv2d(cfg[24], cfg[28], kernel_size=1, stride=1, bias=False)
        self.P28 = []
        self.bn28 = nn.BatchNorm2d(cfg[28])
        self.conv29 = nn.Conv2d(cfg[28], cfg[29], kernel_size=3,stride=1, padding=1, bias=False)
        self.P29 = []
        self.bn29 = nn.BatchNorm2d(cfg[29])
        self.conv30 = nn.Conv2d(cfg[29], cfg[24], kernel_size=1, stride=1, bias=False)
        self.bn30 = nn.BatchNorm2d(cfg[24])
        self.P30 = []
        self.relu10 = nn.ReLU(inplace=True)
        self.C88 = nn.Identity()

        self.conv31 = nn.Conv2d(cfg[24], cfg[31], kernel_size=1, stride=1, bias=False)
        self.P31 = []
        self.bn31 = nn.BatchNorm2d(cfg[31])
        self.conv32 = nn.Conv2d(cfg[31], cfg[32], kernel_size=3,stride=1, padding=1, bias=False)
        self.P32 = []
        self.bn32 = nn.BatchNorm2d(cfg[32])
        self.conv33 = nn.Conv2d(cfg[32], cfg[24], kernel_size=1, stride=1, bias=False)
        self.bn33 = nn.BatchNorm2d(cfg[24])
        self.P33 = []
        self.relu11 = nn.ReLU(inplace=True)
        self.C96 = nn.Identity()

        self.conv34 = nn.Conv2d(cfg[24], cfg[34], kernel_size=1, stride=1, bias=False)
        self.P34 = []
        self.bn34 = nn.BatchNorm2d(cfg[34])
        self.conv35 = nn.Conv2d(cfg[34], cfg[35], kernel_size=3,stride=1, padding=1, bias=False)
        self.P35 = []
        self.bn35 = nn.BatchNorm2d(cfg[35])
        self.conv36 = nn.Conv2d(cfg[35], cfg[24], kernel_size=1, stride=1, bias=False)
        self.bn36 = nn.BatchNorm2d(cfg[24])
        self.P36 = []
        self.relu12 = nn.ReLU(inplace=True)
        self.C104 = nn.Identity()

        self.conv37 = nn.Conv2d(cfg[24], cfg[37], kernel_size=1, stride=1, bias=False)
        self.P37 = []
        self.bn37 = nn.BatchNorm2d(cfg[37])
        self.conv38 = nn.Conv2d(cfg[37], cfg[38], kernel_size=3,stride=1, padding=1, bias=False)
        self.P38 = []
        self.bn38 = nn.BatchNorm2d(cfg[38])
        self.conv39 = nn.Conv2d(cfg[38], cfg[24], kernel_size=1, stride=1, bias=False)
        self.bn39 = nn.BatchNorm2d(cfg[24])
        self.P39 = []
        self.relu13 = nn.ReLU(inplace=True)
        self.C105 = nn.Identity()
        self.C106 = nn.Identity()


        self.conv40 = nn.Conv2d(cfg[24], cfg[40], kernel_size=1, stride=1, bias=False)
        self.P40 = []
        self.bn40 = nn.BatchNorm2d(cfg[40])
        self.conv41 = nn.Conv2d(cfg[40], cfg[41], kernel_size=3,stride=(2,2), padding=1, bias=False)
        self.P41 = []
        self.bn41 = nn.BatchNorm2d(cfg[41])
        self.conv42 = nn.Conv2d(cfg[41], cfg[42], kernel_size=1, stride=1, bias=False)
        self.bn42 = nn.BatchNorm2d(cfg[42])
        self.P42 = []
        self.relu14 = nn.ReLU(inplace=True)
        self.C122 = nn.Identity()
        self.downsample4 = nn.Conv2d(cfg[24], cfg[42], kernel_size=1,stride=(2,2),bias=False)
        self.downsample4_bn = nn.BatchNorm2d(cfg[42])
        self.P4_4 = []
        self.C123 = nn.Identity()

        self.conv43 = nn.Conv2d(cfg[42], cfg[43], kernel_size=1, stride=1, bias=False)
        self.P43 = []
        self.bn43 = nn.BatchNorm2d(cfg[43])
        self.conv44 = nn.Conv2d(cfg[43], cfg[44], kernel_size=3,stride=1, padding=1, bias=False)
        self.P44 = []
        self.bn44 = nn.BatchNorm2d(cfg[44])
        self.conv45 = nn.Conv2d(cfg[44], cfg[42], kernel_size=1, stride=1, bias=False)
        self.bn45 = nn.BatchNorm2d(cfg[42])
        self.P45 = []
        self.relu15 = nn.ReLU(inplace=True)
        self.C131 = nn.Identity()

        self.conv46 = nn.Conv2d(cfg[42], cfg[46], kernel_size=1, stride=1, bias=False)
        self.P46 = []
        self.bn46 = nn.BatchNorm2d(cfg[46])
        self.conv47 = nn.Conv2d(cfg[46], cfg[47], kernel_size=3,stride=1, padding=1, bias=False)
        self.P47 = []
        self.bn47 = nn.BatchNorm2d(cfg[47])
        self.conv48 = nn.Conv2d(cfg[47], cfg[42], kernel_size=1, stride=1, bias=False)
        self.bn48 = nn.BatchNorm2d(cfg[42])
        self.P48 = []
        self.relu16 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(cfg[-1], num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _jisuan(self,p_index,input):
        for i in p_index:
            if input==i[0]:
                return p_index
        p_index.append(input[0])
        return p_index



    def forward(self, x):

        x = self.conv0(x)
        out = self.bn0(x)
        out = self.relu0(out)
        self.P0.append(calculate_channel_l2_norm(out))
        out = self.maxpool(out)
        residual = out

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        self.P1.append(calculate_channel_l2_norm(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu1(out)
        self.P2.append(calculate_channel_l2_norm(out))
        out = self.conv3(out)
        out = self.bn3(out)
        self.P3.append(calculate_channel_l2_norm(out))
        residual = self.downsample1(residual)
        residual = self.downsample1_bn(residual)
        self.P1_1.append(calculate_channel_l2_norm(residual))
        residual = residual
        out += residual
        out = self.relu1(out)


        residual = out
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu2(out)
        self.P4.append(calculate_channel_l2_norm(out))
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu2(out)
        self.P5.append(calculate_channel_l2_norm(out))
        out = self.conv6(out)
        out = self.bn6(out)
        self.P6.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu2(out)


        residual = out
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu3(out)
        self.P7.append(calculate_channel_l2_norm(out))
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu3(out)
        self.P8.append(calculate_channel_l2_norm(out))
        out = self.conv9(out)
        out = self.bn9(out)
        self.P9.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu3(out)

        residual = out
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu4(out)
        self.P10.append(calculate_channel_l2_norm(out))
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu4(out)
        self.P11.append(calculate_channel_l2_norm(out))
        out = self.conv12(out)
        out = self.bn12(out)
        self.P12.append(calculate_channel_l2_norm(out))
        residual = self.downsample2(residual)
        residual = self.downsample2_bn(residual)
        self.P2_2.append(calculate_channel_l2_norm(residual))
        residual = residual
        out += residual
        out = self.relu4(out)


        residual = out
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.relu5(out)
        self.P13.append(calculate_channel_l2_norm(out))
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu5(out)
        self.P14.append(calculate_channel_l2_norm(out))
        out = self.conv15(out)
        out = self.bn15(out)
        self.P15.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu5(out)

        residual = out
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu6(out)
        self.P16.append(calculate_channel_l2_norm(out))
        out = self.conv17(out)
        out = self.bn17(out)
        out = self.relu6(out)
        self.P17.append(calculate_channel_l2_norm(out))
        out = self.conv18(out)
        out = self.bn18(out)
        self.P18.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu6(out)

        residual = out
        out = self.conv19(out)
        out = self.bn19(out)
        out = self.relu7(out)
        self.P19.append(calculate_channel_l2_norm(out))
        out = self.conv20(out)
        out = self.bn20(out)
        out = self.relu7(out)
        self.P20.append(calculate_channel_l2_norm(out))
        out = self.conv21(out)
        out = self.bn21(out)
        self.P21.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu7(out)

        residual = out
        out = self.conv22(out)
        out = self.bn22(out)
        out = self.relu8(out)
        self.P22.append(calculate_channel_l2_norm(out))
        out = self.conv23(out)
        out = self.bn23(out)
        out = self.relu8(out)
        self.P23.append(calculate_channel_l2_norm(out))
        out = self.conv24(out)
        out = self.bn24(out)
        self.P24.append(calculate_channel_l2_norm(out))
        residual = self.downsample3(residual)
        residual = self.downsample3_bn(residual)
        self.P3_3.append(calculate_channel_l2_norm(residual))
        residual = residual
        out += residual
        out = self.relu8(out)


        residual = out
        out = self.conv25(out)
        out = self.bn25(out)
        out = self.relu9(out)
        self.P25.append(calculate_channel_l2_norm(out))
        out = self.conv26(out)
        out = self.bn26(out)
        out = self.relu9(out)
        self.P26.append(calculate_channel_l2_norm(out))
        out = self.conv27(out)
        out = self.bn27(out)
        self.P27.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu9(out)


        residual = out
        out = self.conv28(out)
        out = self.bn28(out)
        out = self.relu10(out)
        self.P28.append(calculate_channel_l2_norm(out))
        out = self.conv29(out)
        out = self.bn29(out)
        out = self.relu10(out)
        self.P29.append(calculate_channel_l2_norm(out))
        out = self.conv30(out)
        out = self.bn30(out)
        self.P30.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu10(out)


        residual = out
        out = self.conv31(out)
        out = self.bn31(out)
        out = self.relu11(out)
        self.P31.append(calculate_channel_l2_norm(out))
        out = self.conv32(out)
        out = self.bn32(out)
        out = self.relu11(out)
        self.P32.append(calculate_channel_l2_norm(out))
        out = self.conv33(out)
        out = self.bn33(out)
        self.P33.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu11(out)


        residual = out
        out = self.conv34(out)
        out = self.bn34(out)
        out = self.relu12(out)
        self.P34.append(calculate_channel_l2_norm(out))
        out = self.conv35(out)
        out = self.bn35(out)
        out = self.relu12(out)
        self.P35.append(calculate_channel_l2_norm(out))
        out = self.conv36(out)
        out = self.bn36(out)
        self.P36.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu12(out)

        residual = out
        out = self.conv37(out)
        out = self.bn37(out)
        out = self.relu13(out)
        self.P37.append(calculate_channel_l2_norm(out))
        out = self.conv38(out)
        out = self.bn38(out)
        out = self.relu13(out)
        self.P38.append(calculate_channel_l2_norm(out))
        out = self.conv39(out)
        out = self.bn39(out)
        self.P39.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu13(out)

        residual = out
        out = self.conv40(out)
        out = self.bn40(out)
        out = self.relu14(out)
        self.P40.append(calculate_channel_l2_norm(out))
        out = self.conv41(out)
        out = self.bn41(out)
        out = self.relu14(out)
        self.P41.append(calculate_channel_l2_norm(out))
        out = self.conv42(out)
        out = self.bn42(out)
        self.P42.append(calculate_channel_l2_norm(out))
        residual = self.downsample4(residual)
        residual = self.downsample4_bn(residual)
        self.P4_4.append(calculate_channel_l2_norm(residual))
        residual = residual
        out += residual
        out = self.relu14(out)

        residual = out
        out = self.conv43(out)
        out = self.bn43(out)
        out = self.relu15(out)
        self.P43.append(calculate_channel_l2_norm(out))
        out = self.conv44(out)
        out = self.bn44(out)
        out = self.relu15(out)
        self.P44.append(calculate_channel_l2_norm(out))
        out = self.conv45(out)
        out = self.bn45(out)
        self.P45.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu15(out)

        residual = out
        out = self.conv46(out)
        out = self.bn46(out)
        out = self.relu16(out)
        self.P46.append(calculate_channel_l2_norm(out))
        out = self.conv47(out)
        out = self.bn47(out)
        out = self.relu16(out)
        self.P47.append(calculate_channel_l2_norm(out))
        out = self.conv48(out)
        out = self.bn48(out)
        self.P48.append(calculate_channel_l2_norm(out))
        residual = residual
        out += residual
        out = self.relu16(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)


        return out

