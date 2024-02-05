import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['mbv2_SPCII']

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    
class SPCII(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(SPCII, self).__init__()
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        
        self.relu = h_swish()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv4 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)    

    def k_size(self, mip):
        b = 1
        gama = 2
        k_size = int(abs((math.log(mip, 2) + b) / gama))
        return k_size-1
    
    def forward(self, x):
        identity = x   
        n,c,h,w = x.size()


        x_h_a = self.pool_h_avg(x) #（n c none, 1）
        x_h_m = self.pool_h_max(x) #
        x_w_a = self.pool_w_avg(x).permute(0, 1, 3, 2) #(n, c, large_value, 1)
        x_w_m = self.pool_w_max(x).permute(0, 1, 3, 2)


        y_h = torch.cat([x_h_a, x_w_a], dim=2) #(n, c, desired_height, 2)     
        y_h = self.conv1(y_h)
        y_h = self.bn1(y_h)  
        y_h = self.relu(y_h)

        y_w = torch.cat([x_w_m, x_h_m], dim=2) #
        y_w = self.conv1(y_w)
        y_w = self.bn1(y_w)
        y_w = self.relu(y_w)
   
        x_h_a, x_w_a = torch.split(y_h, [h, w], dim=2)
        x_w_a = x_w_a.permute(0, 1, 3, 2)

        x_h_m, x_w_m = torch.split(y_w, [h, w], dim=2)
        x_w_m = x_w_m.permute(0, 1, 3, 2)

        y_h = x_h_a + x_h_m
        y_w = x_w_a + x_w_m
        
        #y_h
        new_length = y_h.size(2) * y_h.size(3)
        output_1d_h = y_h.view(y_h.size(0), y_h.size(1), new_length)
        in_channels = y_h.size(1)
        conv1d = nn.Conv1d(in_channels, in_channels, self.k_size(in_channels)).to(output_1d_h.device)               
        y_h = conv1d(output_1d_h).unsqueeze(-1)
        
        #y_w
        new_length = y_w.size(2) * y_w.size(3)  
        output_1d_w = y_w.view(y_w.size(0), y_w.size(1), new_length)
        in_channels = y_w.size(1)
        conv1d = nn.Conv1d(in_channels, in_channels, self.k_size(in_channels)).to(output_1d_w.device)
        y_w = conv1d(output_1d_w).unsqueeze(-2)

        y_h = self.conv4(y_h).sigmoid()
        y_w = self.conv5(y_w).sigmoid()

        y_h = y_h.expand(-1, -1, h, w)
        y_w = y_w.expand(-1, -1, h, w)
        
        y =  identity * y_h* y_w
        return y

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # coordinate attention
                SPCII(hidden_dim, hidden_dim),
                
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y

class MBV2_SPCII(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.):
        super(MBV2_SPCII, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(output_channel, num_classes)
                )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                #print(m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# model = mbv2_ca(num_classes=1000, width_mult=1.0)  # You can change parameters as needed
# input_data = torch.randn(1, 3, 224, 224)  # Replace with actual input data
# with torch.no_grad():
#     output = model(input_data)
# print("Output shape:", output.shape)
# print("Output values:", output)
