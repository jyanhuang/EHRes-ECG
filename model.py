import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=17,
        stride=stride,
        padding=8,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv_subsampling(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlockHeartNet(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlockHeartNet only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in BasicBlockHeartNet")

        # pre-activation residual branch
        self.conv1 = conv_block(inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv_block(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

            # 用 ceil_mode=True 保证奇数长度时与主分支尺寸一致
            if identity.size(-1) > out.size(-1):
                identity = F.max_pool1d(
                    identity,
                    kernel_size=self.stride,
                    stride=self.stride,
                    ceil_mode=True
                )

        if identity.size(-1) != out.size(-1):
            raise ValueError(
                f"Identity and output sizes do not match: "
                f"{identity.size()} vs {out.size()}"
            )

        out += identity
        return out

class EHRes(nn.Module):
    """
    Your method model:
    Conv1D -> BN + ReLU -> HRB -> Dropout -> GAP -> FC

    HRB:
    layer0, layer1, layer2, layer2_, layer3, layer3_, layer4, layer4_, layer5
    """

    def __init__(
        self,
        layers=(1, 2, 2, 2, 2, 2, 2, 2, 1),
        num_classes=5,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        block=BasicBlockHeartNet,
        dropout_p=0.6,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self._norm_layer = norm_layer
        self.inplanes = 32
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must have length 3")

        self.groups = groups
        self.base_width = width_per_group

        # Front-end
        self.conv1 = conv_block(1, self.inplanes, stride=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # HRB
        self.layer0 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer2_ = self._make_layer(block, 128, layers[3], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[4], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer3_ = self._make_layer(block, 256, layers[5], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[6], stride=2, dilate=replace_stride_with_dilation[2])
        self.layer4_ = self._make_layer(block, 512, layers[7], stride=2, dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_layer(block, 1024, layers[8], stride=2, dilate=replace_stride_with_dilation[2])

        self.dropout = nn.Dropout(p=dropout_p)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockHeartNet):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_subsampling(self.inplanes, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )

        self.inplanes = planes * block.expansion

        # keep your original REB counting logic:
        # blocks=1 -> 3 REBs
        # blocks=2 -> 4 REBs
        for _ in range(1, blocks + 2):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_(x)
        x = self.layer3(x)
        x = self.layer3_(x)
        x = self.layer4(x)
        x = self.layer4_(x)
        x = self.layer5(x)

        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x