import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        * ResidualBlock 모듈 구조 정의
        :param channels: channels 수
        """

        super(ResidualBlock, self).__init__()

        # (N, channels, H, W) -> (N, channels, H, W)
        self.layer = nn.Sequential(
            # (N, channels, H, W) -> (N, channels, H+2, W+2)
            nn.ReflectionPad2d(padding=1),
            # (N, channels, H+2, W+2) -> (N, channels, H, W)
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            # (N, channels, H, W) -> (N, channels, H+2, W+2)
            nn.ReflectionPad2d(padding=1),
            # (N, channels, H+2, W+2) -> (N, channels, H, W)
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=channels)
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, channels, H, W)
        """

        # (N, channels, H, W) -> (N, channels, H, W)
        return x + self.layer(x)


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        """
        * DisBlock 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param normalize: InstanceNorm2d 여부
        """

        super(DisBlock, self).__init__()

        # Conv2d
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True)]
        # InstanceNorm2d
        if normalize:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        # LeakyReLU
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, out_channels, H/2, W/2)
        """

        # (N, in_channels, H, W) -> (N, out_channels, H/2, W/2)
        x = self.block(x)

        return x
