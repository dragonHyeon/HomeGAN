import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        * BasicConv2d 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        """

        super(BasicConv2d, self).__init__()

        # (Conv + BatchNorm + ReLU) 블록
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001),
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, out_channels, H, W)
        """

        # (N, in_channels, H, W) -> (N, out_channels, H, W)
        out = self.block(x)

        return out
