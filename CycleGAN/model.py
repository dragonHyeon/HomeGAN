import torch.nn as nn

from CycleGAN.layer import ResidualBlock, DisBlock


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_residual_blocks=9):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param num_residual_blocks: Residual blocks 몇 개 쌓을지
        """

        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        # (N, in_channels (3), H (256), W (256)) -> (N, out_features (64), 256, 256)
        out_features = 64
        layer = [
            # (N, in_channels (3), H (256), W (256)) -> (N, in_channels (3), H+6, W+6)
            nn.ReflectionPad2d(padding=3),
            # (N, 3, 262, 262) -> (N, out_features (64), 256, 256)
            nn.Conv2d(in_channels=in_channels, out_channels=out_features, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=out_features),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        # (N, 64, 256, 256) -> (N, 256, 64, 64)
        for _ in range(2):
            in_features = out_features
            out_features *= 2
            layer += [
                # (N, in_channels, H, W) -> (N, out_features (in_channels*2), H/2, W/2)
                nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(num_features=out_features),
                nn.ReLU(inplace=True)
            ]

        # Residual blocks
        # (N, 256, 64, 64) -> (N, 256, 64, 64)
        for _ in range(num_residual_blocks):
            # (N, in_channels, H, W) -> (N, in_channels, H, W)
            layer += [ResidualBlock(channels=out_features)]

        # Upsampling
        # (N, 256, 64, 64) -> (N, 64, 256, 256)
        for _ in range(2):
            in_features = out_features
            out_features //= 2
            layer += [
                # (N, in_channels, H, W) -> (N, out_features (in_channels//2), H*2, W*2)
                nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(num_features=out_features),
                nn.ReLU(inplace=True)
            ]

        # Output layer
        # (N, 64, 256, 256) -> (N, out_channels (3), 256, 256)
        layer += [
            # (N, 64, 256, 256) -> (N, 64, 262, 262)
            nn.ReflectionPad2d(padding=3),
            # (N, 64, 262, 262) -> (N, out_channels (3), 256, 256)
            nn.Conv2d(in_channels=out_features, out_channels=out_channels, kernel_size=7, stride=1, padding=0, bias=True),
            nn.Tanh()
        ]

        # (N, in_channels (3), H (256), W (256)) -> (N, out_channels (3), H (256), W (256))
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (3), H (256), W (256))
        :return: 배치 개수 만큼의 변환된 이미지. (N, out_channels (3), H (256), W (256))
        """

        # (N, in_channels (3), H (256), W (256)) -> (N, out_channels (3), H (256), W (256))
        return self.layer(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        """

        super(Discriminator, self).__init__()

        # 16 x 16 PatchGAN
        self.main = nn.Sequential(
            # (N, in_channels (3), H (256), W (256)) -> (N, 64, H/2, W/2)
            DisBlock(in_channels=in_channels, out_channels=64, normalize=False),
            # (N, 64, H/2, W/2) -> (N, 128, H/4, W/4)
            DisBlock(in_channels=64, out_channels=128),
            # (N, 128, H/4, W/4) -> (N, 256, H/8, W/8)
            DisBlock(in_channels=128, out_channels=256),
            # (N, 256, H/8, W/8) -> (N, 512, H/16, W/16)
            DisBlock(in_channels=256, out_channels=512),
            # (N, 512, H/16, W/16) -> (N, 1, H/16 (16), W/16 (16))
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (3), H (256), W (256))
        :return: 배치 개수 만큼의 16 x 16 patch 의 참 거짓 판별 결과. (N, 1, H/16 (16), W/16 (16))
        """

        # (N, in_channels (3), H (256), W (256)) -> (N, 1, H/16 (16), W/16 (16))
        x = self.main(x)

        return x
