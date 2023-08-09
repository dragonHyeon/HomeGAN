import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, out_channels=1):
        """
        * 모델 구조 정의
        :param out_channels: out_channels 수
        """

        super(Generator, self).__init__()

        # 생성자
        self.main = nn.Sequential(
            # (N, 100, 1, 1) -> (N, 1024, 4, 4)
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            # (N, 1024, 4, 4) -> (N, 512, 8, 8)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            # (N, 512, 8, 8) -> (N, 256, 16, 16)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # (N, 256, 16, 16) -> (N, 128, 32, 32)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # (N, 128, 32, 32) -> (N, out_channels (1), 64, 64)
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        """
        * 순전파
        :param z: 배치 개수 만큼의 입력 노이즈. (N, 100, 1, 1)
        :return: 배치 개수 만큼의 생성된 이미지. (N, out_channels (1), 64, 64)
        """

        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        """

        super(Discriminator, self).__init__()

        # 판별자
        self.main = nn.Sequential(
            # (N, in_channels (1), 64, 64) -> (N, 128, 32, 32)
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 128, 32, 32) -> (N, 256, 16, 16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 256, 16, 16) -> (N, 512, 8, 8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 512, 8, 8) -> (N, 1024, 4, 4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 1024, 4, 4) -> (N, 1, 1, 1)
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (1), 64, 64)
        :return: 배치 개수 만큼의 참 거짓 판별 결과. (N)
        """

        # (N, in_channels (1), 64, 64) -> (N, 1, 1, 1)
        x = self.main(x)
        # (N, 1, 1, 1) -> (N)
        x = x.view(-1)

        return x
