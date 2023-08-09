import torch.nn as nn

from SinGAN.layer import BasicConv2d


class Generator(nn.Module):
    def __init__(self, channels, num_hidden):
        """
        * 모델 구조 정의
        :param channels: channels 수
        :param num_hidden: num_hidden 수
        """

        super(Generator, self).__init__()

        # (N (1), channels (3), H, W) -> (N (1), num_hidden, H, W)
        self.head = BasicConv2d(in_channels=channels, out_channels=num_hidden)

        # (N (1), num_hidden, H, W) -> (N (1), num_hidden, H, W)
        self.body = nn.Sequential(
            BasicConv2d(in_channels=num_hidden, out_channels=num_hidden),
            BasicConv2d(in_channels=num_hidden, out_channels=num_hidden),
            BasicConv2d(in_channels=num_hidden, out_channels=num_hidden)
        )

        # (N (1), num_hidden, H, W) -> (N (1), channels (3), H, W)
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, prev_x):
        """
        * 순전파
        :param z: 입력 노이즈. (N (1), channels (3), H, W)
        :param prev_x: 이전 단계에서 생성된 fake_x 를 scale_factor 만큼 upscale 한 입력. (N (1), channels (3), H, W)
        :return: 생성된 이미지. (N (1), channels (3), H, W)
        """

        x = self.head(z + prev_x)
        x = self.body(x)
        x = self.tail(x)
        x = x + prev_x

        # (N (1), channels (3), H, W) -> (N (1), channels (3), H, W)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, num_hidden):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        :param num_hidden: num_hidden 수
        """

        super(Discriminator, self).__init__()

        # (N (1), in_channels (3), H, W) -> (N (1), num_hidden, H, W)
        self.head = BasicConv2d(in_channels=in_channels, out_channels=num_hidden)

        # (N (1), num_hidden, H, W) -> (N (1), num_hidden, H, W)
        self.body = nn.Sequential(
            BasicConv2d(in_channels=num_hidden, out_channels=num_hidden),
            BasicConv2d(in_channels=num_hidden, out_channels=num_hidden),
            BasicConv2d(in_channels=num_hidden, out_channels=num_hidden)
        )

        # (N (1), num_hidden, H, W) -> (N (1), 1, H, W)
        self.tail = nn.Conv2d(in_channels=num_hidden, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        * 순전파
        :param x: 입력. (N (1), in_channels (3), H, W)
        :return: H x W patch 의 참 거짓 판별 결과. (N (1), 1, H, W)
        """

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        # (N (1), in_channels (3), H, W) -> (N (1), 1, H, W)
        return x
