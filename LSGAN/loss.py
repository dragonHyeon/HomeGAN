import torch.nn as nn

loss_fn = nn.BCELoss()


"""
nn.BCELoss()
def forward(self, input: Tensor, target: Tensor) -> Tensor:

사용 설명
batch size = 2 인 경우
input(예측한 값): model 의 output 값. ex) [0.4, 0.9]
target(실제 값): target 값. ex) [0, 1]
([[0.4], [0.9]], [[0], [1]] 이렇게 들어와도 상관 없음. 모양만 서로 같으면 됨)

loss_fn = nn.BCELoss()
loss_fn(input, target)
"""
