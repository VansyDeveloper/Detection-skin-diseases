import torch
from torch import nn

def get_activation(activation: str) -> nn.Module:
    """
    Возвращает функцию активации по запросе по имени.
    
    Параметры:
        activation (str, "silu" or "relu"): Имя функции активации.
    """
    if activation.lower() == "silu":
        return nn.SiLU(inplace=True)
    return nn.ReLU(inplace=True)

def get_normalization(normalization: str, num_channels: int, num_groups: int = None, eps: float = 1e-5, affine=True, momentum: float = 0.1) -> nn.Module:
    """
    Загрузка нормализации.

    Параметры:
        normalization (str, "batch" or "group"): Тип нормализации.
        num_channels (int): Количество карт признаков.
        num_groups (int, default None): Количество групп.
        momentum (float, default 0.1): Значение, используемое для вычисления скользящего среднего (running_mean)
            и скользящей дисперсии (running_var). Может быть установлено в None для накопительного подсчета
            среднего значения (то есть простого среднего).
        eps (float, default 1e-5): Значение, добавляемое к знаменателю для численной устойчивости.
        affine (bool, default True): Позволяет иметь обучаемые аффинные параметры.
    """
    if normalization == "group":
        return nn.GroupNorm(
            num_groups=num_groups if num_groups < num_channels else num_channels,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        )
    return nn.BatchNorm2d(num_features=num_channels, eps=eps, affine=affine)

class ResidualBlock(nn.Module):

    """
    Блок сети ResNet.

    Параметры:
        in_channels (int): Количество каналов (слоев карт признаков) на входе.
        out_channels (int): Количество каналов (слоев карт признаков) на выходе.
        activation (str, "silu" or "relu"): Функция активации.
        scale_factor (float, default = 1): Фактор уменьшения размера количества слоев меду входной и выходной свёртками.
        dropout (float, default = 0.0): Вероятность dropout.
        normalization (str, "batch" or "group"): Тип нормализации.
        num_groups (int, default None): Количество групп.
        momentum (float, default 0.1): Значение, используемое для вычисления скользящего среднего (running_mean)
            и скользящей дисперсии (running_var). Может быть установлено в None для накопительного подсчета
            среднего значения (то есть простого среднего).
        eps (float, default 1e-5): Значение, добавляемое к знаменателю для численной устойчивости.
        affine (bool, default True): Позволяет иметь обучаемые аффинные параметры.        
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: str = 'silu',
            scale_factor: int = 4,
            dropout: float = 0.0,
            normalization: str = "group",
            num_groups: int = 32,
            momentum: float = 0.1,
            eps: float = 1e-5,
            affine: bool = True,
        ):
        super().__init__()

        # Количество срытых каналов внутри свёртки.
        hidden_channels = in_channels // scale_factor
        
        self.activation = get_activation(activation)

        # Свёртка.

        self.norm1 = get_normalization(
            normalization=normalization,
            num_channels=in_channels,
            num_groups=num_groups,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,   
            padding=1,
            bias=False,
        )

        nn.init.normal_(self.conv1.weight, 0.0, 0.02)

        self.norm2 = get_normalization(
            normalization=normalization,
            num_channels=hidden_channels,
            num_groups=num_groups,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )        

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        nn.init.normal_(self.conv2.weight, 0.0, 0.02)

        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else nn.Identity()

        # Свёртка для увеличения количество каналов остаточного соединения.
        self.up_channels = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels - in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ) if out_channels > in_channels  else None

        if self.up_channels != None:
            nn.init.normal_(self.up_channels.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Свёртка.
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        x = self.dropout(x)

        # Увеличение каналов остаточного соединения если out_channels > in_channels
        if self.up_channels != None:
            up = self.up_channels(residual)
            residual = torch.cat([residual, up], dim=1)

        # Складываем результат свёртки и остаточного соединения.
        x = x + residual

        return x
    
class Model(nn.Module):

    """
    Параметры:
        num_classes (int): Количество классов на выходе.
        map (list): Карта модели в виде списка хранящего в себе списки определяющие ступени свёртки
            в которых определены количество каналов на выходе блоков ResidualBlock. 
            Соответственно, внешний список определяет количество ступеней, а список определяющий
            количество каналов также определяет количество блоков в каждой ступени.
            После каждой ступени кроме последней будет производится понижение размерности карт признаков.
        activation (str, "relu" or "silu", default "relu"): Функция активации.
        dropout (float, default = 0.0): Вероятность dropout для ResidualBlock.
        normalization (str, "batch" or "group"): Тип нормализации.
        num_groups (int, default 16): Количество групп.
        momentum (float, default 0.1): Значение, используемое для вычисления скользящего среднего (running_mean)
            и скользящей дисперсии (running_var). Может быть установлено в None для накопительного подсчета
            среднего значения (то есть простого среднего).
        eps (float, default 1e-5): Значение, добавляемое к знаменателю для численной устойчивости.
        affine (bool, default True): Позволяет иметь обучаемые аффинные параметры.         
    """

    def __init__(
        self,
        num_classes: int,
        map: list = [
            [64, 64, 128], # 32 -> 16
            [128, 128, 128, 128, 256], # 16 -> 8
            [256, 256, 256, 256, 256, 256], # 8 -> 4
        ],
        activation: str = "relu",
        dropout: float = 0.0,
        normalization: str = "group",
        num_groups: int = 16,
        momentum: float = 0.1,
        eps: float = 1e-5,
        affine: bool = True,        
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        in_channels = map[0][0]

        activation_fn = get_activation(activation=activation)

        conv = nn.Conv2d(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        nn.init.normal_(conv.weight, 0.0, 0.02)
        self.blocks.append(conv)

        self.blocks.append(get_normalization(
            normalization=normalization,
            num_channels=in_channels,
            num_groups=num_groups,
            eps=eps,
            momentum=momentum,
            affine=affine,
        ))
        self.blocks.append(activation_fn)
        # self.blocks.append(nn.MaxPool2d(2))
        
        # Ступени и блоки свёртки.
        for level_index, level in enumerate(map):
            for block_index, out_channels in enumerate(level):
                self.blocks.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        activation=activation,
                        dropout=dropout,
                        normalization=normalization,
                        num_groups=num_groups,
                        momentum=momentum,
                        eps=eps,
                        affine=affine,
                    )
                )
                in_channels = out_channels

            # Закрываем последний блок.
            # self.blocks.append(get_normalization(
            #     normalization=normalization,
            #     num_channels=out_channels,
            #     num_groups=num_groups,
            #     eps=eps,
            #     momentum=momentum,
            #     affine=affine,
            # ))
            # self.blocks.append(activation_fn)

            # Уменьшение размерности.
            if level_index < len(map) - 1:
                self.blocks.append(nn.MaxPool2d(2))

        # Закрываем последний блок.
        self.blocks.append(get_normalization(
            normalization=normalization,
            num_channels=out_channels,
            num_groups=num_groups,
            eps=eps,
            momentum=momentum,
            affine=affine,
        ))

        self.blocks.append(activation_fn)

        # Уменьшаем размерность карт признаков в два раза.
        # self.blocks.append(nn.MaxPool2d(2))

        self.fc = nn.Linear(out_channels, num_classes, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 0.02)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Пропускаем данные через блоки.
        for block in self.blocks:
            x = block(x)

        # Выпрямляем и усредняем.
        # Аналог AvgPool2d.
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        x = torch.mean(x, dim=2)

        # Вычисляем вероятность с помощью полносвязного слоя.
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    pass