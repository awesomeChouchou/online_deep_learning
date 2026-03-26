# Used AI assistance for conceptual hints and logic structuring.
from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            self.batchnorm = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(2)
          

        def forward(self, x):
            x = self.c1(x)
            x = self.batchnorm(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x


    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        cnn_layers = []
        c1 = in_channels
        c2 = 32

        for _ in range(3):
            cnn_layers.append(self.Block(c1, c2))
            c1 = c2
            c2 *= 2

        self.network = torch.nn.Sequential(*cnn_layers)

        #20%의 뉴런을 무작위로 끔
        # self.dropout = nn.Dropout(0.2)
        # self.classifier = nn.Linear(c1*4*4, num_classes)

        # self.gap = nn.AdaptiveAvgPool2d(1) # 어떤 크기든 1x1로 압축 (B, C, 1, 1)
        # self.classifier = nn.Sequential(
        # nn.Linear(c1, 128),  # 중간에 차원을 살짝 키워 일반화 성능 확보
        # nn.ReLU(),
        # nn.BatchNorm1d(128), # 1D 데이터용 배치 정규화
        # nn.Dropout(0.3),     # 드롭아웃 비율을 조금 더 높여도 됨
        # nn.Linear(128, num_classes)
        # )
        self.classifier = nn.Sequential(
        nn.Linear(c1 * 8 * 8, 256), 
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5), # 과적합 방지를 위해 조금 높게 설정
        nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        features = self.network(z)

        # pooled = features.mean(dim=[2,3])
        # flat = features.view(features.size(0),-1)
        # flat = self.dropout(flat)
        # logits = self.classifier(pooled)
        # logits = self.classifier(flat)


        # x = self.gap(features)
        # x = torch.flatten(x, 1) # (B, C)

        x = torch.flatten(features, 1) # (B, 8192)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    class UpBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1)
            self.conv = nn.Sequential(
                nn.BatchNorm2d(out_c * 2),  # after concat with skip
                nn.ReLU(),
                nn.Conv2d(out_c * 2, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        def forward(self, x, skip):
            x = self.up(x)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Encoder
        self.down1 = self.DownBlock(in_channels, 32)   # /2
        self.down2 = self.DownBlock(32, 64)             # /4
        self.down3 = self.DownBlock(64, 128)            # /8
        self.down4 = self.DownBlock(128, 256)           # /16

        # Decoder with skip connections
        self.up0 = self.UpBlock(256, 128)   # /8
        self.up1 = self.UpBlock(128, 64)    # /4
        self.up2 = self.UpBlock(64, 32)     # /2
        self.up3 = self.UpBlock(32, 32)     # /1 (skip from input conv)

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Heads
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.depth_regressor = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        s0 = self.input_conv(z)  # (B, 32, H, W)
        s1 = self.down1(z)       # (B, 32, H/2, W/2)
        s2 = self.down2(s1)      # (B, 64, H/4, W/4)
        s3 = self.down3(s2)      # (B, 128, H/8, W/8)
        s4 = self.down4(s3)      # (B, 256, H/16, W/16)

        # Decoder with skip connections
        u0 = self.up0(s4, s3)    # (B, 128, H/8, W/8)
        u1 = self.up1(u0, s2)    # (B, 64, H/4, W/4)
        u2 = self.up2(u1, s1)    # (B, 32, H/2, W/2)
        u3 = self.up3(u2, s0)    # (B, 32, H, W)

        logits = self.classifier(u3)
        depth = torch.sigmoid(self.depth_regressor(u3)).squeeze(1)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        return pred, raw_depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()

