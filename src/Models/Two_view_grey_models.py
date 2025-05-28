import sys
import time
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch as th
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import Bottleneck, ResNet

import wandb

print("Using custom ResNet model for two-view CNN.")


class MyResNet(ResNet):
    def __init__(
        self,
        block: type[Union[Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_channels: int = 1,
        inplanes: int = 16,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )
        print(f"[TRAIN_SYNT_DEBUG] Inside init of myresnet")
        sys.stdout.flush()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.input_channels = input_channels

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            self.input_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        print(f"[TRAIN_SYNT_DEBUG] before calling _make_layer")
        sys.stdout.flush()

        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(
            block,
            self.inplanes * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            self.inplanes * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            self.inplanes * 8,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]


def resnext29_16x4d(*, weights=None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-29 model with 16 groups and a width of 4."""
    _ovewrite_named_param(kwargs, "groups", 16)
    _ovewrite_named_param(kwargs, "width_per_group", 4)

    model = MyResNet(block=Bottleneck, layers=[2, 2, 2, 2], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class TwoViewCNN(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        task=1,
        num_views=2,
        input_channels=1,
        resnext_inplanes=16,
        learning_rate=1e-3,
    ):
        super(TwoViewCNN, self).__init__()
        self.num_classes = num_classes
        self.num_views = num_views
        self.learning_rate = learning_rate
        self.task = task  # 1 for task1, 2 for task2
        self.confmat_titles = ["Confusion Matrix"]
        self.save_hyperparameters()

        assert self.task in [1, 2], "Task must be either 1 or 2"

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = nn.ModuleList(
            [MulticlassConfusionMatrix(num_classes=num_classes)]
        )

        self.check_path = "checkpoints/best_model.ckpt"

        self.resnexts = nn.ModuleList()
        for _ in range(num_views):
            print(f"[TRAIN_SYNT_DEBUG] Appending resnext {_}")
            sys.stdout.flush()

            self.resnexts.append(
                #        ResNeXt(
                #            cardinality=4,
                #            depth=11,
                #            nlabels=num_classes,
                #            base_width=2,
                #            widen_factor=2,
                resnext29_16x4d(
                    weights=None,
                    progress=True,
                    num_classes=num_classes,
                    input_channels=input_channels,
                    inplanes=resnext_inplanes,
                )
            )
        # change the classifier layer to identity
        for resnext in self.resnexts:
            #    resnext.classifier = nn.Identity()
            resnext.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2 * 512, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

        # Define metrics
        self.train_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=num_classes, task="multiclass")

        self.batch_start_time = time.time()

    def forward(self, x):
        # print(len(x), x[0].shape)
        x = [resnext(x_i) for resnext, x_i in zip(self.resnexts, x)]
        # print(len(x), x[0].shape)
        x = th.cat(x, dim=1)
        # print(x.shape)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y1, y2 = batch
        y = y1 if self.task == 1 else y2
        # x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y1, y2 = batch
        y = y1 if self.task == 1 else y2

        logits = self(x)

        self.log("val_loss", self.loss(logits, y))
        self.log("val_accuracy", self.val_accuracy(logits, y))

    def test_step(self, batch, batch_idx):
        x, y1, y2 = batch
        y = y1 if self.task == 1 else y2

        logits = self(x)

        self.log("test_loss", self.loss(logits, y))
        self.log("test_accuracy", self.test_accuracy(logits, y))

        # Update confusion matrix
        self.confusion_matrix[0].update(th.argmax(logits, dim=1), y)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer]

    @rank_zero_only
    def on_test_epoch_start(self):
        for i in range(len(self.confusion_matrix)):
            # Reset confusion matrix if it was used before
            self.confusion_matrix[i].reset()

    @rank_zero_only
    def on_test_epoch_end(self):
        for i in range(len(self.confusion_matrix)):
            # Compute confusion matrix
            cm = self.confusion_matrix[i].compute().cpu().numpy()

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            if len(self.confmat_titles) == 1:
                ax.set_title(self.confmat_titles[0])
            else:
                ax.set_title(self.confmat_titles[i])

            # Log confusion matrix to wandb
            if isinstance(self.logger, WandbLogger):
                wandb.log({self.confmat_titles[i]: wandb.Image(fig)})
                plt.close(fig)


if __name__ == "__main__":
    print(
        "This is a module for TwoViewCNN model. It should not be run directly. and fuck of"
    )
