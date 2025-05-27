import time

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import init
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix

import wandb


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(
        self, in_channels, out_channels, stride, cardinality, base_width, widen_factor
    ):
        """Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 4.0)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(
            D,
            D,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "shortcut_conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
            )
            self.shortcut.add_module("shortcut_bn", nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.gelu(self.bn_reduce.forward(bottleneck))
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.gelu(self.bn.forward(bottleneck))
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.gelu(residual + bottleneck)


class ResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [
            4,
            4 * self.widen_factor,
            8 * self.widen_factor,
            16 * self.widen_factor,
        ]

        self.conv_1_3x3 = nn.Conv2d(1, 4, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(4)
        self.stage_1 = self.block("stage_1", self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block("stage_2", self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block("stage_3", self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3] * 4, nlabels)
        init.kaiming_normal_(self.classifier.weight)

        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode="fan_out")
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = "%s_bottleneck_%d" % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(
                    name_,
                    ResNeXtBottleneck(
                        in_channels,
                        out_channels,
                        pool_stride,
                        self.cardinality,
                        self.base_width,
                        self.widen_factor,
                    ),
                )
            else:
                block.add_module(
                    name_,
                    ResNeXtBottleneck(
                        out_channels,
                        out_channels,
                        1,
                        self.cardinality,
                        self.base_width,
                        self.widen_factor,
                    ),
                )
        return block

    def forward(self, x):
        #    print(x.shape)
        x = self.conv_1_3x3.forward(x)
        #    print(x.shape)
        x = F.gelu(self.bn_1.forward(x))
        x = self.stage_1.forward(x)
        #    print(x.shape)
        x = self.stage_2.forward(x)
        #    print(x.shape)
        x = self.stage_3.forward(x)
        #    print(x.shape)
        x = F.avg_pool2d(x, 64, 64)
        #    print(x.shape)
        # x = x.view(-1, self.stages[3])
        x = x.view(x.size(0), -1)
        #    print(x.shape)
        return self.classifier(x)


class TwoViewCNN(pl.LightningModule):
    def __init__(self, num_classes, num_views=2, learning_rate=1e-3):
        super(TwoViewCNN, self).__init__()
        self.num_classes = num_classes
        self.num_views = num_views
        self.learning_rate = learning_rate
        self.confmat_titles = ["Confusion Matrix"]
        self.save_hyperparameters()

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = nn.ModuleList(
            [MulticlassConfusionMatrix(num_classes=num_classes)]
        )

        self.check_path = "checkpoints/best_model.ckpt"

        self.resnexts = nn.ModuleList()
        for _ in range(num_views):
            self.resnexts.append(
                ResNeXt(
                    cardinality=4,
                    depth=20,
                    nlabels=num_classes,
                    base_width=2,
                    widen_factor=2,
                )
            )
        # change the classifier layer to identity
        for resnext in self.resnexts:
            resnext.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 64), nn.GELU(), nn.Linear(64, num_classes)
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
        x, y, _ = batch
        batch_load_time = time.time() - self.batch_start_time
        # print(f"Time to get batch: {batch_load_time:.4f} seconds")

        process_start_time = time.time()
        logits = self(x)
        loss = self.loss(logits, y)
        process_time = time.time() - process_start_time
        # print(f"Time to process batch: {process_time:.4f} seconds")

        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(logits, y))
        self.batch_start_time = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)

        self.log("val_loss", self.loss(logits, y))
        self.log("val_accuracy", self.val_accuracy(logits, y))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
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
