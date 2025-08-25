import os
import sys

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch as th
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.classification import Accuracy, F1Score, MulticlassConfusionMatrix

from Mirai_Risk_Prediction_Model.asymmetry_model.mirai_localized_dif_head import (
    extract_mirai_backbone,
)

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Add the Mirai model directory to Python path
mirai_path = os.path.join(project_root, "Mirai_Risk_Prediction_Model")
sys.path.insert(0, mirai_path)


class Breast_backbone(pl.LightningModule):
    """
    A base class for the models that are used to classify breast cancer. They inherit from this.
    """

    def __init__(self, num_class, learning_rate=1e-3):
        super(Breast_backbone, self).__init__()

        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.save_hyperparameters()  # Stores all arguments passed to __init__

        self.confusion_matrix = nn.ModuleList(
            [MulticlassConfusionMatrix(num_classes=num_class)]
        )
        self.confmat_titles = "Confusion Matrix"

        self.f1 = F1Score(num_classes=num_class, average="macro", task="multiclass")
        self.accuracy = Accuracy(
            num_classes=num_class, average="macro", task="multiclass"
        )

        self.check_path = "checkpoints/best_model.ckpt"

    def compute_metrics(self, y_hat, y, prefix: str = None, postfix: str = None):
        y_pred = th.argmax(y_hat, dim=1)
        metrics = {
            "loss": self.loss(y_hat, y),
            "f1": self.f1(y_pred, y),
            "acc": self.accuracy(y_pred, y),
        }
        if prefix is not None:
            metrics = {prefix + key: value for key, value in metrics.items()}
        if postfix is not None:
            metrics = {key + postfix: value for key, value in metrics.items()}
        return metrics

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

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
                ax.set_title(self.confmat_titles)
            else:
                ax.set_title(self.confmat_titles[i])

            # Log confusion matrix to wandb
            if isinstance(self.logger, WandbLogger):
                wandb.log({self.confmat_titles[i]: wandb.Image(fig)})
                plt.close(fig)


class Mirai_two_view_model(Breast_backbone):
    """
    A model that uses two MIRAI encoders to extract features from two views of the breast. The two views are trained separately and then concatenated to a single
    linear layer that outputs the final prediction.
    """

    def __init__(self, num_class, drop=0.3, learning_rate=1e-3, task=1):
        super(Mirai_two_view_model, self).__init__(num_class, learning_rate)

        # two separate featurizers for CC an MLO views respectively
        self.task = task

        assert self.task in [1, 2], "Task must be either 1 (cancer) or 2 (density)"

        self.resnet = extract_mirai_backbone(
            os.path.join(
                mirai_path, "asymmetry_model/mgh_mammo_MIRAI_Base_May20_2019.p"
            )
        )

        # Freeze the MIRAI backbone - prevent parameters from being updated during training
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.cc_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.Dropout(drop),
        )

        self.mlo_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.Dropout(drop),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        cc = self.cc_fc(self.resnet(x[0]))
        mlo = self.mlo_fc(self.resnet(x[1]))
        x = th.cat([cc, mlo], dim=1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y1, y2 = batch
        y = y1 if self.task == 1 else y2
        y_hat = self(x)
        metrics = self.compute_metrics(y_hat, y, prefix="train_")
        self.log_dict(metrics, sync_dist=True)
        return metrics["train_loss"]

    def validation_step(self, batch, batch_idx):
        x, y1, y2 = batch
        y = y1 if self.task == 1 else y2
        y_hat = self(x)
        metrics = self.compute_metrics(y_hat, y, prefix="val_")
        self.log_dict(metrics, sync_dist=True)
        return metrics["val_loss"]

    def test_step(self, batch, batch_idx):
        x, y1, y2 = batch
        y = y1 if self.task == 1 else y2
        y_hat = self(x)
        metrics = self.compute_metrics(y_hat, y, prefix="test_")
        self.log_dict(metrics, sync_dist=True)
        self.confusion_matrix[0].update(th.argmax(y_hat, dim=1), y)
        return metrics["test_loss"]

    def get_resnet_outputs(self, batch):
        self.eval()
        with th.no_grad():
            x, y1, y2 = batch
            x = [self.cc(self.resnet(x[0])), self.mlo(self.resnet(x[1]))]
        self.train()
        return x


if __name__ == "__main__":
    # Example usage
    model = Mirai_two_view_model(num_class=5, drop=0.5, learning_rate=1e-5, task=2)
    print(model)

    # Example input
    tens = th.randn(2, 3, 512, 512)
    out = model([tens, tens])
    print(out.shape)  # Should print the shape of the output tensor
