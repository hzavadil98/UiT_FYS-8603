import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch as th
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.classification import Accuracy, F1Score, MulticlassConfusionMatrix

import wandb


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
            ax.set_title(self.confmat_titles[i])

            # Log confusion matrix to wandb
            wandb.log({self.confmat_titles[i]: wandb.Image(fig)})
            plt.close(fig)


class Four_view_single_featurizer(Breast_backbone):
    """
    nn.Module encapsulating a single resnet and adding an extra linear layer.
    """

    def __init__(self, num_class, drop=0.3, learning_rate=1e-3, view: int = 0):
        super(Four_view_single_featurizer, self).__init__(num_class, learning_rate)

        self.confmat_titles = [f"Confusion Matrix view-{view}"]

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = self.compute_metrics(y_hat, y, prefix="train_")
        self.log_dict(metrics, sync_dist=True)
        return metrics["train_loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = self.compute_metrics(y_hat, y, prefix="val_")
        self.log_dict(metrics, sync_dist=True)
        return metrics["val_loss"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = self.compute_metrics(y_hat, y, prefix="test_")
        self.log_dict(metrics, sync_dist=True)

        self.confusion_matrix[0].update(th.argmax(y_hat, dim=1), y)
        return metrics["test_loss"]
