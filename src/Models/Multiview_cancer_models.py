import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch as th
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning.loggers import WandbLogger
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
            if isinstance(self.logger, WandbLogger):
                wandb.log({self.confmat_titles[i]: wandb.Image(fig)})
                plt.close(fig)


class Four_view_single_featurizer(nn.Module):
    """
    nn.Module encapsulating a single resnet and adding an extra linear layer.
    """

    def __init__(self, num_class, drop=0.3):
        super(Four_view_single_featurizer, self).__init__()

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


class Four_view_two_branch_model(Breast_backbone):
    """
    A model containing four individual resnets for each view image. The model has two branches, one for the left breast and one for the right breast.
    it gives two predictions - one for each breast.
    TODO - train with two separate optimizers, one for each branch
    """

    def __init__(self, num_class, weights_file=None, drop=0.3, learning_rate=1e-3):
        super(Four_view_two_branch_model, self).__init__(num_class, learning_rate)

        self.confusion_matrix = nn.ModuleList(
            [MulticlassConfusionMatrix(num_classes=num_class) for _ in range(3)]
        )
        self.confmat_titles = [
            "Left_Confusion_Matrix",
            "Right_Confusion_Matrix",
            "Overall_Confusion_Matrix",
        ]

        # Define 4 separate internal resnets separate for each view image
        self.resnets = nn.ModuleList(
            [
                Four_view_single_featurizer(num_class, drop)
                for _ in range(4)
                # models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                # for _ in range(4)
            ]
        )

        if weights_file is not None:
            for i, resnet in enumerate(self.resnets):
                resnet.load_state_dict(
                    th.load(weights_file, map_location=th.device("cpu"))["state_dict"]
                )

        for resnet in self.resnets:
            resnet.fc = nn.Identity()

        self.fc_left = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, num_class),
        )

        self.fc_right = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        # x is a list of 4 images from 4 views - LCC, LMLO, RCC, RMLO
        x = [self.resnets[i](image) for i, image in enumerate(x)]
        x_left = th.cat([x[0], x[1]], dim=1)
        x_right = th.cat([x[2], x[3]], dim=1)

        out_left = self.fc_left(x_left)
        out_right = self.fc_right(x_right)

        return out_left, out_right

    def compute_branch_metrics(self, y_left, y_right, y, prefix: str = None):
        # compute separate metrics for each branch
        metrics_left = self.compute_metrics(y_left, y[:, 0], postfix="_left")
        metrics_right = self.compute_metrics(y_right, y[:, 1], postfix="_right")
        # loss for the optimizer
        loss = metrics_left["loss_left"] + metrics_right["loss_right"]
        # find the branch tha returns higher cancer score
        y_max = th.max(y, dim=1).values
        y_left_pred = th.argmax(y_left, dim=1)
        y_right_pred = th.argmax(y_right, dim=1)

        # keeps the logits of the branch with the higher target value - if both branches predict the same score, the right branch is chosen
        worse_pred = y_left_pred > y_right_pred
        worse_pred = th.where(worse_pred[:, None], y_left, y_right)

        # computes the metrics for the worse predicted case
        metrics_overall = self.compute_metrics(worse_pred, y_max, postfix="_overall")
        metrics = {**metrics_left, **metrics_right, **metrics_overall}
        if prefix is not None:
            metrics = {prefix + key: value for key, value in metrics.items()}
        return loss, metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_left, y_right = self.forward(x)
        loss, metrics = self.compute_branch_metrics(y_left, y_right, y, prefix="train_")
        self.log_dict(metrics, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_left, y_right = self.forward(x)
        loss, metrics = self.compute_branch_metrics(y_left, y_right, y, prefix="val_")
        self.log_dict(metrics, sync_dist=True)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_left, y_right = self.forward(x)
        loss, metrics = self.compute_branch_metrics(y_left, y_right, y, prefix="test_")
        self.log_dict(metrics, sync_dist=True)

        y_max = th.max(y, dim=1).values
        y_left_pred = th.argmax(y_left, dim=1)
        y_right_pred = th.argmax(y_right, dim=1)
        # keeps the logits of the branch with the higher target value
        y_worse_pred = th.maximum(y_left, y_right)

        self.confusion_matrix[0].update(y_left_pred, y[:, 0])
        self.confusion_matrix[1].update(y_right_pred, y[:, 1])
        self.confusion_matrix[2].update(y_worse_pred, y_max)
        return loss


class Four_view_two_branch_model1(Four_view_two_branch_model):
    """
    A 4v2b model with separate optimizers for each branch
    """

    def __init__(self, num_class, weights_file=None, drop=0.3, learning_rate=1e-3):
        super(Four_view_two_branch_model, self).__init__(
            num_class, weights_file, drop, learning_rate
        )

    def configure_optimizers(self):
        optimizers = [
            th.optim.Adam(
                [
                    {self.resnets[0].parameters()},
                    {self.resnets[1].parameters()},
                    {self.fc_left.parameters()},
                ],
                lr=self.learning_rate,
            ),
            th.optim.Adam(
                [
                    {self.resnets[2].parameters()},
                    {self.resnets[3].parameters()},
                    {self.fc_right.parameters()},
                ],
                lr=self.learning_rate,
            ),
        ]
        return optimizers


class Four_view_featurizers(Breast_backbone):
    """
    This class is used to train a model that uses 4 separate resnets to extract features from 4 different views of the breast. The four view models are
    trained separately, each with their own optimizer. Individual resnets are included  in one model to benefit from using a single dataloader - this
    is probably suboptimal as you cannot chose an optimal epoch for a single resnet.
    """

    def __init__(self, num_class, drop=0.3, learning_rate=1e-3):
        super(Four_view_featurizers, self).__init__(num_class, learning_rate)

        self.featurizers = nn.ModuleList(
            [Four_view_single_featurizer(num_class, drop) for _ in range(4)]
        )
        self.confusion_matrix = nn.ModuleList(
            [MulticlassConfusionMatrix(num_classes=num_class) for _ in range(4)]
        )
        self.confmat_titles = [f"Confusion_Matrix_{i}" for i in range(4)]

        self.automatic_optimization = False

    def forward(self, x):
        return [featurizer(x[i]) for i, featurizer in enumerate(self.featurizers)]

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizers = self.optimizers()
        y_hats = self(x)
        # Compute metrics (loss, f1 score, accuracy) for each prediction
        metrics = {
            k: v
            for i, y_hat in enumerate(y_hats)
            for k, v in self.compute_metrics(
                y_hat, y[:, 0 if i < 2 else 1], prefix="train_", postfix=f"_{i}"
            ).items()
        }

        # Iterate over each optimizer and zero the gradients, perform manual backward pass and update the model parameters
        for i, opt in enumerate(optimizers):
            opt.zero_grad()
            self.manual_backward(metrics[f"train_loss_{i}"])
            opt.step()
        # Log the metrics
        self.log_dict(metrics, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hats = self(x)
        metrics = {
            k: v
            for i, y_hat in enumerate(y_hats)
            for k, v in self.compute_metrics(
                y_hat, y[:, 0 if i < 2 else 1], prefix="val_", postfix=f"_{i}"
            ).items()
        }
        self.log_dict(metrics, sync_dist=True)
        self.log(
            "val_loss", th.stack([metrics[f"val_loss_{i}"] for i in range(4)]).mean()
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hats = self(x)
        metrics = {
            k: v
            for i, y_hat in enumerate(y_hats)
            for k, v in self.compute_metrics(
                y_hat, y[:, 0 if i < 2 else 1], prefix="test_", postfix=f"_{i}"
            ).items()
        }
        self.log_dict(metrics, sync_dist=True)

        for i, y_hat in enumerate(y_hats):
            self.confusion_matrix[i].update(
                th.argmax(y_hat, dim=1), y[:, 0 if i < 2 else 1]
            )

    def configure_optimizers(self):
        optimizers = [
            th.optim.Adam(featurizer.parameters(), lr=self.learning_rate)
            for featurizer in self.featurizers
        ]
        return optimizers


class Two_view_model(Breast_backbone):
    """
    A model that uses two resnets to extract features from two views of the breast. The two views are trained separately and then concatenated to a single
    linear layer that outputs the final prediction.
    """

    def __init__(self, num_class, weights_file=None, drop=0.3, learning_rate=1e-3):
        super(Two_view_model, self).__init__(num_class, learning_rate)

        # two separate featurizers for CC an MLO views respectively
        self.resnets = nn.ModuleList(
            [Four_view_single_featurizer(num_class, drop) for _ in range(2)]
        )

        if weights_file is not None:
            for i, resnet in enumerate(self.resnets):
                resnet.load_state_dict(
                    th.load(weights_file, map_location=th.device("cpu"))["state_dict"]
                )

        for resnet in self.resnets:
            resnet.fc = nn.Linear(512, 128)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x = [self.resnets[i](image) for i, image in enumerate(x)]
        x = th.cat(x, dim=1)
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

    def get_resnet_outputs(self, batch):
        self.eval()
        with th.no_grad():
            x, y = batch
            x = [self.resnets[i](image) for i, image in enumerate(x)]
        self.train()
        return x
