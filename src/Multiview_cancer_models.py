import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch as th
import torch.nn as nn
import torchvision.models as models
import wandb
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.classification import (Accuracy, F1Score,
                                         MulticlassConfusionMatrix)
from torchvision.models import ResNet18_Weights


class Breast_backbone(pl.LightningModule):
    def __init__(self, num_class, learning_rate=1e-3):
        super(Breast_backbone, self).__init__()

        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.max_val_f1 = 0
        self.best_epoch = 0

        self.test_pred = []  # collect predictions
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_class)

        self.f1 = F1Score(num_classes=num_class, average="macro", task="multiclass")
        self.accuracy = Accuracy(
            num_classes=num_class, average="macro", task="multiclass"
        )

        self.check_path = "wandb/Weights_checkpoint.pth"

    def compute_metrics(self, y_hat, y):
        y_pred = th.argmax(y_hat, dim=1)
        return self.loss(y_hat, y), self.f1(y_pred, y), self.accuracy(y_pred, y)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_validation_epoch_end(self):
        validation_f1 = self.trainer.callback_metrics["val_f1"]
        if validation_f1 > self.max_val_f1:
            self.max_val_f1 = validation_f1
            th.save(self.state_dict(), self.check_path)
            self.best_epoch = self.current_epoch

    @rank_zero_only
    def on_train_end(self):
        print(f"Best epoch was epoch {self.best_epoch}. Loading its state.")
        self.load_state_dict(th.load(self.check_path))

    @rank_zero_only
    def on_test_epoch_end(self):
        # Reset confusion matrix if it was used before
        if self.confusion_matrix is not None:
            self.confusion_matrix.reset()
        
        # Compute confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # Log confusion matrix to wandb
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        
        
    
class Four_view_two_branch_model(Breast_backbone):
    def __init__(self, num_class, drop = 0.3, learning_rate=1e-3):
        super(Four_view_two_branch_model, self).__init__(num_class, learning_rate)

        # Define 4 separate internal resnets separate for each view image
        self.resnets = nn.ModuleList([models.resnet18(weights = 'DEFAULT') for _ in range(4)])
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
        loss_left, f1_left, acc_left = self.compute_metrics(y_left, y[0])
        loss_right, f1_right, acc_right = self.compute_metrics(y_right, y[1])
        # loss for the optimizer
        loss = loss_left + loss_right
        # find the branch tha returns higher cancer score
        y_max = th.max(y[0], y[1], dim=1).values
        y_left_pred = th.argmax(y_left, dim=1)
        y_right_pred = th.argmax(y_right, dim=1)
        # keeps the logits of the branch with the higher target value - if both branches predict the same score, the right branch is chosen
        worse_pred = th.where(y_left_pred > y_right_pred, y_left, y_right)        
        # computes the metrics for the worse predicted case
        loss_overall, f1_overall, acc_overall = self.compute_metrics(worse_pred, y_max)
        metrics = {
            "loss": loss,
            "loss_left": loss_left,
            "loss_right": loss_right,
            "loss_overall": loss_overall,
            "f1_left": f1_left,
            "f1_right": f1_right,
            "f1_overall": f1_overall,
            "acc_left": acc_left,
            "acc_right": acc_right,
            "acc_overall": acc_overall
        }
        if prefix is not None:
            metrics = {prefix + key: value for key, value in metrics.items()}
        return loss, metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_left, y_right = self.forward(x)
        loss, metrics  = self.compute_branch_metrics(y_left, y_right, y, prefix = 'train_')
        self.log_dict(metrics, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_left, y_right = self.forward(x)
        loss, metrics  = self.compute_branch_metrics(y_left, y_right, y, prefix = 'val_')
        self.log_dict(metrics, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_left, y_right = self.forward(x)
        loss, metrics  = self.compute_branch_metrics(y_left, y_right, y, prefix = 'test_')
        self.log_dict(metrics, sync_dist=True)
        
        y_max = th.max(y[0], y[1], dim=1).values
        y_left_pred = th.argmax(y_left, dim = 1)
        y_right_pred = th.argmax(y_right, dim = 1)
        # keeps the logits of the branch with the higher target value
        worse_pred = th.where(y_left_pred > y_right_pred, y_left_pred, y_right_pred)
        
        self.test_pred.extend(worse_pred.cpu().numpy())
        self.confusion_matrix.update(worse_pred, y_max)
        return loss