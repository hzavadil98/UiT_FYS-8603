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
            if isinstance(self.confmat_titles, str):
                title = self.confmat_titles
            elif len(self.confmat_titles) == 1:
                title = self.confmat_titles[0]
            else:
                title = self.confmat_titles[i]
            ax.set_title(title)

            # Log confusion matrix to wandb
            if isinstance(self.logger, WandbLogger):
                wandb.log({title: wandb.Image(fig)})
            plt.close(fig)


class Single_view_model(Breast_backbone):
    """
    nn.Module encapsulating a single resnet and adding an extra linear layer.
    """

    def __init__(
        self, num_class, weights_file=None, drop=0.3, learning_rate=1e-3, task=1
    ):
        super(Single_view_model, self).__init__(num_class, learning_rate)
        print("inner flag 1")
        self.task = task

        assert task in [1, 2], "Task must be 1 (cancer) or 2 (density)"
        print("inner flag 1.5")
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        print("inner flag 2")
        self.fc = nn.Sequential(
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)

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

    def get_resnet_outputs(self, x):
        was_training = self.training
        self.eval()
        original_fc = self.resnet.fc
        self.resnet.fc = nn.Identity()
        with th.no_grad():
            x = self.resnet(x)
        self.resnet.fc = original_fc
        if was_training:
            self.train()
        return x

    def get_activation(self, name):
        """
        Hook function to capture activations from a layer.
        """

        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def get_gradient(self, name):
        """
        Hook function to capture gradients from a layer.
        """

        def hook(model, input, output):
            self.gradients[name] = output[0].detach()

        return hook

    def generate_weighted_cam(self, x, cam_weights):
        """
        Generate Weighted CAM heatmap for the input image.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            cam_weights: vector of the weights to use in CAM averaging (eg. AJIVE loadings), must have the same length as the output number of features

        Returns:
            cam: Weighted-CAM heatmap of shape (batch_size, height, width) normalized between 0-1.
            predicted_class: Predicted class for the input
            ajive_score: Scalar product of GA-pooled feature maps with cam_weights
        """
        # checking that cam_weights is of the shape (1, num_features, 1, 1) for broadcasting
        cam_weights_original = (
            cam_weights.clone()
            if isinstance(cam_weights, th.Tensor)
            else th.tensor(cam_weights)
        )
        if cam_weights_original.dim() > 1:
            cam_weights_original = cam_weights_original.squeeze()

        cam_weights = (
            cam_weights
            if isinstance(cam_weights, th.Tensor)
            else th.tensor(cam_weights)
        )
        if cam_weights.dim() == 1:
            cam_weights = cam_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif cam_weights.dim() == 2 and cam_weights.shape[0] == 1:
            cam_weights = cam_weights.unsqueeze(2).unsqueeze(3)
        elif cam_weights.dim() == 4:
            pass
        else:
            raise ValueError(
                "cam_weights must be of shape (num_features,), (1, num_features), or (1, num_features, 1, 1)"
            )

        self.eval()
        self.activations = {}

        # Register hooks on the last convolutional layer (layer4)
        self.resnet.layer4[-1].register_forward_hook(self.get_activation("layer4"))

        # Forward pass - clone to make it a leaf variable that can require gradients
        x = x.clone().detach().requires_grad_(False)
        output = self(x)

        predicted_class = output.argmax(dim=1)

        activations = self.activations["layer4"]

        # Compute AJIVE score: scalar product of GA-pooled activations with cam_weights
        # Global average pooling over spatial dimensions (H, W)
        ga_pooled_activations = activations.mean(
            dim=[2, 3]
        )  # Shape: (batch_size, num_features)

        # Scalar product with cam_weights (use original 1D version)
        ajive_score = (ga_pooled_activations * cam_weights_original).sum(
            dim=1
        )  # Shape: (batch_size,)

        # Compute weighted combination of activations
        cam = (cam_weights * activations).sum(dim=1)

        # Apply ReLU
        cam = th.nn.functional.relu(cam)

        # Normalize
        batch_size = cam.shape[0]
        for i in range(batch_size):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            if cam_max > cam_min:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)

        return cam.detach().cpu(), predicted_class, ajive_score.detach().cpu()

    def generate_grad_cam(self, x, target_class=None):
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            target_class: Target class for computing gradients. If None, uses predicted class.

        Returns:
            cam: Grad-CAM heatmap of shape (batch_size, height, width)
        """
        self.eval()
        self.activations = {}
        self.gradients = {}

        # Register hooks on the last convolutional layer (layer4)
        self.resnet.layer4[-1].register_forward_hook(self.get_activation("layer4"))
        self.resnet.layer4[-1].register_full_backward_hook(self.get_gradient("layer4"))

        # Forward pass - clone to make it a leaf variable that can require gradients
        x = x.clone().detach().requires_grad_(True)
        output = self(x)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass
        self.zero_grad()
        one_hot = th.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        gradients = self.gradients["layer4"]
        activations = self.activations["layer4"]

        # Global average pooling of gradients over height and width dimensions
        weights = gradients.mean(dim=[2, 3], keepdim=True)

        # Compute weighted combination of activations
        cam = (weights * activations).sum(dim=1)

        # Apply ReLU
        cam = th.nn.functional.relu(cam)

        # Normalize
        batch_size = cam.shape[0]
        for i in range(batch_size):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            if cam_max > cam_min:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)

        return cam.detach().cpu(), target_class


class Four_view_single_featurizer(Breast_backbone):
    """
    nn.Module encapsulating a single resnet and adding an extra linear layer.
    """

    def __init__(
        self, num_class, weights_file=None, drop=0.3, learning_rate=1e-3, view: int = 0
    ):
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

        if weights_file is not None:
            self.load_state_dict(
                th.load(weights_file, map_location=th.device("cpu"))["state_dict"]
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


class Single_view_AJIVE_heads(Breast_backbone):
    """
    nn.Module encapsulating a single resnet and providing functionality to fix AJIVE projections on the features and train these heads only.
    """

    head_names = ("joint", "individual", "both")

    def __init__(
        self,
        model: Single_view_model,
        ajive_model,
        num_class,
        learning_rate=1e-3,
        hidden_dim: int = 128,
        drop: float = 0.3,
        active_head: str = "both",
        view_index: int | None = None,
    ):
        super().__init__(num_class, learning_rate)

        self.backbone = model
        self.ajive_model = ajive_model
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.drop = drop

        if not hasattr(self.backbone, "task"):
            raise ValueError("model must expose a task attribute with value 1 or 2")

        self.task = self.backbone.task
        if self.task not in [1, 2]:
            raise ValueError("model.task must be 1 (cancer) or 2 (density)")

        self.view_index = self.task - 1 if view_index is None else view_index
        if self.view_index not in [0, 1]:
            raise ValueError("view_index must be 0 (cancer) or 1 (density)")

        if active_head not in self.head_names:
            raise ValueError(f"active_head must be one of {self.head_names}")
        self.active_head = active_head

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        view_specific = self._get_view_specific_block()
        joint_loadings = self._to_2d_tensor(view_specific.joint_.loadings_)
        individual_loadings = self._to_2d_tensor(view_specific.individual_.loadings_)

        if joint_loadings.shape[1] == 0:
            raise ValueError("AJIVE joint loadings are empty")
        if individual_loadings.shape[1] == 0:
            raise ValueError("AJIVE individual loadings are empty")

        both_loadings = th.cat([joint_loadings, individual_loadings], dim=1)

        self.register_buffer("joint_loadings", joint_loadings)
        self.register_buffer("individual_loadings", individual_loadings)
        self.register_buffer("both_loadings", both_loadings)

        self.projections = nn.ModuleDict(
            {
                "joint": self._make_fixed_projection(self.joint_loadings),
                "individual": self._make_fixed_projection(self.individual_loadings),
                "both": self._make_fixed_projection(self.both_loadings),
            }
        )

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.projections[name].out_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(hidden_dim, num_class),
                )
                for name in self.head_names
            }
        )

        self.confmat_titles = [f"AJIVE {self.active_head} head confusion matrix"]

    def _get_view_specific_block(self):
        view_specific = getattr(self.ajive_model, "view_specific_", None)
        if view_specific is None:
            view_specific = getattr(self.ajive_model, "view_specific", None)
        if view_specific is None:
            raise AttributeError(
                "AJIVE model must expose view_specific_ or view_specific"
            )
        return view_specific[self.view_index]

    @staticmethod
    def _to_2d_tensor(values):
        tensor = values if isinstance(values, th.Tensor) else th.tensor(values)
        tensor = tensor.detach().clone().float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        return tensor

    @staticmethod
    def _make_fixed_projection(loadings: th.Tensor):
        layer = nn.Linear(loadings.shape[0], loadings.shape[1], bias=False)
        with th.no_grad():
            layer.weight.copy_(loadings.T)
        for param in layer.parameters():
            param.requires_grad = False
        return layer

    def set_active_head(self, head_name: str):
        if head_name not in self.head_names:
            raise ValueError(f"head_name must be one of {self.head_names}")
        self.active_head = head_name
        self.confmat_titles = [f"AJIVE {self.active_head} head confusion matrix"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def get_activation(self, name):
        """
        Hook function to capture activations from a layer.
        """

        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def get_gradient(self, name):
        """
        Hook function to capture gradients from a layer.
        """

        def hook(model, input, output):
            self.gradients[name] = output[0].detach()

        return hook

    def _extract_backbone_features(self, x):
        resnet = self.backbone.resnet
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)
        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        x = resnet.layer4(x)
        x = resnet.avgpool(x)
        x = th.flatten(x, 1)
        return x

    def _forward_head(self, features, head_name: str):
        projected = self.projections[head_name](features)
        return self.heads[head_name](projected)

    def forward(self, x, head_name: str | None = None):
        head_name = self.active_head if head_name is None else head_name
        if head_name not in self.head_names:
            raise ValueError(f"head_name must be one of {self.head_names}")
        features = self._extract_backbone_features(x)
        return self._forward_head(features, head_name)

    def _select_targets(self, batch):
        x, y1, y2 = batch
        y = y1 if self.task == 1 else y2
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._select_targets(batch)
        y_hat = self(x, head_name=self.active_head)
        metrics = self.compute_metrics(y_hat, y, prefix=f"train_{self.active_head}_")
        self.log_dict(metrics, sync_dist=True)
        return metrics[f"train_{self.active_head}_loss"]

    def validation_step(self, batch, batch_idx):
        x, y = self._select_targets(batch)
        y_hat = self(x, head_name=self.active_head)
        metrics = self.compute_metrics(y_hat, y, prefix=f"val_{self.active_head}_")
        self.log_dict(metrics, sync_dist=True)
        return metrics[f"val_{self.active_head}_loss"]

    def test_step(self, batch, batch_idx):
        x, y = self._select_targets(batch)
        y_hat = self(x, head_name=self.active_head)
        metrics = self.compute_metrics(y_hat, y, prefix=f"test_{self.active_head}_")
        self.log_dict(metrics, sync_dist=True)

        self.confusion_matrix[0].update(th.argmax(y_hat, dim=1), y)
        return metrics[f"test_{self.active_head}_loss"]

    def generate_grad_cam(self, x, target_class=None, head_name: str | None = None):
        """
        Generate Grad-CAM heatmap for the input image using one of the AJIVE heads.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            target_class: Target class for computing gradients. If None, uses predicted class.
            head_name: Which AJIVE head to use. Defaults to the active head.

        Returns:
            cam: Grad-CAM heatmap of shape (batch_size, height, width)
            target_class: Target class used for the backward pass
        """
        head_name = self.active_head if head_name is None else head_name
        if head_name not in self.head_names:
            raise ValueError(f"head_name must be one of {self.head_names}")

        self.eval()
        self.activations = {}
        self.gradients = {}

        target_layer = self.backbone.resnet.layer4[-1]
        forward_handle = target_layer.register_forward_hook(
            self.get_activation("layer4")
        )
        backward_handle = target_layer.register_full_backward_hook(
            self.get_gradient("layer4")
        )

        try:
            x = x.clone().detach().requires_grad_(True)
            output = self(x, head_name=head_name)

            if target_class is None:
                target_class = output.argmax(dim=1)
            elif not isinstance(target_class, th.Tensor):
                target_class = th.tensor(target_class, device=output.device)
            else:
                target_class = target_class.to(output.device)

            self.zero_grad(set_to_none=True)
            one_hot = th.zeros_like(output)
            one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
            output.backward(gradient=one_hot)

            gradients = self.gradients["layer4"]
            activations = self.activations["layer4"]
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1)
            cam = th.nn.functional.relu(cam)

            for i in range(cam.shape[0]):
                cam_min = cam[i].min()
                cam_max = cam[i].max()
                if cam_max > cam_min:
                    cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)

            return cam.detach().cpu(), target_class.detach().cpu()
        finally:
            forward_handle.remove()
            backward_handle.remove()
