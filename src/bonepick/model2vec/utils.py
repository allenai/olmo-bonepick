from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from model2vec.train import StaticModelForClassification as _StaticModelForClassification
from model2vec.train.base import FinetunableStaticModel, TextDataset
from model2vec.train.classifier import LabelType
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch import nn
from tqdm import trange

from bonepick.data.tokenizers import parallel_tokenize
from bonepick.logger import LOGGER

_RANDOM_SEED = 42


class StaticModelForClassification(_StaticModelForClassification):
    def _prepare_dataset(
        self,
        X: list[str],
        y: "LabelType",
        max_length: int = 512,
        num_proc: int | None = None,
        max_chunk_size: int = 20_000,
    ) -> "TextDataset":
        """
        Prepare a dataset. For multilabel classification, each target is converted into a multi-hot vector.

        :param X: The texts.
        :param y: The labels.
        :param max_length: The maximum length of the input.
        :return: A TextDataset.
        """
        # This is a speed optimization.
        # assumes a mean token length of 10, which is really high, so safe.
        tokenized = parallel_tokenize(
            texts=X,
            tokenizer=self.tokenizer,
            max_length=max_length,
            num_proc=num_proc,
            max_chunk_size=max_chunk_size,
        )
        if self.multilabel:
            # Convert labels to multi-hot vectors
            num_classes = len(self.classes_)
            labels_tensor = torch.zeros(len(y), num_classes, dtype=torch.float)
            mapping = {label: idx for idx, label in enumerate(self.classes_)}
            for i, sample_labels in enumerate(y):
                indices = [mapping[label] for label in sample_labels]
                labels_tensor[i, indices] = 1.0
        else:
            labels_tensor = torch.tensor(
                [self.classes_.index(label) for label in cast(list[str], y)],
                dtype=torch.long,
            )
        return TextDataset(tokenized, labels_tensor)


class StaticModelForRegression(FinetunableStaticModel):
    """A regression model based on static embeddings."""

    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        n_layers: int = 1,
        hidden_dim: int = 512,
        pad_id: int = 0,
        token_mapping: list[int] | None = None,
        weights: torch.Tensor | None = None,
        freeze: bool = False,
        out_dim: int | None = None,  # Ignored, always 1 for regression
    ) -> None:
        """Initialize a regression model.

        :param vectors: The embeddings.
        :param tokenizer: The tokenizer.
        :param n_layers: Number of hidden layers.
        :param hidden_dim: Hidden dimension size.
        :param pad_id: The padding token ID.
        :param token_mapping: Token mapping for the embeddings.
        :param weights: Pre-trained token weights.
        :param freeze: Whether to freeze embeddings during training.
        :param out_dim: Ignored (always 1 for regression).
        """
        del out_dim  # Always 1 for regression
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        super().__init__(
            vectors=vectors,
            out_dim=1,  # Single output for regression
            pad_id=pad_id,
            tokenizer=tokenizer,
            token_mapping=token_mapping,
            weights=weights,
            freeze=freeze,
        )

    def construct_head(self) -> nn.Sequential:
        """Construct a regression head."""
        if self.n_layers == 0:
            return nn.Sequential(nn.Linear(self.embed_dim, self.out_dim))
        modules: list[nn.Module] = [
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(self.n_layers - 1):
            modules.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        modules.append(nn.Linear(self.hidden_dim, self.out_dim))

        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        return nn.Sequential(*modules)

    def predict(self, X: list[str], show_progress_bar: bool = False, batch_size: int = 1024) -> np.ndarray:
        """Predict regression values for texts.

        :param X: The texts to predict on.
        :param show_progress_bar: Whether to show a progress bar.
        :param batch_size: The batch size.
        :return: The predictions as a 1D numpy array.
        """
        pred = []
        for batch_start in trange(0, len(X), batch_size, disable=not show_progress_bar):
            logits = self._predict_single_batch(X[batch_start : batch_start + batch_size])
            pred.append(logits.squeeze(-1).cpu().numpy())
        return np.concatenate(pred, axis=0)

    @torch.no_grad()
    def _predict_single_batch(self, X: list[str]) -> torch.Tensor:
        input_ids = self.tokenize(X)
        vectors, _ = self.forward(input_ids)
        return vectors

    def fit(
        self,
        X: list[str],
        y: list[float] | np.ndarray,
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
        X_val: list[str] | None = None,
        y_val: list[float] | np.ndarray | None = None,
    ) -> "StaticModelForRegression":
        """Fit the regression model.

        :param X: The texts to train on.
        :param y: The target values.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size. If None, chosen automatically.
        :param min_epochs: The minimum number of epochs.
        :param max_epochs: The maximum number of epochs. -1 means train until early stopping.
        :param early_stopping_patience: Patience for early stopping. None disables it.
        :param test_size: Test size for train-validation split.
        :param device: Device to train on.
        :param X_val: Validation texts.
        :param y_val: Validation targets.
        :return: The fitted model.
        """
        pl.seed_everything(_RANDOM_SEED)
        LOGGER.info("Re-initializing model.")

        # Re-initialize the head
        self.head = self.construct_head()
        self.embeddings = nn.Embedding.from_pretrained(
            self.vectors.clone(), freeze=self.freeze, padding_idx=self.pad_id
        )
        self.w = self.construct_weights()
        self.train()

        if (X_val is not None) != (y_val is not None):
            raise ValueError("Both X_val and y_val must be provided together, or neither.")

        if X_val is not None and y_val is not None:
            train_texts = X
            train_targets = y
            validation_texts = X_val
            validation_targets = y_val
        else:
            train_texts, validation_texts, train_targets, validation_targets = train_test_split(
                X, y, test_size=test_size, random_state=_RANDOM_SEED, shuffle=True
            )

        if batch_size is None:
            base_number = int(min(max(1, (len(train_texts) / 30) // 32), 16))
            batch_size = int(base_number * 32)
            LOGGER.info("Batch size automatically set to %d.", batch_size)

        LOGGER.info("Preparing train dataset.")
        train_dataset = self._prepare_dataset(train_texts, train_targets)
        LOGGER.info("Preparing validation dataset.")
        val_dataset = self._prepare_dataset(validation_texts, validation_targets)

        lightning_module = _RegressionLightningModule(self, learning_rate=learning_rate)

        n_train_batches = len(train_dataset) // batch_size
        callbacks: list[Callback] = []
        if early_stopping_patience is not None:
            callback = EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience)
            callbacks.append(callback)

        if n_train_batches < 250:
            val_check_interval = None
            check_val_every_epoch = 1
        else:
            val_check_interval = max(250, 2 * len(val_dataset) // batch_size)
            check_val_every_epoch = None

        with TemporaryDirectory() as tempdir:
            trainer = pl.Trainer(
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                callbacks=callbacks,
                val_check_interval=val_check_interval,
                check_val_every_n_epoch=check_val_every_epoch,
                accelerator=device,
                default_root_dir=tempdir,
            )

            trainer.fit(
                lightning_module,
                train_dataloaders=train_dataset.to_dataloader(shuffle=True, batch_size=batch_size),
                val_dataloaders=val_dataset.to_dataloader(shuffle=False, batch_size=batch_size),
            )
            best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            best_model_weights = torch.load(best_model_path, weights_only=True)

        state_dict = {}
        for weight_name, weight in best_model_weights["state_dict"].items():
            if "loss_function" in weight_name:
                continue
            state_dict[weight_name.removeprefix("model.")] = weight

        self.load_state_dict(state_dict)
        self.eval()
        return self

    def _prepare_dataset(
        self,
        X: list[str],
        y: list[float] | np.ndarray,
        max_length: int = 512,
        num_proc: int | None = None,
        max_chunk_size: int = 20_000,
    ) -> TextDataset:
        """Prepare a dataset for regression.

        :param X: The texts.
        :param y: The target values.
        :param max_length: Maximum token length.
        :return: A TextDataset.
        """
        tokenized = parallel_tokenize(
            texts=X,
            tokenizer=self.tokenizer,
            max_length=max_length,
            num_proc=num_proc,
            max_chunk_size=max_chunk_size,
        )
        targets_tensor = torch.tensor(y, dtype=torch.float32)

        return TextDataset(tokenized, targets_tensor)

    def evaluate(self, X: list[str], y: list[float] | np.ndarray, batch_size: int = 1024) -> dict[str, float]:
        """Evaluate the regression model.

        :param X: The texts.
        :param y: The ground truth values.
        :param batch_size: The batch size.
        :return: Dictionary with MSE, RMSE, MAE, and R² metrics.
        """
        self.eval()
        predictions = self.predict(X, show_progress_bar=True, batch_size=batch_size)
        y_array = np.array(y)

        mse = float(np.mean((predictions - y_array) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(predictions - y_array)))

        ss_res = np.sum((y_array - predictions) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


class _RegressionLightningModule(pl.LightningModule):
    """Lightning module for regression training."""

    def __init__(self, model: StaticModelForRegression, learning_rate: float) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out.squeeze(-1), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        head_out, _ = self.model(x)
        predictions = head_out.squeeze(-1)
        loss = self.loss_function(predictions, y)

        # Compute R² for logging
        ss_res = torch.sum((y - predictions) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        self.log("val_loss", loss)
        self.log("val_r2", r2, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            threshold=0.03,
            threshold_mode="rel",
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def monkey_patch_get_latest_model_path():
    from model2vec import hf_utils

    old_get_latest_model_path = hf_utils._get_latest_model_path

    def new_get_latest_model_path(model_id: str) -> Path | None:
        # get around the fact that the cache name might be too long for some filesystems
        if Path(model_id).exists() and Path(model_id).is_dir():
            return Path(model_id)
        return old_get_latest_model_path(model_id)

    hf_utils._get_latest_model_path = new_get_latest_model_path


monkey_patch_get_latest_model_path()
