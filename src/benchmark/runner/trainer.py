"""Training utilities for time-series forecasting models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import os

from benchmark.registry.losses import get_loss
from benchmark.utils.training import (
    CheckpointManager,
    EarlyStopping,
    adjust_learning_rate,
)


@dataclass
class TrainResult:
    """Summary of training outputs.

    Parameters
    ----------
    best_model_path : str
        Path to the best checkpoint saved during training.
    train_time_sec : float
        Total training time in seconds.
    """

    best_model_path: str
    train_time_sec: float


def _make_decoder_input(
    batch_y: torch.Tensor, label_len: int, pred_len: int, device: torch.device
) -> torch.Tensor:
    """Build the decoder input by concatenating label and zero padding.

    Parameters
    ----------
    batch_y : torch.Tensor
        Target series for the batch.
    label_len : int
        Number of past steps provided to the decoder.
    pred_len : int
        Number of future steps to predict.
    device : torch.device
        Device to place the decoder input on.

    Returns
    -------
    torch.Tensor
        Decoder input of shape (B, label_len + pred_len, C).
    """
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
    return dec_inp


def _call_model(model: nn.Module, batch_x, batch_x_mark, dec_inp, batch_y_mark):
    """Call model with or without temporal marks based on its signature.

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    batch_x : torch.Tensor
        Input sequence.
    batch_x_mark : torch.Tensor | None
        Time features for input sequence.
    dec_inp : torch.Tensor
        Decoder input sequence.
    batch_y_mark : torch.Tensor | None
        Time features for target sequence.

    Returns
    -------
    torch.Tensor
        Model outputs.
    """
    try:
        return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    except TypeError:
        return model(batch_x)


def train(
    model: nn.Module,
    train_loader,
    vali_loader,
    device: torch.device,
    epochs: int,
    patience: int,
    loss_name: str,
    loss_params: dict,
    optimizer: torch.optim.Optimizer,
    lradj: str,
    base_lr: float,
    total_epochs: int,
    label_len: int,
    pred_len: int,
    features: str,
    use_amp: bool,
    checkpoint_dir: str,
    checkpoint_cfg,
) -> TrainResult:
    """Train a model with early stopping and checkpointing.

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    train_loader : DataLoader
        Training data loader.
    vali_loader : DataLoader
        Validation data loader.
    device : torch.device
        Target device.
    epochs : int
        Number of epochs to train.
    patience : int
        Early stopping patience.
    loss_name : str
        Loss name.
    loss_params : dict
        Keyword arguments for loss construction.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    lradj : str
        Learning rate schedule type.
    base_lr : float
        Base learning rate for scheduling.
    total_epochs : int
        Total epochs used for scheduling.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    features : str
        Feature mode ("M", "S", "MS").
    use_amp : bool
        Whether to enable mixed precision.
    checkpoint_dir : str
        Directory to save checkpoints.
    checkpoint_cfg : TrainCheckpointConfig
        Checkpointing settings.

    Returns
    -------
    TrainResult
        Best checkpoint path and training time.
    """
    model.train()
    criterion = get_loss(loss_name, **loss_params)
    early_stopping = EarlyStopping(patience=patience)
    checkpoint_manager = CheckpointManager(
        strategy=checkpoint_cfg.strategy,
        save_k=checkpoint_cfg.save_k,
        path=checkpoint_dir,
    )
    use_amp = use_amp and device.type != "cpu"
    scaler = torch.amp.GradScaler() if use_amp else None

    start_time = time.perf_counter()
    for epoch in range(epochs):
        epoch_losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = _make_decoder_input(batch_y, label_len, pred_len, device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = _call_model(
                        model, batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )
                    outputs, batch_y_sliced = _slice_pred_target(
                        outputs, batch_y, pred_len, features
                    )
                    loss = criterion(outputs, batch_y_sliced)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = _call_model(
                    model, batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                outputs, batch_y_sliced = _slice_pred_target(
                    outputs, batch_y, pred_len, features
                )
                loss = criterion(outputs, batch_y_sliced)
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())

        vali_loss = validate(
            model, vali_loader, device, criterion, label_len, pred_len, features
        )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        current_lr = optimizer.param_groups[0].get("lr", base_lr)
        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss: {train_loss:.6f} | "
            f"val_loss: {vali_loss:.6f} | lr: {current_lr:.6g}"
        )
        is_best = early_stopping.step(vali_loss)
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save(model.state_dict(), best_path)
        checkpoint_manager.save(model, epoch + 1, vali_loss, is_best)
        if early_stopping.early_stop:
            break
        adjust_learning_rate(optimizer, epoch + 1, lradj, base_lr, total_epochs)

    train_time = time.perf_counter() - start_time
    best_model_path = f"{checkpoint_dir}/best_checkpoint.pth"
    model.load_state_dict(torch.load(best_model_path))
    return TrainResult(best_model_path=best_model_path, train_time_sec=train_time)


def validate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion: nn.Module,
    label_len: int,
    pred_len: int,
    features: str,
) -> float:
    """Evaluate validation loss for early stopping.

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    data_loader : DataLoader
        Validation data loader.
    device : torch.device
        Target device.
    criterion : nn.Module
        Loss function.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    features : str
        Feature mode ("M", "S", "MS").

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = _make_decoder_input(batch_y, label_len, pred_len, device)
            outputs = _call_model(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = _slice_pred_target(
                outputs, batch_y, pred_len, features
            )
            loss = criterion(outputs, batch_y_sliced)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _slice_pred_target(
    outputs: torch.Tensor, batch_y: torch.Tensor, pred_len: int, features: str
):
    """Slice prediction and target to the forecast horizon and feature mode.

    Parameters
    ----------
    outputs : torch.Tensor
        Raw model outputs.
    batch_y : torch.Tensor
        Ground-truth target sequences.
    pred_len : int
        Prediction horizon length.
    features : str
        Feature mode ("M", "S", "MS").

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Sliced outputs and targets.
    """
    f_dim = -1 if features == "MS" else 0
    outputs = outputs[:, -pred_len:, f_dim:]
    batch_y = batch_y[:, -pred_len:, f_dim:]
    return outputs, batch_y
