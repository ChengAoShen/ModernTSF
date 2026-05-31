"""Training utilities for time-series forecasting models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import os

from benchmark.runner.callbacks import Callback, CallbackContext
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


_AUX_LOSS_ATTRS = ("aux_loss", "last_moe_loss", "last_aux_loss")


def _collect_aux_loss(model: nn.Module) -> torch.Tensor | None:
    """Read an optional auxiliary loss exposed by the model, if any.

    Non-invasive convention: a model may stash an auxiliary regularization term
    (e.g. a MoE load-balancing loss) on itself as ``aux_loss`` (or, for
    compatibility, ``last_moe_loss`` / ``last_aux_loss``). When present and a
    finite scalar/tensor, the trainer adds it to the main training loss. Models
    that do not set any of these attributes are completely unaffected -- the
    return value is ``None`` and the train loop behaves as before.

    Parameters
    ----------
    model : nn.Module
        The model being trained (the underlying module if DataParallel-wrapped).

    Returns
    -------
    torch.Tensor | None
        A finite auxiliary loss tensor, or ``None`` when not applicable.
    """
    target = model.module if isinstance(model, nn.DataParallel) else model
    for attr in _AUX_LOSS_ATTRS:
        aux = getattr(target, attr, None)
        if aux is None:
            continue
        if not torch.is_tensor(aux):
            continue
        if aux.numel() != 1 or not torch.isfinite(aux):
            continue
        return aux
    return None


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
    callbacks: list[Callback] | None = None,
) -> TrainResult:
    """Train a model with early stopping and checkpointing.

    Pluggable callbacks
    -------------------
    An optional ``callbacks`` list (default empty) lets training tricks hook
    into the loop without changing default behavior. When ``callbacks`` is empty
    every hook is skipped and the loop is byte-identical to the pre-callback
    trainer. See ``benchmark.runner.callbacks`` for the hook contract.

    Auxiliary-loss convention
    -------------------------
    After each forward pass the trainer reads an optional auxiliary loss off the
    model via ``getattr(model, 'aux_loss', None)`` (also checking
    ``last_moe_loss`` / ``last_aux_loss`` for compatibility). When a finite
    scalar tensor is found it is added to the main loss; this is a strict no-op
    for models that never set those attributes. See ``_collect_aux_loss``.

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

    callbacks = callbacks or []
    # Fast path flag: when there are no callbacks, the loop must remain
    # byte-identical to the pre-callback trainer (zero_grad + step every batch,
    # no hook invocations, no extra tensor reslicing).
    has_callbacks = bool(callbacks)
    ctx = CallbackContext(
        model=model,
        optimizer=optimizer,
        total_epochs=epochs,
        pred_len=pred_len,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    start_time = time.perf_counter()
    for epoch in range(epochs):
        epoch_losses = []
        ctx.epoch = epoch
        n_batches = len(train_loader) if hasattr(train_loader, "__len__") else -1
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = _make_decoder_input(batch_y, label_len, pred_len, device)

            ctx.batch_idx = batch_idx
            ctx.is_last_batch = n_batches >= 0 and batch_idx == n_batches - 1

            if not has_callbacks:
                # --- Default path (unchanged behavior) ---
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
                        aux = _collect_aux_loss(model)
                        if aux is not None:
                            loss = loss + aux
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
                    aux = _collect_aux_loss(model)
                    if aux is not None:
                        loss = loss + aux
                    loss.backward()
                    optimizer.step()
                epoch_losses.append(loss.item())
                continue

            # --- Callback path ---
            do_step = _train_step_with_callbacks(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                use_amp=use_amp,
                device=device,
                batch_x=batch_x,
                batch_y=batch_y,
                batch_x_mark=batch_x_mark,
                batch_y_mark=batch_y_mark,
                dec_inp=dec_inp,
                pred_len=pred_len,
                features=features,
                callbacks=callbacks,
                ctx=ctx,
            )
            epoch_losses.append(ctx.extra["last_loss_value"])

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
        for cb in callbacks:
            cb.on_epoch_end(ctx)
        if early_stopping.early_stop:
            break
        adjust_learning_rate(optimizer, epoch + 1, lradj, base_lr, total_epochs)

    train_time = time.perf_counter() - start_time
    best_model_path = f"{checkpoint_dir}/best_checkpoint.pth"
    model.load_state_dict(torch.load(best_model_path))
    return TrainResult(best_model_path=best_model_path, train_time_sec=train_time)


def _run_compute_loss_hooks(callbacks, ctx, criterion):
    """Invoke on_compute_loss hooks, recomputing the loss if a hook resliced.

    A hook may either return a replacement loss (short-circuit) or mutate
    ``ctx.outputs`` / ``ctx.targets`` in place and return None. In the latter
    case the loss is recomputed from the (possibly resliced) tensors after all
    hooks run. ``ctx.loss`` is updated in place.
    """
    resliced = False
    initial_outputs = ctx.outputs
    initial_targets = ctx.targets
    for cb in callbacks:
        returned = cb.on_compute_loss(ctx)
        if returned is not None:
            ctx.loss = returned
        if ctx.outputs is not initial_outputs or ctx.targets is not initial_targets:
            resliced = True
            initial_outputs = ctx.outputs
            initial_targets = ctx.targets
    if resliced:
        ctx.loss = criterion(ctx.outputs, ctx.targets)


def _train_step_with_callbacks(
    *,
    model,
    optimizer,
    criterion,
    scaler,
    use_amp,
    device,
    batch_x,
    batch_y,
    batch_x_mark,
    batch_y_mark,
    dec_inp,
    pred_len,
    features,
    callbacks,
    ctx: CallbackContext,
) -> bool:
    """Run a single training micro-batch with callback hooks.

    Returns whether an optimizer step was performed. Stores the (unscaled)
    base-loss value for logging in ``ctx.extra['last_loss_value']``.

    The optimizer step / zero_grad coordination follows the gradient-accumulation
    contract: when a callback's ``on_optimizer_step`` returns False the step and
    the subsequent ``zero_grad`` are both skipped so gradients accumulate.
    """
    if use_amp:
        with torch.amp.autocast(device_type=device.type):
            outputs = _call_model(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = _slice_pred_target(
                outputs, batch_y, pred_len, features
            )
            ctx.outputs = outputs
            ctx.targets = batch_y_sliced
            base_loss = criterion(outputs, batch_y_sliced)
            aux = _collect_aux_loss(model)
            if aux is not None:
                base_loss = base_loss + aux
            ctx.loss = base_loss
            ctx.extra["last_loss_value"] = base_loss.item()
            _run_compute_loss_hooks(callbacks, ctx, criterion)
        scaler.scale(ctx.loss).backward()
        scaler.unscale_(optimizer)
        for cb in callbacks:
            cb.on_backward(ctx)
        do_step = _resolve_step(callbacks, ctx)
        if do_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # Keep the scaler's internal state consistent without stepping.
            scaler.update()
    else:
        outputs = _call_model(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs, batch_y_sliced = _slice_pred_target(
            outputs, batch_y, pred_len, features
        )
        ctx.outputs = outputs
        ctx.targets = batch_y_sliced
        base_loss = criterion(outputs, batch_y_sliced)
        aux = _collect_aux_loss(model)
        if aux is not None:
            base_loss = base_loss + aux
        ctx.loss = base_loss
        ctx.extra["last_loss_value"] = base_loss.item()
        _run_compute_loss_hooks(callbacks, ctx, criterion)
        ctx.loss.backward()
        for cb in callbacks:
            cb.on_backward(ctx)
        do_step = _resolve_step(callbacks, ctx)
        if do_step:
            optimizer.step()
            optimizer.zero_grad()
    return do_step


def _resolve_step(callbacks, ctx: CallbackContext) -> bool:
    """Resolve whether to step the optimizer this micro-batch.

    A step is performed unless any callback's ``on_optimizer_step`` returns
    False (gradient accumulation). Hooks returning None/True do not veto.
    """
    do_step = True
    for cb in callbacks:
        decision = cb.on_optimizer_step(ctx)
        if decision is False:
            do_step = False
    return do_step


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
