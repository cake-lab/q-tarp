"""Reusable trainer for the k-fold QET predictor."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from standard_metrics import regression_metrics


class Trainer:
    """Reusable trainer for PyTorch/PyG graph regression models."""

    def __init__(self, model, criterion, optimizer, scheduler, device, output_dir, writer=None, debug_trace=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.best_valid_mse = float("inf")
        self.best_model_state = None
        self.history = {"train": [], "valid": []}
        self.debug_trace = debug_trace
        self._debug_logged_train_batch = False
        self._debug_logged_eval_batch = False
        self._writer_disabled_reason = None

    def _tag(self, name):
        return name.replace(".", "/")

    def _disable_writer(self, reason):
        if self.writer is None:
            return
        self._writer_disabled_reason = str(reason)
        print(
            f"[trainer] disabling tensorboard writer due to logging failure: {reason}",
            flush=True,
        )
        try:
            self.writer.close()
        except Exception:
            pass
        self.writer = None

    def _safe_add_scalar(self, tag, value, step):
        if self.writer is None:
            return
        try:
            self.writer.add_scalar(tag, value, step)
        except (BlockingIOError, OSError, RuntimeError, ValueError) as exc:
            self._disable_writer(exc)

    def _safe_flush_writer(self):
        if self.writer is None:
            return
        try:
            self.writer.flush()
        except (BlockingIOError, OSError, RuntimeError, ValueError) as exc:
            self._disable_writer(exc)

    def close_writer(self):
        if self.writer is None:
            return
        try:
            self.writer.close()
        except (BlockingIOError, OSError, RuntimeError, ValueError):
            pass
        self.writer = None

    def _log_parameter_norms(self, step):
        if self.writer is None:
            return
        for name, parameter in self.model.named_parameters():
            self._safe_add_scalar(
                f"param_norm/{self._tag(name)}",
                parameter.detach().norm().item(),
                step,
            )

    def _log_gradient_norms(self, grad_norm_sums, grad_norm_counts, step):
        if self.writer is None:
            return
        for name, grad_sum in grad_norm_sums.items():
            count = grad_norm_counts.get(name, 0)
            if count == 0:
                continue
            self._safe_add_scalar(
                f"grad_norm/{self._tag(name)}",
                grad_sum / count,
                step,
            )

    def train_one_epoch(self, loader):
        self.model.train()
        all_pred = []
        all_target = []
        grad_norm_sums = {}
        grad_norm_counts = {}
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(self.device)
            if self.debug_trace and (batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader)):
                print(
                    f"[trainer] train batch {batch_idx}/{len(loader)} "
                    f"x={tuple(batch.x.shape)} edge_index={tuple(batch.edge_index.shape)} "
                    f"global_features={tuple(batch.global_features.shape)} y={tuple(batch.y.shape)}",
                    flush=True,
                )
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(batch)
            target = batch.y.float().view_as(pred)
            if self.debug_trace and not self._debug_logged_train_batch:
                print(
                    f"[trainer] first train batch target preview={target.detach().cpu()[:8].tolist()}",
                    flush=True,
                )
                print(
                    f"[trainer] first train batch pred preview={pred.detach().cpu()[:8].tolist()}",
                    flush=True,
                )
                self._debug_logged_train_batch = True
            loss = self.criterion(pred, target)
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                grad_norm_sums[name] = grad_norm_sums.get(name, 0.0) + parameter.grad.detach().norm().item()
                grad_norm_counts[name] = grad_norm_counts.get(name, 0) + 1
            self.optimizer.step()
            all_pred.append(pred.detach().cpu())
            all_target.append(target.detach().cpu())
        pred = torch.cat(all_pred, dim=0) if all_pred else torch.empty(0)
        target = torch.cat(all_target, dim=0) if all_target else torch.empty(0)
        metrics = regression_metrics(target, pred)
        metrics["pred"] = pred.numpy()
        metrics["target"] = target.numpy()
        return metrics, grad_norm_sums, grad_norm_counts

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_pred = []
        all_target = []
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(self.device)
            if self.debug_trace and (batch_idx == 1 or batch_idx == len(loader)):
                print(
                    f"[trainer] eval batch {batch_idx}/{len(loader)} "
                    f"x={tuple(batch.x.shape)} edge_index={tuple(batch.edge_index.shape)} "
                    f"global_features={tuple(batch.global_features.shape)} y={tuple(batch.y.shape)}",
                    flush=True,
                )
            pred = self.model(batch)
            target = batch.y.float().view_as(pred)
            if self.debug_trace and not self._debug_logged_eval_batch:
                print(
                    f"[trainer] first eval batch target preview={target.detach().cpu()[:8].tolist()}",
                    flush=True,
                )
                print(
                    f"[trainer] first eval batch pred preview={pred.detach().cpu()[:8].tolist()}",
                    flush=True,
                )
                self._debug_logged_eval_batch = True
            all_pred.append(pred.detach().cpu())
            all_target.append(target.detach().cpu())
        pred = torch.cat(all_pred, dim=0) if all_pred else torch.empty(0)
        target = torch.cat(all_target, dim=0) if all_target else torch.empty(0)
        metrics = regression_metrics(target, pred)
        metrics["pred"] = pred.numpy()
        metrics["target"] = target.numpy()
        return metrics

    def save_checkpoint(self, epoch, valid_metrics, is_best):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "valid_metrics": {k: v for k, v in valid_metrics.items() if k not in ["pred", "target"]},
            "best_valid_mse": self.best_valid_mse,
        }
        torch.save(checkpoint, self.output_dir / "last_checkpoint.pth")
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_checkpoint.pth")
            torch.save(self.model.state_dict(), self.output_dir / "model.pth")

    def save_train_only_checkpoint(self, epoch, train_metrics):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "train_metrics": {k: v for k, v in train_metrics.items() if k not in ["pred", "target"]},
        }
        torch.save(checkpoint, self.output_dir / "last_checkpoint.pth")

    def fit(self, train_loader, valid_loader, num_epochs):
        for epoch in range(int(num_epochs)):
            print(f"[trainer] epoch {epoch + 1}/{int(num_epochs)} start", flush=True)
            train_metrics, grad_norm_sums, grad_norm_counts = self.train_one_epoch(train_loader)
            valid_metrics = self.evaluate(valid_loader)
            if self.scheduler is not None:
                self.scheduler.step()
            is_best = valid_metrics["mse"] < self.best_valid_mse
            if is_best:
                self.best_valid_mse = valid_metrics["mse"]
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.save_checkpoint(epoch, valid_metrics, is_best=is_best)
            train_record = {k: v for k, v in train_metrics.items() if k not in ["pred", "target"]}
            train_record["epoch"] = epoch
            valid_record = {k: v for k, v in valid_metrics.items() if k not in ["pred", "target"]}
            valid_record["epoch"] = epoch
            valid_record["best_mse"] = self.best_valid_mse
            self.history["train"].append(train_record)
            self.history["valid"].append(valid_record)
            print(
                f"[trainer] epoch {epoch + 1}/{int(num_epochs)} "
                f"train_rmse={train_metrics['rmse']:.6f} "
                f"valid_rmse={valid_metrics['rmse']:.6f} "
                f"valid_r2={valid_metrics['r2']:.6f} "
                f"best_mse={self.best_valid_mse:.6f}",
                flush=True,
            )
            if self.writer is not None:
                self._safe_add_scalar("train/mse", train_metrics["mse"], epoch)
                self._safe_add_scalar("train/rmse", train_metrics["rmse"], epoch)
                self._safe_add_scalar("train/nmse", train_metrics["nmse"], epoch)
                self._safe_add_scalar("train/r2", train_metrics["r2"], epoch)
                self._safe_add_scalar("valid/mse", valid_metrics["mse"], epoch)
                self._safe_add_scalar("valid/rmse", valid_metrics["rmse"], epoch)
                self._safe_add_scalar("valid/nmse", valid_metrics["nmse"], epoch)
                self._safe_add_scalar("valid/r2", valid_metrics["r2"], epoch)
                self._safe_add_scalar("valid/best_mse", self.best_valid_mse, epoch)
                self._safe_add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
                self._log_parameter_norms(epoch)
                self._log_gradient_norms(grad_norm_sums, grad_norm_counts, epoch)
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        with open(self.output_dir / "history.json", "w") as file:
            json.dump(self.history, file, indent=2)
        self._safe_flush_writer()
        return self.history

    def fit_fixed_epochs(self, train_loader, num_epochs):
        """Train for a fixed number of epochs without a validation split."""
        self.history = {"train": [], "valid": []}
        for epoch in range(int(num_epochs)):
            print(f"[trainer] final epoch {epoch + 1}/{int(num_epochs)} start", flush=True)
            train_metrics, grad_norm_sums, grad_norm_counts = self.train_one_epoch(train_loader)
            if self.scheduler is not None:
                self.scheduler.step()
            self.save_train_only_checkpoint(epoch, train_metrics)
            train_record = {k: v for k, v in train_metrics.items() if k not in ["pred", "target"]}
            train_record["epoch"] = epoch
            self.history["train"].append(train_record)
            print(
                f"[trainer] final epoch {epoch + 1}/{int(num_epochs)} "
                f"train_rmse={train_metrics['rmse']:.6f} train_r2={train_metrics['r2']:.6f}",
                flush=True,
            )
            if self.writer is not None:
                self._safe_add_scalar("train/mse", train_metrics["mse"], epoch)
                self._safe_add_scalar("train/rmse", train_metrics["rmse"], epoch)
                self._safe_add_scalar("train/nmse", train_metrics["nmse"], epoch)
                self._safe_add_scalar("train/r2", train_metrics["r2"], epoch)
                self._safe_add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
                self._log_parameter_norms(epoch)
                self._log_gradient_norms(grad_norm_sums, grad_norm_counts, epoch)

        torch.save(self.model.state_dict(), self.output_dir / "model.pth")
        with open(self.output_dir / "history.json", "w") as file:
            json.dump(self.history, file, indent=2)
        self._safe_flush_writer()
        return self.history

    def test_and_save(self, test_loader):
        print("[trainer] starting final test evaluation", flush=True)
        self._debug_logged_eval_batch = False
        test_metrics = self.evaluate(test_loader)
        print(
            f"[trainer] test_rmse={test_metrics['rmse']:.6f} "
            f"test_r2={test_metrics['r2']:.6f} test_mse={test_metrics['mse']:.6f}",
            flush=True,
        )
        if self.writer is not None:
            self._safe_add_scalar("test/mse", test_metrics["mse"], 0)
            self._safe_add_scalar("test/rmse", test_metrics["rmse"], 0)
            self._safe_add_scalar("test/nmse", test_metrics["nmse"], 0)
            self._safe_add_scalar("test/r2", test_metrics["r2"], 0)
            self._safe_add_scalar("test/error", test_metrics["mse"], 0)
        np.savez(
            self.output_dir / "test_predictions.npz",
            pred=test_metrics["pred"],
            target=test_metrics["target"],
        )
        safe_metrics = {k: v for k, v in test_metrics.items() if k not in ["pred", "target"]}
        with open(self.output_dir / "test_metrics.json", "w") as file:
            json.dump(safe_metrics, file, indent=2)
        return test_metrics
