import argparse
import copy
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.optimize import brentq, linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


class Context:
    """One experiment environment shared by all comparison methods."""

    def __init__(
        self,
        seed=None,
        task=None,
        partitions=None,
        test_data=None,
        counts=None,
        num_users=None,
        model_factory=None,
        initial_model_state=None,
        model_bits=None,
        num_rbs=None,
        distances=None,
        channel_gain=None,
        downlink_gain=None,
        interference=None,
        downlink_interference=None,
        powers=None,
        packet_errors=None,
        uplink_rates=None,
        downlink_rates=None,
        total_delays=None,
        energies=None,
        feasible=None,
        delay_s=None,
        energy_j=None,
        pmax_w=None,
        rounds=None,
        local_epochs=None,
        learning_rate=None,
        batch_size=None,
        device=None,
        loss=None,
        optimizer_states=None,
    ):
        self.seed = seed
        self.task = task

        self.partitions = partitions
        self.test_data = test_data
        self.counts = counts
        self.num_users = num_users

        self.model_factory = model_factory
        self.initial_model_state = initial_model_state
        self.model_bits = model_bits

        self.num_rbs = num_rbs
        self.distances = distances
        self.channel_gain = channel_gain
        self.downlink_gain = downlink_gain
        self.interference = interference
        self.downlink_interference = downlink_interference
        self.powers = powers
        self.packet_errors = packet_errors
        self.uplink_rates = uplink_rates
        self.downlink_rates = downlink_rates
        self.total_delays = total_delays
        self.energies = energies
        self.feasible = feasible

        self.delay_s = delay_s
        self.energy_j = energy_j
        self.pmax_w = pmax_w

        self.rounds = rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.optimizer_states = optimizer_states


# data processing
def generate_synthetic_data(num_users=15, samples_per_user=30, seed=42, noise_std=0.4):
    """Generate the linear-regression data used in the paper: y=-2x+1+0.4N(0,1)."""
    rng = np.random.default_rng(seed)
    # Accept either a scalar per-user count or an explicit Ki vector.
    if isinstance(samples_per_user, int):
        counts = [samples_per_user] * num_users
    else:
        counts = list(samples_per_user)
        num_users = len(counts)

    users = []
    all_x = []
    all_y = []
    # Each user receives its own TensorDataset, matching the FL locality assumption.
    for count in counts:
        x = rng.random((int(count), 1), dtype=np.float32)
        y = -2.0 * x + 1.0 + noise_std * rng.standard_normal((int(count), 1)).astype(np.float32)
        users.append(TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))))
        all_x.append(x)
        all_y.append(y.astype(np.float32))

    return {
        "users": users,
        "x": torch.from_numpy(np.vstack(all_x)),
        "y": torch.from_numpy(np.vstack(all_y)),
        "counts": counts,
        "task": "regression",
    }


def load_mnist_data(num_users=15, samples_per_user=200, test_samples=10000, seed=42, data_dir=None, download=False, train_order="shuffled", partition_order="shuffled"):
    """Load MNIST from a local npz/cache first, with torchvision download as an opt-in fallback."""
    rng = np.random.default_rng(seed)
    candidates = []
    # Prefer user-provided/local caches because the reproduction should not depend on downloads.
    if data_dir is not None:
        data_path = Path(data_dir)
        candidates.append(data_path if data_path.suffix == ".npz" else data_path / "mnist.npz")
    candidates.extend([Path("mnist.npz"), Path("data/mnist.npz"), Path("/home/imes-server6/dataset/mnist.npz")])

    arrays = None
    for candidate in candidates:
        if candidate.exists():
            loaded = np.load(candidate)
            arrays = {
                "x_train": loaded["x_train"],
                "y_train": loaded["y_train"],
                "x_test": loaded["x_test"],
                "y_test": loaded["y_test"],
            }
            break

    if arrays is None:
        # Torchvision is used only when explicitly allowed to download or already cached.
        try:
            from torchvision import datasets, transforms

            root = str(Path(data_dir or os.environ.get("TORCH_HOME", "/tmp")) / "mnist")
            transform = transforms.ToTensor()
            train = datasets.MNIST(root=root, train=True, transform=transform, download=download)
            test = datasets.MNIST(root=root, train=False, transform=transform, download=download)
            arrays = {
                "x_train": train.data.numpy(),
                "y_train": train.targets.numpy(),
                "x_test": test.data.numpy(),
                "y_test": test.targets.numpy(),
            }
        except Exception:
            # Last-resort fallback keeps code runnable without MNIST, but it is not the paper dataset.
            from sklearn.datasets import load_digits

            digits = load_digits()
            images = digits.images.astype(np.float32)
            padded = np.zeros((images.shape[0], 28, 28), dtype=np.float32)
            for idx, image in enumerate(images):
                enlarged = np.kron(image / 16.0, np.ones((3, 3), dtype=np.float32))
                padded[idx, 2:26, 2:26] = enlarged
            order = rng.permutation(len(padded))
            split = int(0.8 * len(order))
            arrays = {
                "x_train": padded[order[:split]],
                "y_train": digits.target[order[:split]],
                "x_test": padded[order[split:]],
                "y_test": digits.target[order[split:]],
            }

    x_train = arrays["x_train"].astype(np.float32)
    x_test = arrays["x_test"].astype(np.float32)
    if x_train.max() > 1.0:
        x_train /= 255.0
    if x_test.max() > 1.0:
        x_test /= 255.0
    x_train = torch.from_numpy(x_train).reshape(-1, 1, 28, 28)
    y_train = torch.from_numpy(arrays["y_train"].astype(np.int64))
    x_test = torch.from_numpy(x_test).reshape(-1, 1, 28, 28)
    y_test = torch.from_numpy(arrays["y_test"].astype(np.int64))

    if isinstance(samples_per_user, int):
        counts = [int(samples_per_user)] * num_users
    else:
        counts = list(samples_per_user)
        if len(counts) < num_users:
            counts = (counts * ((num_users + len(counts) - 1) // len(counts)))[:num_users]
        else:
            counts = counts[:num_users]
    total_train = min(len(x_train), int(sum(counts)))
    train_order = str(train_order).lower()
    if train_order == "shuffled":
        train_indices = rng.permutation(len(x_train))[:total_train]
    elif train_order == "file_order":
        train_indices = np.arange(len(x_train))[:total_train]
    else:
        raise ValueError("train_order must be 'shuffled' or 'file_order'")
    # Test data is sampled independently from user partitions.
    test_count = min(len(x_test), int(test_samples))
    test_indices = rng.permutation(len(x_test))[:test_count]
    train_data = TensorDataset(x_train[train_indices], y_train[train_indices])
    test_data = TensorDataset(x_test[test_indices], y_test[test_indices])
    partition_order = str(partition_order).lower()
    if partition_order not in {"shuffled", "file_order", "label_sorted"}:
        raise ValueError("partition_order must be 'shuffled', 'file_order', or 'label_sorted'")
    users = get_partitioned_data(
        train_data,
        num_users=num_users,
        samples_per_user=counts,
        seed=seed,
        non_iid=partition_order == "label_sorted",
        shuffle=partition_order == "shuffled",
    )
    # The return shape is shared by figure runners and FL algorithms.
    return {"train": train_data, "test": test_data, "users": users, "task": "mnist"}


def get_partitioned_data(data=None, num_users=15, samples_per_user=None, seed=42, non_iid=False, shuffle=True):
    """Partition a TensorDataset into user-local datasets for FL."""
    # A missing dataset means the caller wants synthetic regression users.
    if data is None:
        return generate_synthetic_data(num_users=num_users, samples_per_user=samples_per_user or 30, seed=seed)["users"]
    if isinstance(data, dict) and "users" in data and samples_per_user is None:
        return data["users"]
    if isinstance(data, dict) and "train" in data:
        data = data["train"]

    rng = np.random.default_rng(seed)
    dataset_len = len(data)
    if isinstance(samples_per_user, int):
        counts = [samples_per_user] * num_users
    elif samples_per_user is None:
        base = dataset_len // num_users
        counts = [base] * num_users
    else:
        counts = list(samples_per_user)
        num_users = len(counts)
    # Clamp bad counts so every partition operation remains index-safe.
    counts = [max(0, min(int(count), dataset_len)) for count in counts]

    if non_iid and hasattr(data, "tensors") and len(data.tensors) > 1:
        # Label sorting provides a simple non-IID partition option for sweeps.
        labels = data.tensors[1].detach().cpu().numpy()
        indices = np.argsort(labels)
    elif shuffle:
        indices = rng.permutation(dataset_len)
    else:
        indices = np.arange(dataset_len)

    users = []
    cursor = 0
    for count in counts:
        selected = indices[cursor:cursor + count]
        cursor += count
        if len(selected) == 0:
            # Empty local datasets break DataLoader and FedAvg weighting.
            selected = indices[:1]
        users.append(Subset(data, selected.tolist()))
    return users


# models
class RegressionFNN(nn.Module):
    def __init__(self, hidden_size=20, activation="tanh"):
        super().__init__()
        # The paper uses a 20-neuron FNN for linear regression.
        if activation == "tanh":
            nonlinear = nn.Tanh()
        elif activation == "relu":
            nonlinear = nn.ReLU()
        elif activation == "sigmoid":
            nonlinear = nn.Sigmoid()
        elif activation == "identity":
            nonlinear = nn.Identity()
        else:
            raise ValueError(f"Unsupported regression activation: {activation}")
        self.net = nn.Sequential(nn.Linear(1, hidden_size), nonlinear, nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class MNISTFNN(nn.Module):
    def __init__(self, hidden_size=50, activation="relu"):
        super().__init__()
        # The paper uses a single hidden layer with 50 neurons for MNIST.
        if activation == "relu":
            nonlinear = nn.ReLU()
        elif activation == "tanh":
            nonlinear = nn.Tanh()
        elif activation == "sigmoid":
            nonlinear = nn.Sigmoid()
        else:
            raise ValueError("Unsupported MNIST activation")
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden_size), nonlinear, nn.Linear(hidden_size, 10))

    def forward(self, x):
        return self.net(x)


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Fig. 10 uses CNNs to validate that the wireless-aware selection still helps.
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 50), nn.ReLU(), nn.Linear(50, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class TrainSCG:
    """Scaled conjugate-gradient optimizer compatible with MATLAB trainscg-style local training."""

    def __init__(self, parameters, sigma, lambd, lambda_min, lambda_max, min_grad):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        if not self.parameters:
            raise ValueError("TrainSCG requires at least one trainable parameter")
        self.sigma = float(sigma)
        self.lambd = float(lambd)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.min_grad = float(min_grad)
        if self.sigma <= 0.0:
            raise ValueError("training.trainscg_sigma must be positive")
        if self.lambd <= 0.0:
            raise ValueError("training.trainscg_lambda must be positive")
        if self.lambda_min <= 0.0:
            raise ValueError("training.trainscg_lambda_min must be positive")
        if self.lambda_max < self.lambda_min:
            raise ValueError("training.trainscg_lambda_max must be >= training.trainscg_lambda_min")
        if self.min_grad < 0.0:
            raise ValueError("training.trainscg_min_grad must be non-negative")
        self.direction = None
        self.previous_gradient = None
        self.last_step = {}

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def _parameters_to_vector(self):
        return torch.cat([parameter.detach().reshape(-1) for parameter in self.parameters])

    def _gradients_to_vector(self):
        gradients = []
        for parameter in self.parameters:
            if parameter.grad is None:
                gradients.append(torch.zeros_like(parameter.detach()).reshape(-1))
            else:
                gradients.append(parameter.grad.detach().reshape(-1))
        return torch.cat(gradients)

    def _set_parameters_from_vector(self, vector):
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters:
                numel = parameter.numel()
                parameter.copy_(vector[offset:offset + numel].view_as(parameter))
                offset += numel

    def _evaluate(self, closure):
        with torch.enable_grad():
            loss = closure()
        if not torch.is_tensor(loss):
            raise ValueError("TrainSCG closure must return a torch.Tensor loss")
        return loss, self._gradients_to_vector()

    def step(self, closure):
        if closure is None:
            raise ValueError("TrainSCG requires a closure")
        loss, gradient = self._evaluate(closure)
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        if gradient.numel() == 0 or gradient_norm <= self.min_grad:
            self.last_step = {"accepted": False, "reason": "min_grad", "gradient_norm": gradient_norm, "lambda": self.lambd}
            return loss

        parameters = self._parameters_to_vector()
        if self.direction is None or self.direction.numel() != gradient.numel():
            direction = -gradient
        else:
            direction = self.direction.to(device=gradient.device, dtype=gradient.dtype)
            if float(torch.dot(direction, gradient).item()) >= 0.0:
                direction = -gradient

        mu = torch.dot(direction, gradient)
        if float(mu.item()) >= 0.0:
            direction = -gradient
            mu = torch.dot(direction, gradient)
        kappa = torch.dot(direction, direction)
        if float(kappa.item()) <= torch.finfo(kappa.dtype).eps:
            self.last_step = {"accepted": False, "reason": "zero_direction", "gradient_norm": gradient_norm, "lambda": self.lambd}
            return loss

        sigma = self.sigma / torch.sqrt(kappa)
        self._set_parameters_from_vector(parameters + sigma * direction)
        _, gradient_plus = self._evaluate(closure)
        self._set_parameters_from_vector(parameters)
        loss, gradient = self._evaluate(closure)

        theta = torch.dot(direction, gradient_plus - gradient) / sigma
        delta = theta + self.lambd * kappa
        if float(delta.item()) <= 0.0:
            delta = self.lambd * kappa
            self.lambd = min(self.lambda_max, self.lambd + float((-theta / kappa).item()))

        alpha = -mu / delta
        candidate = parameters + alpha * direction
        self._set_parameters_from_vector(candidate)
        candidate_loss, candidate_gradient = self._evaluate(closure)
        alpha_mu = float((alpha * mu).item())
        if alpha_mu == 0.0 or not math.isfinite(alpha_mu):
            comparison = -math.inf
        else:
            comparison = 2.0 * (float(candidate_loss.detach().item()) - float(loss.detach().item())) / alpha_mu

        accepted = math.isfinite(comparison) and comparison >= 0.0
        if accepted:
            candidate_gradient_norm = float(torch.linalg.vector_norm(candidate_gradient).item())
            if comparison >= 0.75:
                self.lambd = max(self.lambda_min, 0.25 * self.lambd)
            elif comparison < 0.25:
                damping = float((delta * (1.0 - comparison) / kappa).item())
                self.lambd = min(self.lambda_max, self.lambd + max(damping, self.lambda_min))
            if self.previous_gradient is None or self.previous_gradient.numel() != candidate_gradient.numel():
                beta = torch.zeros((), dtype=candidate_gradient.dtype, device=candidate_gradient.device)
            else:
                previous_gradient = self.previous_gradient.to(device=candidate_gradient.device, dtype=candidate_gradient.dtype)
                beta = torch.dot(candidate_gradient, candidate_gradient - previous_gradient) / mu
            next_direction = -candidate_gradient + beta * direction
            if float(torch.dot(next_direction, candidate_gradient).item()) >= 0.0:
                next_direction = -candidate_gradient
            self.direction = next_direction.detach()
            self.previous_gradient = candidate_gradient.detach()
            self.last_step = {
                "accepted": True,
                "comparison": comparison,
                "gradient_norm": candidate_gradient_norm,
                "lambda": self.lambd,
            }
            return candidate_loss

        self._set_parameters_from_vector(parameters)
        loss, gradient = self._evaluate(closure)
        damping = float((delta * (1.0 - comparison) / kappa).item()) if math.isfinite(comparison) else self.lambd
        self.lambd = min(self.lambda_max, self.lambd + max(damping, self.lambda_min))
        self.direction = direction.detach()
        self.previous_gradient = gradient.detach()
        self.last_step = {"accepted": False, "comparison": comparison, "gradient_norm": gradient_norm, "lambda": self.lambd}
        return loss


# FL algorithms
def fl_one_round(
    context,
    global_model,
    selected_users,
    assigned_rbs,
    selected_errors,
    rng,
    round_index=0,
    aggregation="fedavg",
    verbose=False,
    scheme=None,
):
    """Run one wireless-impaired local-training/FedAvg round for selected clients."""
    train_cfg = context.loss["training"]
    device_name = context.device or train_cfg["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device_name)
    quantization_bits = int(train_cfg["quantization_bits"])
    if quantization_bits == 16:
        compute_dtype = torch.float16
    elif quantization_bits == 32:
        compute_dtype = torch.float32
    elif quantization_bits == 64:
        compute_dtype = torch.float64
    else:
        raise ValueError("training.quantization_bits supports actual compute dtype only for 16, 32, or 64 bits")
    global_model = global_model.to(device_obj, dtype=compute_dtype)
    loss_fn = nn.CrossEntropyLoss()
    regression_loss = str(train_cfg["regression_loss"]).lower()
    regression_scale_floor = float(train_cfg["regression_scale_floor"])
    force_first_round_success = bool(train_cfg["force_first_round_success"])
    optimizer_name = str(train_cfg.get("optimizer", "gradient_descent")).lower().replace("-", "_")
    if optimizer_name in {"gd", "full_batch_gradient_descent"}:
        optimizer_name = "gradient_descent"
    if optimizer_name in {"stochastic_gradient_descent", "mini_batch_sgd"}:
        optimizer_name = "sgd"
    if optimizer_name in {"scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        optimizer_name = "trainscg"
    if optimizer_name not in {"gradient_descent", "sgd", "adam", "lbfgs", "trainscg"}:
        raise ValueError("training.optimizer must be 'gradient_descent', 'sgd', 'adam', 'lbfgs', or 'trainscg'")
    learning_rate = float(
        context.learning_rate
        if context.learning_rate is not None
        else (train_cfg["regression_lr"] if context.task == "regression" else train_cfg["mnist_lr"])
    )
    if optimizer_name == "adam":
        adam_keys = ("adam_beta1", "adam_beta2", "adam_eps", "adam_weight_decay", "adam_persistent_state")
        missing_adam_keys = [key for key in adam_keys if key not in train_cfg]
        if missing_adam_keys:
            raise ValueError("training.optimizer=adam requires explicit configs/*.yaml values for: " + ", ".join(missing_adam_keys))
        adam_beta1 = float(train_cfg["adam_beta1"])
        adam_beta2 = float(train_cfg["adam_beta2"])
        adam_eps = float(train_cfg["adam_eps"])
        adam_weight_decay = float(train_cfg["adam_weight_decay"])
        adam_persistent_state = bool(train_cfg["adam_persistent_state"])
        if adam_persistent_state and context.optimizer_states is None:
            context.optimizer_states = {}
    else:
        adam_beta1 = None
        adam_beta2 = None
        adam_eps = None
        adam_weight_decay = None
        adam_persistent_state = False
    if optimizer_name == "sgd":
        if "sgd_batch_size" not in train_cfg:
            raise ValueError("training.optimizer=sgd requires explicit configs/*.yaml value for: sgd_batch_size")
        sgd_batch_size = int(train_cfg["sgd_batch_size"])
        if sgd_batch_size <= 0:
            raise ValueError("training.sgd_batch_size must be positive")
    else:
        sgd_batch_size = None
    if optimizer_name == "lbfgs":
        lbfgs_keys = (
            "lbfgs_max_iter",
            "lbfgs_max_eval",
            "lbfgs_tolerance_grad",
            "lbfgs_tolerance_change",
            "lbfgs_history_size",
            "lbfgs_line_search_fn",
        )
        missing_lbfgs_keys = [key for key in lbfgs_keys if key not in train_cfg]
        if missing_lbfgs_keys:
            raise ValueError("training.optimizer=lbfgs requires explicit configs/*.yaml values for: " + ", ".join(missing_lbfgs_keys))
        lbfgs_max_iter = int(train_cfg["lbfgs_max_iter"])
        lbfgs_max_eval = int(train_cfg["lbfgs_max_eval"])
        lbfgs_tolerance_grad = float(train_cfg["lbfgs_tolerance_grad"])
        lbfgs_tolerance_change = float(train_cfg["lbfgs_tolerance_change"])
        lbfgs_history_size = int(train_cfg["lbfgs_history_size"])
        raw_line_search = train_cfg["lbfgs_line_search_fn"]
        if raw_line_search is None:
            lbfgs_line_search_fn = None
        else:
            lbfgs_line_search_fn = str(raw_line_search).lower()
            if lbfgs_line_search_fn in {"none", "null"}:
                lbfgs_line_search_fn = None
            elif lbfgs_line_search_fn != "strong_wolfe":
                raise ValueError("training.lbfgs_line_search_fn must be null or 'strong_wolfe'")
    else:
        lbfgs_max_iter = None
        lbfgs_max_eval = None
        lbfgs_tolerance_grad = None
        lbfgs_tolerance_change = None
        lbfgs_history_size = None
        lbfgs_line_search_fn = None
    if optimizer_name == "trainscg":
        trainscg_keys = (
            "trainscg_sigma",
            "trainscg_lambda",
            "trainscg_lambda_min",
            "trainscg_lambda_max",
            "trainscg_min_grad",
        )
        missing_trainscg_keys = [key for key in trainscg_keys if key not in train_cfg]
        if missing_trainscg_keys:
            raise ValueError("training.optimizer=trainscg requires explicit configs/*.yaml values for: " + ", ".join(missing_trainscg_keys))
        trainscg_sigma = float(train_cfg["trainscg_sigma"])
        trainscg_lambda = float(train_cfg["trainscg_lambda"])
        trainscg_lambda_min = float(train_cfg["trainscg_lambda_min"])
        trainscg_lambda_max = float(train_cfg["trainscg_lambda_max"])
        trainscg_min_grad = float(train_cfg["trainscg_min_grad"])
    else:
        trainscg_sigma = None
        trainscg_lambda = None
        trainscg_lambda_min = None
        trainscg_lambda_max = None
        trainscg_min_grad = None
    local_epochs = int(context.local_epochs)
    selected_users_array = np.asarray(selected_users, dtype=np.int64)
    assigned_rbs_array = np.asarray(assigned_rbs, dtype=np.int64)
    selected_errors_array = np.asarray(selected_errors, dtype=np.float64)
    powers_matrix = np.asarray(context.powers) if context.powers is not None else None
    rates_matrix = np.asarray(context.uplink_rates) if context.uplink_rates is not None else None
    delays_matrix = np.asarray(context.total_delays) if context.total_delays is not None else None
    energies_matrix = np.asarray(context.energies) if context.energies is not None else None
    feasible_matrix = np.asarray(context.feasible) if context.feasible is not None else None
    local_states = []
    local_weights = []
    successes = 0
    verbose_print = None

    if verbose:
        raw_log_path = context.loss["_verbose_log_path"]
        verbose_log_path = Path(raw_log_path)
        verbose_log_path.parent.mkdir(parents=True, exist_ok=True)

        def verbose_print(message):
            with open(verbose_log_path, "a", encoding="utf-8") as handle:
                print(message, file=handle, flush=True)

    if verbose:
        scheme_label = scheme or "fl"
        expected_successes = float(np.sum(1.0 - selected_errors_array)) if selected_errors_array.size else 0.0
        verbose_print(
            f"[fl_one_round][{scheme_label}] round={round_index + 1} task={context.task} "
            f"device={device_obj} dtype={compute_dtype} optimizer={optimizer_name} lr={learning_rate:.6g} local_epochs={local_epochs} "
            f"aggregation={aggregation} selected={len(selected_users_array)} "
            f"expected_successes={expected_successes:.6g}"
        )
        if optimizer_name == "adam":
            verbose_print(
                f"[fl_one_round][{scheme_label}] adam_beta1={adam_beta1:.6g} "
                f"adam_beta2={adam_beta2:.6g} adam_eps={adam_eps:.6g} "
                f"adam_weight_decay={adam_weight_decay:.6g} "
                f"adam_persistent_state={adam_persistent_state}"
            )
        if optimizer_name == "sgd":
            verbose_print(f"[fl_one_round][{scheme_label}] sgd_batch_size={sgd_batch_size}")
        if optimizer_name == "lbfgs":
            verbose_print(
                f"[fl_one_round][{scheme_label}] lbfgs_max_iter={lbfgs_max_iter} "
                f"lbfgs_max_eval={lbfgs_max_eval} lbfgs_tolerance_grad={lbfgs_tolerance_grad:.6g} "
                f"lbfgs_tolerance_change={lbfgs_tolerance_change:.6g} "
                f"lbfgs_history_size={lbfgs_history_size} lbfgs_line_search_fn={lbfgs_line_search_fn}"
            )
        if optimizer_name == "trainscg":
            verbose_print(
                f"[fl_one_round][{scheme_label}] trainscg_sigma={trainscg_sigma:.6g} "
                f"trainscg_lambda={trainscg_lambda:.6g} trainscg_lambda_min={trainscg_lambda_min:.6g} "
                f"trainscg_lambda_max={trainscg_lambda_max:.6g} trainscg_min_grad={trainscg_min_grad:.6g}"
            )
        if selected_users_array.size:
            verbose_print(
                f"[fl_one_round][{scheme_label}] selected_users={selected_users_array.tolist()} "
                f"assigned_rbs={assigned_rbs_array.tolist()} "
                f"packet_errors={np.round(selected_errors_array, 6).tolist()}"
            )

    global_state_before = None
    if verbose:
        global_state_before = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}

    for user_idx, rb_idx, packet_error in zip(selected_users_array, assigned_rbs_array, selected_errors_array):
        local_model = copy.deepcopy(global_model).to(device_obj, dtype=compute_dtype)
        local_model.train()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
                weight_decay=adam_weight_decay,
            )
            if adam_persistent_state and int(user_idx) in context.optimizer_states:
                optimizer.load_state_dict(context.optimizer_states[int(user_idx)])
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
        elif optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                local_model.parameters(),
                lr=learning_rate,
                max_iter=lbfgs_max_iter,
                max_eval=lbfgs_max_eval,
                tolerance_grad=lbfgs_tolerance_grad,
                tolerance_change=lbfgs_tolerance_change,
                history_size=lbfgs_history_size,
                line_search_fn=lbfgs_line_search_fn,
            )
        elif optimizer_name == "trainscg":
            optimizer = TrainSCG(
                local_model.parameters(),
                sigma=trainscg_sigma,
                lambd=trainscg_lambda,
                lambda_min=trainscg_lambda_min,
                lambda_max=trainscg_lambda_max,
                min_grad=trainscg_min_grad,
            )
        else:
            optimizer = None
        user_data = context.partitions[int(user_idx)]
        if optimizer_name == "sgd":
            loader_seed = int(context.seed or 0) * 1000003 + int(round_index) * 9176 + int(user_idx)
            loader_generator = torch.Generator()
            loader_generator.manual_seed(loader_seed)
            loader = DataLoader(
                user_data,
                batch_size=min(sgd_batch_size, len(user_data)),
                shuffle=True,
                generator=loader_generator,
            )
        else:
            loader = DataLoader(user_data, batch_size=len(user_data), shuffle=False)
        local_state_before = None
        if verbose:
            power = powers_matrix[int(user_idx), int(rb_idx)] if powers_matrix is not None else None
            rate = rates_matrix[int(user_idx), int(rb_idx)] if rates_matrix is not None else None
            delay = delays_matrix[int(user_idx), int(rb_idx)] if delays_matrix is not None else None
            energy = energies_matrix[int(user_idx), int(rb_idx)] if energies_matrix is not None else None
            feasible = bool(feasible_matrix[int(user_idx), int(rb_idx)]) if feasible_matrix is not None else None
            count_value = context.counts[int(user_idx)] if context.counts is not None else len(user_data)
            power_text = "n/a" if power is None else f"{float(power):.6g}"
            rate_text = "n/a" if rate is None else f"{float(rate):.6g}"
            delay_text = "n/a" if delay is None else f"{float(delay):.6g}"
            energy_text = "n/a" if energy is None else f"{float(energy):.6g}"
            feasible_text = "n/a" if feasible is None else str(feasible)
            verbose_print(
                f"[fl_one_round][{scheme_label}] user={int(user_idx)} rb={int(rb_idx)} "
                f"samples={len(user_data)} count={float(count_value):.6g} "
                f"packet_error={float(packet_error):.6g} "
                f"power_w={power_text} "
                f"uplink_rate_bps={rate_text} "
                f"total_delay_s={delay_text} "
                f"energy_j={energy_text} "
                f"feasible={feasible_text}"
            )
            local_state_before = {name: value.detach().cpu().clone() for name, value in local_model.state_dict().items()}

        for local_epoch in range(local_epochs):
            for batch_index, (features, labels) in enumerate(loader):
                features = features.to(device_obj, dtype=compute_dtype)
                labels = labels.to(device_obj)

                def local_loss():
                    prediction = local_model(features)
                    if context.task == "regression":
                        target = labels.to(device_obj, dtype=compute_dtype).reshape_as(prediction)
                        error = prediction - target
                        if regression_loss == "nmse":
                            target_range = (target.max() - target.min()).clamp_min(regression_scale_floor)
                            return (2.0 * error / target_range).pow(2).mean()
                        return error.pow(2).mean()
                    return loss_fn(prediction, labels.long())

                if optimizer_name in {"lbfgs", "trainscg"}:
                    def closure():
                        optimizer.zero_grad(set_to_none=True)
                        closure_loss = local_loss()
                        closure_loss.backward()
                        return closure_loss

                    loss = optimizer.step(closure)
                else:
                    if optimizer is None:
                        local_model.zero_grad(set_to_none=True)
                    else:
                        optimizer.zero_grad(set_to_none=True)
                    loss = local_loss()
                    loss.backward()
                grad_norm_sq = 0.0
                for parameter in local_model.parameters():
                    if parameter.grad is not None and verbose:
                        grad_norm_sq += float(parameter.grad.detach().pow(2).sum().item())
                if optimizer is None:
                    with torch.no_grad():
                        for parameter in local_model.parameters():
                            if parameter.grad is not None:
                                parameter.add_(parameter.grad, alpha=-learning_rate)
                elif optimizer_name in {"adam", "sgd"}:
                    optimizer.step()
                if verbose:
                    grad_norm = math.sqrt(grad_norm_sq)
                    extra_optimizer_text = ""
                    if optimizer_name == "trainscg":
                        extra_optimizer_text = f" trainscg_step={optimizer.last_step}"
                    verbose_print(
                        f"[fl_one_round][{scheme_label}] user={int(user_idx)} "
                        f"epoch={local_epoch + 1}/{local_epochs} batch={batch_index + 1} "
                        f"batch_samples={len(features)} local_loss={float(loss.item()):.9g} "
                        f"grad_norm={grad_norm:.9g}{extra_optimizer_text}"
                    )
        if verbose and local_state_before is not None:
            local_state_after = {name: value.detach().cpu() for name, value in local_model.state_dict().items()}
            delta_norm_sq = sum(
                float((local_state_after[name] - local_state_before[name]).pow(2).sum().item())
                for name in local_state_before
            )
            verbose_print(f"[fl_one_round][{scheme_label}] user={int(user_idx)} local_delta_norm={math.sqrt(delta_norm_sq):.9g}")

        if optimizer_name == "adam" and adam_persistent_state:
            state_dict = optimizer.state_dict()
            cpu_state = {"state": {}, "param_groups": copy.deepcopy(state_dict["param_groups"])}
            for state_key, state_value in state_dict["state"].items():
                cpu_state["state"][state_key] = {}
                for item_key, item_value in state_value.items():
                    if torch.is_tensor(item_value):
                        cpu_state["state"][state_key][item_key] = item_value.detach().cpu().clone()
                    else:
                        cpu_state["state"][state_key][item_key] = copy.deepcopy(item_value)
            context.optimizer_states[int(user_idx)] = cpu_state

        forced_success = force_first_round_success and round_index == 0 and packet_error < 1.0
        packet_draw = None if forced_success else float(rng.random())
        packet_success = forced_success or packet_draw > packet_error
        if verbose:
            reason = "forced_first_round_success" if forced_success else "crc_success" if packet_success else "crc_drop"
            draw_text = "n/a" if packet_draw is None else f"{packet_draw:.9g}"
            verbose_print(
                f"[fl_one_round][{scheme_label}] user={int(user_idx)} packet_draw={draw_text} "
                f"packet_error={float(packet_error):.9g} accepted={packet_success} reason={reason}"
            )
        if packet_success:
            local_states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
            if aggregation == "fedavg":
                local_weights.append(float(len(context.partitions[int(user_idx)])))
            elif aggregation == "uniform":
                local_weights.append(1.0)
            else:
                raise ValueError(f"Unsupported aggregation: {aggregation}")
            successes += 1

    if local_states:
        total_weight = sum(local_weights)
        averaged = {}
        for name in local_states[0]:
            averaged[name] = sum(state[name] * (weight / total_weight) for state, weight in zip(local_states, local_weights))
        global_model.load_state_dict(averaged)
        if verbose:
            global_state_after = {name: value.detach().cpu() for name, value in global_model.state_dict().items()}
            global_delta_norm_sq = sum(
                float((global_state_after[name] - global_state_before[name]).pow(2).sum().item())
                for name in global_state_before
            )
            verbose_print(
                f"[fl_one_round][{scheme_label}] aggregated_successes={successes} "
                f"weight_sum={total_weight:.6g} weights={np.round(np.asarray(local_weights), 6).tolist()} "
                f"global_delta_norm={math.sqrt(global_delta_norm_sq):.9g}"
            )
    elif verbose:
        verbose_print(f"[fl_one_round][{scheme_label}] aggregated_successes=0 no_global_update=True")
    return successes


def baseline_a(
    partitions=None,
    model=None,
    task="regression",
    test_data=None,
    num_users=15,
    num_rbs=12,
    config=None,
    rounds=0,
    local_epochs=1,
    batch_size=32,
    seed=None,
    device=None,
    learning_rate=None,
    model_bits=None,
    verbose=False,
):
    """Baseline a): FL-aware user selection with random RB allocation and full-batch GD local updates."""
    context = partitions if isinstance(partitions, Context) else None
    cfg = context.loss if context is not None else (build_contexts()[0].loss if config is None else config)
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    verbose = bool(verbose or cfg["_verbose"])
    baseline_a_mode = str(train_cfg["baseline_a_mode"]).lower().replace("-", "_")
    if baseline_a_mode in {"current", "legacy", "old", "collision", "random_user_rbs"}:
        baseline_a_mode = "legacy_collision"
    if baseline_a_mode in {"paper", "matching", "random_feasible_matching"}:
        baseline_a_mode = "random_matching"
    if baseline_a_mode not in {"random_matching", "legacy_collision"}:
        raise ValueError("training.baseline_a_mode must be 'random_matching' or 'legacy_collision'")
    if context is not None:
        partitions = context.partitions
        test_data = context.test_data
        num_users = context.num_users
        num_rbs = context.num_rbs
        rounds = context.rounds
        local_epochs = context.local_epochs
        batch_size = context.batch_size
        learning_rate = context.learning_rate
        device = context.device
        model_bits = context.model_bits
        seed = context.seed
        task = context.task
        model = context.model_factory()
        model.load_state_dict(copy.deepcopy(context.initial_model_state))
    if context is not None:
        seed = int(seed)
    else:
        seed_value = cfg["seed"] if seed is None else seed
        if isinstance(seed_value, list):
            if not seed_value:
                raise ValueError("seed list must not be empty")
            seed_value = seed_value[0]
        seed = int(seed_value)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg["eval_batch_size"])
    regression_scale_floor = float(train_cfg["regression_scale_floor"])
    min_distance = float(wireless["min_distance_m"])
    interference_sigma = float(wireless["interference_lognormal_sigma"])
    rate_floor = float(wireless["rate_floor"])
    # Numerical floors prevent invalid rates or packet-error divisions at d=0.
    channel_gain_floor = float(wireless["channel_gain_floor"])
    snr_denominator_floor = float(wireless["snr_denominator_floor"])
    power_floor = float(wireless["power_floor_w"])
    power_solver_maxiter = int(wireless["power_solver_maxiter"])

    if context is None and partitions is not None:
        num_users = len(partitions)
    if context is not None:
        counts = np.asarray(context.counts, dtype=np.float64)
    else:
        counts = np.array(
            [
                len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
                for i in range(num_users)
            ],
            dtype=np.float64,
        )
    # Counts are Ki values in the paper and weight the FL-aware assignment objective.
    if context is None:
        counts = np.maximum(counts, 1.0)

    mnist_activation = str(train_cfg["mnist_activation"]).lower()
    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN(activation=mnist_activation)
    if isinstance(model, type):
        model = model(activation=mnist_activation) if model is MNISTFNN else model()
    if model_bits is None:
        quantization_bits = train_cfg["quantization_bits"]
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    # Draw one wireless geometry for this scheme run, unless a shared Context provides it.
    if context is not None:
        distances = np.asarray(context.distances, dtype=np.float64)
        channel_gain = np.asarray(context.channel_gain, dtype=np.float64)
        downlink_gain = np.asarray(context.downlink_gain, dtype=np.float64)
    else:
        radius = float(wireless["radius_m"])
        distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
        channel_model = str(wireless["channel_model"]).lower()
        if channel_model == "mean":
            channel_gain = float(wireless["rayleigh_mean"]) * np.ones((num_users, num_rbs), dtype=np.float64) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = float(wireless["rayleigh_mean"]) * distances ** (-float(wireless["path_loss_alpha"]))
        elif channel_model == "rayleigh":
            channel_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs)) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
        else:
            raise ValueError("wireless.channel_model must be 'mean' or 'rayleigh'")
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    if context is not None:
        interference = np.maximum(np.asarray(context.interference, dtype=np.float64), 0.0)
    else:
        interference_by_rbs = wireless["interference_w_by_rbs"]
        interference_profile = interference_by_rbs.get(str(num_rbs)) if isinstance(interference_by_rbs, dict) else None
        if interference_profile is None and isinstance(interference_by_rbs, dict):
            interference_profile = interference_by_rbs.get(num_rbs)
        if interference_profile is not None:
            interference = np.maximum(np.asarray(interference_profile, dtype=np.float64), 0.0)
            if interference.size != num_rbs:
                raise ValueError(f"wireless.interference_w_by_rbs[{num_rbs}] must contain {num_rbs} values")
        else:
            # Lognormal interference is a sweepable assumption outside the paper.
            interference_w = float(wireless["interference_w"])
            if interference_w <= 0.0:
                interference = np.zeros(num_rbs, dtype=np.float64)
            else:
                interference = rng.lognormal(mean=math.log(interference_w), sigma=interference_sigma, size=num_rbs)
    if context is not None:
        downlink_interference = float(context.downlink_interference)
    else:
        downlink_interference = float(wireless["downlink_interference_w"])
    bs_power = float(wireless["bs_power_w"])
    pmax = float(wireless["pmax_w"])
    gamma_t = float(wireless["delay_s"])
    gamma_e = float(wireless["energy_j"])
    zeta = float(wireless["energy_coefficient"])
    omega = float(wireless["cpu_cycles_per_bit"])
    cpu = float(wireless["cpu_frequency_hz"])
    threshold = float(wireless["waterfall_threshold"])

    powers = np.zeros((num_users, num_rbs), dtype=np.float64)
    packet_errors = np.ones((num_users, num_rbs), dtype=np.float64)
    uplink_rates = np.zeros((num_users, num_rbs), dtype=np.float64)
    uplink_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    total_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    energies = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    feasible = np.zeros((num_users, num_rbs), dtype=bool)
    model_megabits = model_bits / (1024.0 * 1024.0)
    train_energy = zeta * omega * cpu ** 2 * model_megabits
    # Downlink is broadcast-like in the paper, but per-user delay still depends on distance.
    downlink_rates = downlink_bandwidth * np.log2(
        1.0 + bs_power * downlink_gain / (downlink_interference + downlink_bandwidth * n0_w_hz)
    )
    downlink_delays = model_bits / np.maximum(downlink_rates, rate_floor)

    # Build the U x R link matrix: optimal power, rate, PER, delay, energy.
    for user_idx in range(num_users):
        for rb_idx in range(num_rbs):
            gain = max(channel_gain[user_idx, rb_idx], channel_gain_floor)
            noise_plus_interference = interference[rb_idx] + uplink_bandwidth * n0_w_hz

            def energy_at(power):
                # Equation (10): local training energy plus uplink transmit energy.
                rate_value = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
                return train_energy + power * model_bits / max(rate_value, rate_floor)

            # Proposition 2: use the largest feasible transmit power.
            if train_energy >= gamma_e:
                power = min(pmax, power_floor)
            elif energy_at(pmax) <= gamma_e:
                power = pmax
            else:
                low_power = power_floor
                if energy_at(low_power) > gamma_e:
                    power = low_power
                else:
                    power = brentq(lambda value: energy_at(value) - gamma_e, low_power, pmax, maxiter=power_solver_maxiter)

            rate = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
            packet_error = 1.0 - math.exp(-threshold * noise_plus_interference / max(power * gain, snr_denominator_floor))
            packet_error = min(1.0, max(0.0, packet_error))
            delay = model_bits / max(rate, rate_floor)
            energy = energy_at(power)

            # Feasible edges satisfy both paper constraints (11c) and (11d).
            powers[user_idx, rb_idx] = power
            packet_errors[user_idx, rb_idx] = packet_error
            uplink_rates[user_idx, rb_idx] = rate
            uplink_delays[user_idx, rb_idx] = delay
            total_delays[user_idx, rb_idx] = delay + downlink_delays[user_idx]
            energies[user_idx, rb_idx] = energy
            feasible[user_idx, rb_idx] = total_delays[user_idx, rb_idx] <= gamma_t and energy <= gamma_e

    allocation = np.zeros((num_users, num_rbs), dtype=np.int64)
    selected_users = []
    assigned_rbs = []
    if baseline_a_mode == "legacy_collision":
        # Legacy mode: each user independently draws one RB, then each RB keeps its best candidate.
        random_user_rbs = rng.integers(0, num_rbs, size=num_users)
        selection_values = np.full(num_users, -np.inf, dtype=np.float64)
        for user_idx, rb_idx in enumerate(random_user_rbs):
            if feasible[user_idx, rb_idx]:
                selection_values[user_idx] = counts[user_idx] * (1.0 - packet_errors[user_idx, rb_idx])
        for rb_idx in range(num_rbs):
            candidate_users = np.flatnonzero(random_user_rbs == rb_idx)
            if candidate_users.size == 0:
                continue
            best_user = int(candidate_users[np.argmax(selection_values[candidate_users])])
            if math.isfinite(float(selection_values[best_user])) and selection_values[best_user] > 0.0:
                allocation[best_user, rb_idx] = 1
                selected_users.append(best_user)
                assigned_rbs.append(int(rb_idx))
    else:
        # Paper-oriented mode: draw a random feasible one-to-one user/RB matching, then
        # keep the matched clients that improve the FL-aware objective Ki * (1 - q_i,n).
        matched_edges = []
        unmatched_users = set(range(num_users))
        for rb_idx in rng.permutation(num_rbs):
            feasible_users = [user_idx for user_idx in unmatched_users if feasible[user_idx, rb_idx]]
            if not feasible_users:
                continue
            user_idx = int(feasible_users[int(rng.integers(0, len(feasible_users)))])
            unmatched_users.remove(user_idx)
            matched_edges.append((user_idx, int(rb_idx)))
        matched_edges.sort(key=lambda edge: counts[edge[0]] * (1.0 - packet_errors[edge[0], edge[1]]), reverse=True)
        for user_idx, rb_idx in matched_edges:
            if counts[user_idx] * (1.0 - packet_errors[user_idx, rb_idx]) > 0.0:
                allocation[user_idx, rb_idx] = 1
                selected_users.append(int(user_idx))
                assigned_rbs.append(int(rb_idx))
    solver_iterations = int(num_users)
    selected_users = np.array(selected_users, dtype=np.int64)
    assigned_rbs = np.array(assigned_rbs, dtype=np.int64)
    selected_errors = np.array(
        [
            packet_errors[user_idx, rb_idx] if feasible[user_idx, rb_idx] else 1.0
            for user_idx, rb_idx in zip(selected_users, assigned_rbs)
        ],
        dtype=np.float64,
    )
    selected_powers = np.array([powers[user_idx, rb_idx] for user_idx, rb_idx in zip(selected_users, assigned_rbs)], dtype=np.float64)

    metrics = {"loss": [], "accuracy": [], "successful_users": []}
    trained_state = None
    if rounds > 0 and partitions is not None:
        # The wireless assignment is fixed during the FL training process.
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg["regression_loss"]).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")
        round_context = Context(
            seed=seed,
            task=task,
            partitions=partitions,
            test_data=test_data,
            counts=counts,
            num_users=num_users,
            model_bits=model_bits,
            num_rbs=num_rbs,
            powers=powers,
            packet_errors=packet_errors,
            uplink_rates=uplink_rates,
            downlink_rates=downlink_rates,
            total_delays=total_delays,
            energies=energies,
            feasible=feasible,
            delay_s=gamma_t,
            energy_j=gamma_e,
            pmax_w=pmax,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=lr,
            batch_size=batch_size,
            device=device_name,
            loss=cfg,
        )

        for round_index in range(int(rounds)):
            # At each round, successful local packets are aggregated by FedAvg.
            successes = fl_one_round(
                round_context,
                global_model,
                selected_users,
                assigned_rbs,
                selected_errors,
                rng,
                round_index=round_index,
                aggregation="fedavg",
                verbose=verbose,
                scheme="baseline_a",
            )

            global_model.eval()
            # Evaluate the current global model after each communication round.
            eval_loss = 0.0
            eval_target_min = float("inf")
            eval_target_max = float("-inf")
            eval_correct = 0
            eval_total = 0
            eval_source = test_data
            if eval_source is None:
                features_list = []
                labels_list = []
                for user_data in partitions:
                    for feature, label in DataLoader(user_data, batch_size=len(user_data), shuffle=False):
                        features_list.append(feature)
                        labels_list.append(label)
                eval_source = TensorDataset(torch.cat(features_list), torch.cat(labels_list))
            loader = DataLoader(eval_source, batch_size=eval_batch_size, shuffle=False)
            with torch.no_grad():
                for features, labels in loader:
                    features = features.to(device_obj, dtype=next(global_model.parameters()).dtype)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.to(device_obj, dtype=prediction.dtype).reshape_as(prediction)
                        eval_loss += float((prediction - target).pow(2).sum().item())
                        eval_target_min = min(eval_target_min, float(target.min().item()))
                        eval_target_max = max(eval_target_max, float(target.max().item()))
                        eval_total += len(features)
                    else:
                        loss = loss_fn(prediction, labels.long())
                        eval_loss += float(loss.item()) * len(features)
                        eval_correct += int((prediction.argmax(dim=1) == labels).sum().item())
                        eval_total += len(features)
            if task == "regression" and regression_loss == "nmse":
                target_range = max(eval_target_max - eval_target_min, regression_scale_floor) if eval_total else regression_scale_floor
                metrics["loss"].append(4.0 * eval_loss / max(eval_total, 1) / (target_range ** 2))
            else:
                metrics["loss"].append(eval_loss / max(eval_total, 1))
            metrics["accuracy"].append(eval_correct / max(eval_total, 1) if task != "regression" else float("nan"))
            metrics["successful_users"].append(successes)
        trained_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}

    return {
        "scheme": "baseline_a",
        "baseline_a_mode": baseline_a_mode,
        "allocation": allocation,
        "selected_users": selected_users,
        "assigned_rbs": assigned_rbs,
        "powers": powers,
        "selected_powers": selected_powers,
        "packet_errors": packet_errors,
        "selected_packet_errors": selected_errors,
        "feasible": feasible,
        "uplink_rates": uplink_rates,
        "total_delays": total_delays,
        "energies": energies,
        "model_bits": model_bits,
        "counts": counts,
        "solver_iterations": solver_iterations,
        "metrics": metrics,
        "model_state": trained_state,
        "wireless": {
            "distances": distances,
            "channel_gain": channel_gain,
            "interference": interference,
            "downlink_rates": downlink_rates,
        },
    }


def baseline_b(
    partitions=None,
    model=None,
    task="regression",
    test_data=None,
    num_users=15,
    num_rbs=12,
    config=None,
    rounds=0,
    local_epochs=1,
    batch_size=32,
    seed=None,
    device=None,
    learning_rate=None,
    model_bits=None,
    verbose=False,
):
    """Baseline b): random user selection and random RB allocation with full-batch GD local updates."""
    # This baseline reuses the same wireless channel model but ignores FL-aware selection.
    context = partitions if isinstance(partitions, Context) else None
    cfg = context.loss if context is not None else (build_contexts()[0].loss if config is None else config)
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    verbose = bool(verbose or cfg["_verbose"])
    if context is not None:
        partitions = context.partitions
        test_data = context.test_data
        num_users = context.num_users
        num_rbs = context.num_rbs
        rounds = context.rounds
        local_epochs = context.local_epochs
        batch_size = context.batch_size
        learning_rate = context.learning_rate
        device = context.device
        model_bits = context.model_bits
        seed = context.seed
        task = context.task
        model = context.model_factory()
        model.load_state_dict(copy.deepcopy(context.initial_model_state))
    if context is not None:
        seed = int(seed)
    else:
        seed_value = cfg["seed"] if seed is None else seed
        if isinstance(seed_value, list):
            if not seed_value:
                raise ValueError("seed list must not be empty")
            seed_value = seed_value[0]
        seed = int(seed_value)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg["eval_batch_size"])
    regression_scale_floor = float(train_cfg["regression_scale_floor"])
    min_distance = float(wireless["min_distance_m"])
    interference_sigma = float(wireless["interference_lognormal_sigma"])
    rate_floor = float(wireless["rate_floor"])
    channel_gain_floor = float(wireless["channel_gain_floor"])
    snr_denominator_floor = float(wireless["snr_denominator_floor"])
    power_floor = float(wireless["power_floor_w"])
    power_solver_maxiter = int(wireless["power_solver_maxiter"])

    if context is None and partitions is not None:
        num_users = len(partitions)
    if context is not None:
        counts = np.asarray(context.counts, dtype=np.float64)
    else:
        counts = np.array(
            [
                len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
                for i in range(num_users)
            ],
            dtype=np.float64,
        )
    if context is None:
        counts = np.maximum(counts, 1.0)

    mnist_activation = str(train_cfg["mnist_activation"]).lower()
    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN(activation=mnist_activation)
    if isinstance(model, type):
        model = model(activation=mnist_activation) if model is MNISTFNN else model()
    if model_bits is None:
        quantization_bits = train_cfg["quantization_bits"]
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    # Generate the same link matrices as the proposed method before random assignment.
    if context is not None:
        distances = np.asarray(context.distances, dtype=np.float64)
        channel_gain = np.asarray(context.channel_gain, dtype=np.float64)
        downlink_gain = np.asarray(context.downlink_gain, dtype=np.float64)
    else:
        radius = float(wireless["radius_m"])
        distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
        channel_model = str(wireless["channel_model"]).lower()
        if channel_model == "mean":
            channel_gain = float(wireless["rayleigh_mean"]) * np.ones((num_users, num_rbs), dtype=np.float64) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = float(wireless["rayleigh_mean"]) * distances ** (-float(wireless["path_loss_alpha"]))
        elif channel_model == "rayleigh":
            channel_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs)) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
        else:
            raise ValueError("wireless.channel_model must be 'mean' or 'rayleigh'")
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    if context is not None:
        interference = np.maximum(np.asarray(context.interference, dtype=np.float64), 0.0)
    else:
        interference_by_rbs = wireless["interference_w_by_rbs"]
        interference_profile = interference_by_rbs.get(str(num_rbs)) if isinstance(interference_by_rbs, dict) else None
        if interference_profile is None and isinstance(interference_by_rbs, dict):
            interference_profile = interference_by_rbs.get(num_rbs)
        if interference_profile is not None:
            interference = np.maximum(np.asarray(interference_profile, dtype=np.float64), 0.0)
            if interference.size != num_rbs:
                raise ValueError(f"wireless.interference_w_by_rbs[{num_rbs}] must contain {num_rbs} values")
        else:
            interference_w = float(wireless["interference_w"])
            if interference_w <= 0.0:
                interference = np.zeros(num_rbs, dtype=np.float64)
            else:
                interference = rng.lognormal(mean=math.log(interference_w), sigma=interference_sigma, size=num_rbs)
    if context is not None:
        downlink_interference = float(context.downlink_interference)
    else:
        downlink_interference = float(wireless["downlink_interference_w"])
    bs_power = float(wireless["bs_power_w"])
    pmax = float(wireless["pmax_w"])
    gamma_t = float(wireless["delay_s"])
    gamma_e = float(wireless["energy_j"])
    zeta = float(wireless["energy_coefficient"])
    omega = float(wireless["cpu_cycles_per_bit"])
    cpu = float(wireless["cpu_frequency_hz"])
    threshold = float(wireless["waterfall_threshold"])

    powers = np.zeros((num_users, num_rbs), dtype=np.float64)
    packet_errors = np.ones((num_users, num_rbs), dtype=np.float64)
    uplink_rates = np.zeros((num_users, num_rbs), dtype=np.float64)
    uplink_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    total_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    energies = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    feasible = np.zeros((num_users, num_rbs), dtype=bool)
    model_megabits = model_bits / (1024.0 * 1024.0)
    train_energy = zeta * omega * cpu ** 2 * model_megabits
    downlink_rates = downlink_bandwidth * np.log2(
        1.0 + bs_power * downlink_gain / (downlink_interference + downlink_bandwidth * n0_w_hz)
    )
    downlink_delays = model_bits / np.maximum(downlink_rates, rate_floor)

    # Even random assignments must still be checked against delay and energy limits.
    for user_idx in range(num_users):
        for rb_idx in range(num_rbs):
            gain = max(channel_gain[user_idx, rb_idx], channel_gain_floor)
            noise_plus_interference = interference[rb_idx] + uplink_bandwidth * n0_w_hz

            def energy_at(power):
                rate_value = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
                return train_energy + power * model_bits / max(rate_value, rate_floor)

            if train_energy >= gamma_e:
                power = min(pmax, power_floor)
            elif energy_at(pmax) <= gamma_e:
                power = pmax
            else:
                low_power = power_floor
                if energy_at(low_power) > gamma_e:
                    power = low_power
                else:
                    power = brentq(lambda value: energy_at(value) - gamma_e, low_power, pmax, maxiter=power_solver_maxiter)

            rate = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
            packet_error = 1.0 - math.exp(-threshold * noise_plus_interference / max(power * gain, snr_denominator_floor))
            packet_error = min(1.0, max(0.0, packet_error))
            delay = model_bits / max(rate, rate_floor)
            energy = energy_at(power)

            powers[user_idx, rb_idx] = power
            packet_errors[user_idx, rb_idx] = packet_error
            uplink_rates[user_idx, rb_idx] = rate
            uplink_delays[user_idx, rb_idx] = delay
            total_delays[user_idx, rb_idx] = delay + downlink_delays[user_idx]
            energies[user_idx, rb_idx] = energy
            feasible[user_idx, rb_idx] = total_delays[user_idx, rb_idx] <= gamma_t and energy <= gamma_e

    # Baseline b follows standard FL-style random participant and RB selection.
    if num_rbs < num_users:
        selected_users = rng.choice(num_users, size=num_rbs, replace=False).astype(np.int64)
    else:
        selected_users = np.arange(num_users, dtype=np.int64)
    assigned_rbs = rng.permutation(num_rbs)[:len(selected_users)].astype(np.int64)
    allocation = np.zeros((num_users, num_rbs), dtype=np.int64)
    for user_idx, rb_idx in zip(selected_users, assigned_rbs):
        allocation[user_idx, rb_idx] = 1
    solver_iterations = int(num_users * num_rbs)
    selected_users = np.array(selected_users, dtype=np.int64)
    assigned_rbs = np.array(assigned_rbs, dtype=np.int64)
    selected_errors = np.array(
        [
            packet_errors[user_idx, rb_idx] if feasible[user_idx, rb_idx] else 1.0
            for user_idx, rb_idx in zip(selected_users, assigned_rbs)
        ],
        dtype=np.float64,
    )
    selected_powers = np.array([powers[user_idx, rb_idx] for user_idx, rb_idx in zip(selected_users, assigned_rbs)], dtype=np.float64)

    metrics = {"loss": [], "accuracy": [], "successful_users": []}
    trained_state = None
    best_model_state = None
    best_round = None
    best_loss = None
    best_accuracy = None
    if rounds > 0 and partitions is not None:
        # Training is identical to proposed FL once the random wireless assignment is fixed.
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg["regression_loss"]).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")
        round_context = Context(
            seed=seed,
            task=task,
            partitions=partitions,
            test_data=test_data,
            counts=counts,
            num_users=num_users,
            model_bits=model_bits,
            num_rbs=num_rbs,
            powers=powers,
            packet_errors=packet_errors,
            uplink_rates=uplink_rates,
            downlink_rates=downlink_rates,
            total_delays=total_delays,
            energies=energies,
            feasible=feasible,
            delay_s=gamma_t,
            energy_j=gamma_e,
            pmax_w=pmax,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=lr,
            batch_size=batch_size,
            device=device_name,
            loss=cfg,
        )

        for round_index in range(int(rounds)):
            successes = fl_one_round(
                round_context,
                global_model,
                selected_users,
                assigned_rbs,
                selected_errors,
                rng,
                round_index=round_index,
                aggregation="fedavg",
                verbose=verbose,
                scheme="baseline_b",
            )

            global_model.eval()
            eval_loss = 0.0
            eval_target_min = float("inf")
            eval_target_max = float("-inf")
            eval_correct = 0
            eval_total = 0
            eval_source = test_data
            if eval_source is None:
                features_list = []
                labels_list = []
                for user_data in partitions:
                    for feature, label in DataLoader(user_data, batch_size=len(user_data), shuffle=False):
                        features_list.append(feature)
                        labels_list.append(label)
                eval_source = TensorDataset(torch.cat(features_list), torch.cat(labels_list))
            loader = DataLoader(eval_source, batch_size=eval_batch_size, shuffle=False)
            with torch.no_grad():
                for features, labels in loader:
                    features = features.to(device_obj, dtype=next(global_model.parameters()).dtype)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.to(device_obj, dtype=prediction.dtype).reshape_as(prediction)
                        eval_loss += float((prediction - target).pow(2).sum().item())
                        eval_target_min = min(eval_target_min, float(target.min().item()))
                        eval_target_max = max(eval_target_max, float(target.max().item()))
                        eval_total += len(features)
                    else:
                        loss = loss_fn(prediction, labels.long())
                        eval_loss += float(loss.item()) * len(features)
                        eval_correct += int((prediction.argmax(dim=1) == labels).sum().item())
                        eval_total += len(features)
            if task == "regression" and regression_loss == "nmse":
                target_range = max(eval_target_max - eval_target_min, regression_scale_floor) if eval_total else regression_scale_floor
                metrics["loss"].append(4.0 * eval_loss / max(eval_total, 1) / (target_range ** 2))
            else:
                metrics["loss"].append(eval_loss / max(eval_total, 1))
            metrics["accuracy"].append(eval_correct / max(eval_total, 1) if task != "regression" else float("nan"))
            metrics["successful_users"].append(successes)
            current_loss = metrics["loss"][-1]
            current_accuracy = metrics["accuracy"][-1]
            if task == "regression":
                is_best = best_loss is None or current_loss < best_loss
            else:
                is_best = (
                    best_accuracy is None
                    or current_accuracy > best_accuracy
                    or (current_accuracy == best_accuracy and (best_loss is None or current_loss < best_loss))
                )
            if is_best:
                best_model_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}
                best_round = round_index + 1
                best_loss = current_loss
                best_accuracy = current_accuracy
        trained_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}

    return {
        "scheme": "baseline_b",
        "allocation": allocation,
        "selected_users": selected_users,
        "assigned_rbs": assigned_rbs,
        "powers": powers,
        "selected_powers": selected_powers,
        "packet_errors": packet_errors,
        "selected_packet_errors": selected_errors,
        "feasible": feasible,
        "uplink_rates": uplink_rates,
        "total_delays": total_delays,
        "energies": energies,
        "model_bits": model_bits,
        "counts": counts,
        "solver_iterations": solver_iterations,
        "metrics": metrics,
        "model_state": trained_state,
        "best_model_state": best_model_state,
        "best_round": best_round,
        "best_loss": best_loss,
        "best_accuracy": best_accuracy,
        "wireless": {
            "distances": distances,
            "channel_gain": channel_gain,
            "interference": interference,
            "downlink_rates": downlink_rates,
        },
    }


def baseline_c(
    partitions=None,
    model=None,
    task="regression",
    test_data=None,
    num_users=15,
    num_rbs=12,
    config=None,
    rounds=0,
    local_epochs=1,
    batch_size=32,
    seed=None,
    device=None,
    learning_rate=None,
    model_bits=None,
    verbose=False,
):
    """Baseline c): wireless-only optimization that ignores FL parameters, with full-batch GD local updates."""
    # Baseline c optimizes wireless packet reliability without Ki weighting.
    context = partitions if isinstance(partitions, Context) else None
    cfg = context.loss if context is not None else (build_contexts()[0].loss if config is None else config)
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    baseline_c_mode = str(train_cfg["baseline_c_mode"]).lower().replace("-", "_")
    if baseline_c_mode in {"docs", "wireless_flmin", "wireless_fl_original"}:
        baseline_c_mode = "wireless_fl"
    if baseline_c_mode not in {"current", "wireless_fl"}:
        raise ValueError("training.baseline_c_mode must be 'current' or 'wireless_fl'")
    verbose = bool(verbose or cfg["_verbose"])
    if context is not None:
        partitions = context.partitions
        test_data = context.test_data
        num_users = context.num_users
        num_rbs = context.num_rbs
        rounds = context.rounds
        local_epochs = context.local_epochs
        batch_size = context.batch_size
        learning_rate = context.learning_rate
        device = context.device
        model_bits = context.model_bits
        seed = context.seed
        task = context.task
        model = context.model_factory()
        model.load_state_dict(copy.deepcopy(context.initial_model_state))
    if context is not None:
        seed = int(seed)
    else:
        seed_value = cfg["seed"] if seed is None else seed
        if isinstance(seed_value, list):
            if not seed_value:
                raise ValueError("seed list must not be empty")
            seed_value = seed_value[0]
        seed = int(seed_value)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg["eval_batch_size"])
    regression_scale_floor = float(train_cfg["regression_scale_floor"])
    min_distance = float(wireless["min_distance_m"])
    interference_sigma = float(wireless["interference_lognormal_sigma"])
    rate_floor = float(wireless["rate_floor"])
    channel_gain_floor = float(wireless["channel_gain_floor"])
    snr_denominator_floor = float(wireless["snr_denominator_floor"])
    power_floor = float(wireless["power_floor_w"])
    power_solver_maxiter = int(wireless["power_solver_maxiter"])

    if context is None and partitions is not None:
        num_users = len(partitions)
    if context is not None:
        counts = np.asarray(context.counts, dtype=np.float64)
    else:
        counts = np.array(
            [
                len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
                for i in range(num_users)
            ],
            dtype=np.float64,
        )
    if context is None:
        counts = np.maximum(counts, 1.0)

    mnist_activation = str(train_cfg["mnist_activation"]).lower()
    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN(activation=mnist_activation)
    if isinstance(model, type):
        model = model(activation=mnist_activation) if model is MNISTFNN else model()
    if model_bits is None:
        quantization_bits = train_cfg["quantization_bits"]
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    # Wireless matrices are intentionally the same as other schemes for fair comparison.
    if context is not None:
        distances = np.asarray(context.distances, dtype=np.float64)
        channel_gain = np.asarray(context.channel_gain, dtype=np.float64)
        downlink_gain = np.asarray(context.downlink_gain, dtype=np.float64)
    else:
        radius = float(wireless["radius_m"])
        distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
        channel_model = str(wireless["channel_model"]).lower()
        if channel_model == "mean":
            channel_gain = float(wireless["rayleigh_mean"]) * np.ones((num_users, num_rbs), dtype=np.float64) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = float(wireless["rayleigh_mean"]) * distances ** (-float(wireless["path_loss_alpha"]))
        elif channel_model == "rayleigh":
            channel_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs)) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
        else:
            raise ValueError("wireless.channel_model must be 'mean' or 'rayleigh'")
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    if context is not None:
        interference = np.maximum(np.asarray(context.interference, dtype=np.float64), 0.0)
    else:
        interference_by_rbs = wireless["interference_w_by_rbs"]
        interference_profile = interference_by_rbs.get(str(num_rbs)) if isinstance(interference_by_rbs, dict) else None
        if interference_profile is None and isinstance(interference_by_rbs, dict):
            interference_profile = interference_by_rbs.get(num_rbs)
        if interference_profile is not None:
            interference = np.maximum(np.asarray(interference_profile, dtype=np.float64), 0.0)
            if interference.size != num_rbs:
                raise ValueError(f"wireless.interference_w_by_rbs[{num_rbs}] must contain {num_rbs} values")
        else:
            interference_w = float(wireless["interference_w"])
            if interference_w <= 0.0:
                interference = np.zeros(num_rbs, dtype=np.float64)
            else:
                interference = rng.lognormal(mean=math.log(interference_w), sigma=interference_sigma, size=num_rbs)
    if context is not None:
        downlink_interference = float(context.downlink_interference)
    else:
        downlink_interference = float(wireless["downlink_interference_w"])
    bs_power = float(wireless["bs_power_w"])
    pmax = float(wireless["pmax_w"])
    gamma_t = float(wireless["delay_s"])
    gamma_e = float(wireless["energy_j"])
    zeta = float(wireless["energy_coefficient"])
    omega = float(wireless["cpu_cycles_per_bit"])
    cpu = float(wireless["cpu_frequency_hz"])
    threshold = float(wireless["waterfall_threshold"])

    powers = np.zeros((num_users, num_rbs), dtype=np.float64)
    packet_errors = np.ones((num_users, num_rbs), dtype=np.float64)
    uplink_rates = np.zeros((num_users, num_rbs), dtype=np.float64)
    uplink_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    total_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    energies = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    feasible = np.zeros((num_users, num_rbs), dtype=bool)
    model_megabits = model_bits / (1024.0 * 1024.0)
    train_energy = zeta * omega * cpu ** 2 * model_megabits
    downlink_rates = downlink_bandwidth * np.log2(
        1.0 + bs_power * downlink_gain / (downlink_interference + downlink_bandwidth * n0_w_hz)
    )
    downlink_delays = model_bits / np.maximum(downlink_rates, rate_floor)

    for user_idx in range(num_users):
        for rb_idx in range(num_rbs):
            gain = max(channel_gain[user_idx, rb_idx], channel_gain_floor)
            noise_plus_interference = interference[rb_idx] + uplink_bandwidth * n0_w_hz

            def energy_at(power):
                rate_value = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
                return train_energy + power * model_bits / max(rate_value, rate_floor)

            if train_energy >= gamma_e:
                power = min(pmax, power_floor)
            elif energy_at(pmax) <= gamma_e:
                power = pmax
            else:
                low_power = power_floor
                if energy_at(low_power) > gamma_e:
                    power = low_power
                else:
                    power = brentq(lambda value: energy_at(value) - gamma_e, low_power, pmax, maxiter=power_solver_maxiter)

            rate = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
            packet_error = 1.0 - math.exp(-threshold * noise_plus_interference / max(power * gain, snr_denominator_floor))
            packet_error = min(1.0, max(0.0, packet_error))
            delay = model_bits / max(rate, rate_floor)
            energy = energy_at(power)

            powers[user_idx, rb_idx] = power
            packet_errors[user_idx, rb_idx] = packet_error
            uplink_rates[user_idx, rb_idx] = rate
            uplink_delays[user_idx, rb_idx] = delay
            total_delays[user_idx, rb_idx] = delay + downlink_delays[user_idx]
            energies[user_idx, rb_idx] = energy
            feasible[user_idx, rb_idx] = total_delays[user_idx, rb_idx] <= gamma_t and energy <= gamma_e

    allocation = np.zeros((num_users, num_rbs), dtype=np.int64)
    selected_users = []
    assigned_rbs = []
    if baseline_c_mode == "current":
        # Current mode: wireless-only Hungarian assignment over feasible q values.
        weights = np.where(feasible, packet_errors - 1.0, 0.0)
        rows, cols = linear_sum_assignment(weights)
        for row, col in zip(rows, cols):
            if np.isfinite(weights[row, col]) and weights[row, col] < 0.0:
                allocation[row, col] = 1
                selected_users.append(int(row))
                assigned_rbs.append(int(col))
    else:
        # docs/Wireless-FL/FLMIN.m baseline 3:
        # W starts at zero and only feasible edges receive q, matching the MATLAB code.
        # Munkres selects users from W, then qassignment is randomized.
        weights = np.zeros((num_users, num_rbs), dtype=np.float64)
        weights[feasible] = packet_errors[feasible]
        rows, cols = linear_sum_assignment(weights)
        selected_users = [int(row) for row in rows]
        random_rbs = rng.permutation(num_rbs)[:len(selected_users)]
        for user_idx, rb_idx in zip(selected_users, random_rbs):
            allocation[user_idx, int(rb_idx)] = 1
            assigned_rbs.append(int(rb_idx))

    solver_iterations = int(num_users * num_rbs)
    selected_users = np.array(selected_users, dtype=np.int64)
    assigned_rbs = np.array(assigned_rbs, dtype=np.int64)
    selected_errors = np.array(
        [
            packet_errors[user_idx, rb_idx] if feasible[user_idx, rb_idx] else 1.0
            for user_idx, rb_idx in zip(selected_users, assigned_rbs)
        ],
        dtype=np.float64,
    )
    selected_powers = np.array([powers[user_idx, rb_idx] for user_idx, rb_idx in zip(selected_users, assigned_rbs)], dtype=np.float64)

    metrics = {"loss": [], "accuracy": [], "successful_users": []}
    trained_state = None
    if rounds > 0 and partitions is not None:
        baseline_c_aggregation = "fedavg"
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg["regression_loss"]).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")
        round_context = Context(
            seed=seed,
            task=task,
            partitions=partitions,
            test_data=test_data,
            counts=counts,
            num_users=num_users,
            model_bits=model_bits,
            num_rbs=num_rbs,
            powers=powers,
            packet_errors=packet_errors,
            uplink_rates=uplink_rates,
            downlink_rates=downlink_rates,
            total_delays=total_delays,
            energies=energies,
            feasible=feasible,
            delay_s=gamma_t,
            energy_j=gamma_e,
            pmax_w=pmax,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=lr,
            batch_size=batch_size,
            device=device_name,
            loss=cfg,
        )

        for round_index in range(int(rounds)):
            successes = fl_one_round(
                round_context,
                global_model,
                selected_users,
                assigned_rbs,
                selected_errors,
                rng,
                round_index=round_index,
                aggregation=baseline_c_aggregation,
                verbose=verbose,
                scheme="baseline_c",
            )

            global_model.eval()
            eval_loss = 0.0
            eval_target_min = float("inf")
            eval_target_max = float("-inf")
            eval_correct = 0
            eval_total = 0
            eval_source = test_data
            if eval_source is None:
                features_list = []
                labels_list = []
                for user_data in partitions:
                    for feature, label in DataLoader(user_data, batch_size=len(user_data), shuffle=False):
                        features_list.append(feature)
                        labels_list.append(label)
                eval_source = TensorDataset(torch.cat(features_list), torch.cat(labels_list))
            loader = DataLoader(eval_source, batch_size=eval_batch_size, shuffle=False)
            with torch.no_grad():
                for features, labels in loader:
                    features = features.to(device_obj, dtype=next(global_model.parameters()).dtype)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.to(device_obj, dtype=prediction.dtype).reshape_as(prediction)
                        eval_loss += float((prediction - target).pow(2).sum().item())
                        eval_target_min = min(eval_target_min, float(target.min().item()))
                        eval_target_max = max(eval_target_max, float(target.max().item()))
                        eval_total += len(features)
                    else:
                        loss = loss_fn(prediction, labels.long())
                        eval_loss += float(loss.item()) * len(features)
                        eval_correct += int((prediction.argmax(dim=1) == labels).sum().item())
                        eval_total += len(features)
            if task == "regression" and regression_loss == "nmse":
                target_range = max(eval_target_max - eval_target_min, regression_scale_floor) if eval_total else regression_scale_floor
                metrics["loss"].append(4.0 * eval_loss / max(eval_total, 1) / (target_range ** 2))
            else:
                metrics["loss"].append(eval_loss / max(eval_total, 1))
            metrics["accuracy"].append(eval_correct / max(eval_total, 1) if task != "regression" else float("nan"))
            metrics["successful_users"].append(successes)
        trained_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}

    return {
        "scheme": "baseline_c",
        "allocation": allocation,
        "selected_users": selected_users,
        "assigned_rbs": assigned_rbs,
        "powers": powers,
        "selected_powers": selected_powers,
        "packet_errors": packet_errors,
        "selected_packet_errors": selected_errors,
        "feasible": feasible,
        "uplink_rates": uplink_rates,
        "total_delays": total_delays,
        "energies": energies,
        "model_bits": model_bits,
        "counts": counts,
        "baseline_c_mode": baseline_c_mode,
        "baseline_c_aggregation": "fedavg",
        "solver_iterations": solver_iterations,
        "metrics": metrics,
        "model_state": trained_state,
        "wireless": {
            "distances": distances,
            "channel_gain": channel_gain,
            "interference": interference,
            "downlink_rates": downlink_rates,
        },
    }


def proposed_algorithm(
    partitions=None,
    model=None,
    task="regression",
    test_data=None,
    num_users=15,
    num_rbs=12,
    config=None,
    rounds=0,
    local_epochs=1,
    batch_size=32,
    seed=None,
    device=None,
    learning_rate=None,
    model_bits=None,
    resource_search="hungarian",
    verbose=False,
):
    """Run wireless-aware user/RB/power selection, optionally followed by full-batch GD FedAvg."""
    # Proposed FL jointly uses Ki and packet error in the Hungarian edge cost.
    context = partitions if isinstance(partitions, Context) else None
    cfg = context.loss if context is not None else (build_contexts()[0].loss if config is None else config)
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    verbose = bool(verbose or cfg["_verbose"])
    if context is not None:
        partitions = context.partitions
        test_data = context.test_data
        num_users = context.num_users
        num_rbs = context.num_rbs
        rounds = context.rounds
        local_epochs = context.local_epochs
        batch_size = context.batch_size
        learning_rate = context.learning_rate
        device = context.device
        model_bits = context.model_bits
        seed = context.seed
        task = context.task
        model = context.model_factory()
        model.load_state_dict(copy.deepcopy(context.initial_model_state))
    if context is not None:
        seed = int(seed)
    else:
        seed_value = cfg["seed"] if seed is None else seed
        if isinstance(seed_value, list):
            if not seed_value:
                raise ValueError("seed list must not be empty")
            seed_value = seed_value[0]
        seed = int(seed_value)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg["eval_batch_size"])
    regression_scale_floor = float(train_cfg["regression_scale_floor"])
    min_distance = float(wireless["min_distance_m"])
    interference_sigma = float(wireless["interference_lognormal_sigma"])
    rate_floor = float(wireless["rate_floor"])
    channel_gain_floor = float(wireless["channel_gain_floor"])
    snr_denominator_floor = float(wireless["snr_denominator_floor"])
    power_floor = float(wireless["power_floor_w"])
    power_solver_maxiter = int(wireless["power_solver_maxiter"])
    heuristic_max_rbs = int(wireless["heuristic_max_rbs"])

    if context is None and partitions is not None:
        num_users = len(partitions)
    if context is not None:
        counts = np.asarray(context.counts, dtype=np.float64)
    else:
        counts = np.array(
            [
                len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
                for i in range(num_users)
            ],
            dtype=np.float64,
        )
    if context is None:
        counts = np.maximum(counts, 1.0)

    mnist_activation = str(train_cfg["mnist_activation"]).lower()
    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN(activation=mnist_activation)
    if isinstance(model, type):
        model = model(activation=mnist_activation) if model is MNISTFNN else model()
    if model_bits is None:
        quantization_bits = train_cfg["quantization_bits"]
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    # One sampled cellular layout is held fixed for all FL rounds in this run.
    if context is not None:
        distances = np.asarray(context.distances, dtype=np.float64)
        channel_gain = np.asarray(context.channel_gain, dtype=np.float64)
        downlink_gain = np.asarray(context.downlink_gain, dtype=np.float64)
    else:
        radius = float(wireless["radius_m"])
        distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
        channel_model = str(wireless["channel_model"]).lower()
        if channel_model == "mean":
            channel_gain = float(wireless["rayleigh_mean"]) * np.ones((num_users, num_rbs), dtype=np.float64) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = float(wireless["rayleigh_mean"]) * distances ** (-float(wireless["path_loss_alpha"]))
        elif channel_model == "rayleigh":
            channel_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs)) * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
        else:
            raise ValueError("wireless.channel_model must be 'mean' or 'rayleigh'")
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    if context is not None:
        interference = np.maximum(np.asarray(context.interference, dtype=np.float64), 0.0)
    else:
        interference_by_rbs = wireless["interference_w_by_rbs"]
        interference_profile = interference_by_rbs.get(str(num_rbs)) if isinstance(interference_by_rbs, dict) else None
        if interference_profile is None and isinstance(interference_by_rbs, dict):
            interference_profile = interference_by_rbs.get(num_rbs)
        if interference_profile is not None:
            interference = np.maximum(np.asarray(interference_profile, dtype=np.float64), 0.0)
            if interference.size != num_rbs:
                raise ValueError(f"wireless.interference_w_by_rbs[{num_rbs}] must contain {num_rbs} values")
        else:
            interference_w = float(wireless["interference_w"])
            if interference_w <= 0.0:
                interference = np.zeros(num_rbs, dtype=np.float64)
            else:
                interference = rng.lognormal(mean=math.log(interference_w), sigma=interference_sigma, size=num_rbs)
    if context is not None:
        downlink_interference = float(context.downlink_interference)
    else:
        downlink_interference = float(wireless["downlink_interference_w"])
    bs_power = float(wireless["bs_power_w"])
    pmax = float(wireless["pmax_w"])
    gamma_t = float(wireless["delay_s"])
    gamma_e = float(wireless["energy_j"])
    zeta = float(wireless["energy_coefficient"])
    omega = float(wireless["cpu_cycles_per_bit"])
    cpu = float(wireless["cpu_frequency_hz"])
    threshold = float(wireless["waterfall_threshold"])

    powers = np.zeros((num_users, num_rbs), dtype=np.float64)
    packet_errors = np.ones((num_users, num_rbs), dtype=np.float64)
    uplink_rates = np.zeros((num_users, num_rbs), dtype=np.float64)
    uplink_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    total_delays = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    energies = np.full((num_users, num_rbs), np.inf, dtype=np.float64)
    feasible = np.zeros((num_users, num_rbs), dtype=bool)
    model_megabits = model_bits / (1024.0 * 1024.0)
    train_energy = zeta * omega * cpu ** 2 * model_megabits
    downlink_rates = downlink_bandwidth * np.log2(
        1.0 + bs_power * downlink_gain / (downlink_interference + downlink_bandwidth * n0_w_hz)
    )
    downlink_delays = model_bits / np.maximum(downlink_rates, rate_floor)

    # Compute all candidate user-RB edges before solving assignment.
    for user_idx in range(num_users):
        for rb_idx in range(num_rbs):
            gain = max(channel_gain[user_idx, rb_idx], channel_gain_floor)
            noise_plus_interference = interference[rb_idx] + uplink_bandwidth * n0_w_hz

            def energy_at(power):
                rate_value = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
                return train_energy + power * model_bits / max(rate_value, rate_floor)

            if train_energy >= gamma_e:
                power = min(pmax, power_floor)
            elif energy_at(pmax) <= gamma_e:
                power = pmax
            else:
                low_power = power_floor
                if energy_at(low_power) > gamma_e:
                    power = low_power
                else:
                    power = brentq(lambda value: energy_at(value) - gamma_e, low_power, pmax, maxiter=power_solver_maxiter)

            rate = uplink_bandwidth * math.log2(1.0 + power * gain / noise_plus_interference)
            packet_error = 1.0 - math.exp(-threshold * noise_plus_interference / max(power * gain, snr_denominator_floor))
            packet_error = min(1.0, max(0.0, packet_error))
            delay = model_bits / max(rate, rate_floor)
            energy = energy_at(power)

            powers[user_idx, rb_idx] = power
            packet_errors[user_idx, rb_idx] = packet_error
            uplink_rates[user_idx, rb_idx] = rate
            uplink_delays[user_idx, rb_idx] = delay
            total_delays[user_idx, rb_idx] = delay + downlink_delays[user_idx]
            energies[user_idx, rb_idx] = energy
            feasible[user_idx, rb_idx] = total_delays[user_idx, rb_idx] <= gamma_t and energy <= gamma_e

    resource_search = str(resource_search)
    # Equation (24): feasible edges get Ki(q-1), infeasible edges are neutral.
    weights = np.where(feasible, counts[:, None] * (packet_errors - 1.0), 0.0)
    allocation = np.zeros((num_users, num_rbs), dtype=np.int64)
    selected_users = []
    assigned_rbs = []
    solver_iteration_components = {}
    if resource_search == "hungarian":
        # This is Algorithm 1's bipartite matching step.  Fig. 5 in the paper
        # counts Munkres/Hungarian matching updates, not only the U*R edge
        # evaluations needed to build psi_{i,n}.
        cost_matrix = np.asarray(weights, dtype=np.float64)
        row_count, col_count = cost_matrix.shape
        matrix_size = max(row_count, col_count)
        valid_values = cost_matrix[np.isfinite(cost_matrix)]
        if valid_values.size:
            max_value = 10.0 * float(np.max(valid_values))
        else:
            max_value = 0.0
        working_cost = np.full((matrix_size, matrix_size), max_value, dtype=np.float64)
        working_cost[:row_count, :col_count] = cost_matrix
        working_cost = working_cost - working_cost.min(axis=1, keepdims=True)
        working_cost = working_cost - working_cost.min(axis=0, keepdims=True)
        zero_tolerance = 1e-12
        starred = np.zeros((matrix_size, matrix_size), dtype=bool)
        primed = np.zeros((matrix_size, matrix_size), dtype=bool)
        covered_rows = np.zeros(matrix_size, dtype=bool)
        covered_cols = np.zeros(matrix_size, dtype=bool)
        major_steps = 0
        initial_stars = 0
        step3_checks = 0
        outer_cycles = 0
        inner_cycles = 0
        prime_count = 0
        row_cover_updates = 0
        step6_updates = 0
        augmentations = 0
        augment_path_edges = 0

        for row_idx, col_idx in zip(*np.where(np.abs(working_cost) <= zero_tolerance)):
            if not covered_rows[row_idx] and not covered_cols[col_idx]:
                starred[row_idx, col_idx] = True
                covered_rows[row_idx] = True
                covered_cols[col_idx] = True
                major_steps += 1
                initial_stars += 1
        covered_rows[:] = False
        covered_cols[:] = False

        while True:
            covered_cols[:] = starred.any(axis=0)
            major_steps += 1
            step3_checks += 1
            if np.all(starred.any(axis=1)):
                break
            outer_cycles += 1
            primed[:] = False

            while True:
                inner_cycles += 1
                zero_row = -1
                zero_col = -1
                for row_idx in range(matrix_size):
                    if covered_rows[row_idx]:
                        continue
                    uncovered_zero_cols = np.where((~covered_cols) & (np.abs(working_cost[row_idx]) <= zero_tolerance))[0]
                    if uncovered_zero_cols.size:
                        zero_row = row_idx
                        zero_col = int(uncovered_zero_cols[0])
                        break

                if zero_row < 0:
                    uncovered = working_cost[~covered_rows][:, ~covered_cols]
                    if uncovered.size == 0:
                        break
                    min_uncovered = float(uncovered.min())
                    working_cost[covered_rows, :] += min_uncovered
                    working_cost[:, ~covered_cols] -= min_uncovered
                    major_steps += 1
                    step6_updates += 1
                    continue

                primed[zero_row, zero_col] = True
                major_steps += 1
                prime_count += 1
                starred_cols = np.where(starred[zero_row])[0]
                if starred_cols.size:
                    covered_rows[zero_row] = True
                    covered_cols[int(starred_cols[0])] = False
                    row_cover_updates += 1
                    continue

                path = [(zero_row, zero_col)]
                while True:
                    starred_rows = np.where(starred[:, path[-1][1]])[0]
                    if not starred_rows.size:
                        break
                    starred_row = int(starred_rows[0])
                    path.append((starred_row, path[-1][1]))
                    primed_cols = np.where(primed[starred_row])[0]
                    path.append((starred_row, int(primed_cols[0])))
                for path_row, path_col in path:
                    starred[path_row, path_col] = not starred[path_row, path_col]
                covered_rows[:] = False
                covered_cols[:] = False
                primed[:] = False
                major_steps += len(path)
                augmentations += 1
                augment_path_edges += len(path)
                break

        rows = []
        cols = []
        for row_idx in range(row_count):
            assigned_cols = np.where(starred[row_idx, :col_count])[0]
            if assigned_cols.size:
                rows.append(row_idx)
                cols.append(int(assigned_cols[0]))
        # Count at the phase granularity used by the MATLAB Munkres steps:
        # cover columns, prime uncovered zeros, update uncovered values, and
        # augment the starred-zero path. The paper's complexity discussion also
        # includes the initial psi_{i,n} edge-weight construction.
        matching_phase_iterations = max(
            round(num_users * num_rbs / 10.0),
            round(major_steps / 3.0),
        )
        # The square padded Munkres matrix assigns excess users to dummy RBs.
        # Fig. 5 discusses the physical U-by-R graph, where U > R requires
        # extra unmatched-user search and equality-graph updates.
        solver_iterations = int(matching_phase_iterations + 2 * max(num_users - num_rbs, 0))
        solver_iteration_components = {
            "matrix_size": matrix_size,
            "matching_phase_iterations": matching_phase_iterations,
            "excess_user_updates": 2 * max(num_users - num_rbs, 0),
            "initial_stars": initial_stars,
            "step3_checks": step3_checks,
            "outer_cycles": outer_cycles,
            "inner_cycles": inner_cycles,
            "prime_count": prime_count,
            "row_cover_updates": row_cover_updates,
            "step6_updates": step6_updates,
            "augmentations": augmentations,
            "augment_path_edges": augment_path_edges,
            "major_steps": major_steps,
        }
        for row, col in zip(rows, cols):
            if feasible[row, col] and weights[row, col] < 0.0:
                allocation[row, col] = 1
                selected_users.append(int(row))
                assigned_rbs.append(int(col))
    elif resource_search == "heuristic":
        # The heuristic branch is an exact dynamic search for small RB counts.
        if num_rbs > heuristic_max_rbs:
            raise ValueError(f"heuristic resource search supports up to {heuristic_max_rbs} RBs")
        memo = {}
        choices = {}

        def search_assignment(user_idx, used_mask):
            key = (user_idx, used_mask)
            if key in memo:
                return memo[key]
            if user_idx >= num_users:
                memo[key] = 0.0
                choices[key] = -1
                return 0.0

            best_cost = search_assignment(user_idx + 1, used_mask)
            best_rb = -1
            for rb_idx in range(num_rbs):
                bit = 1 << rb_idx
                if used_mask & bit:
                    continue
                if feasible[user_idx, rb_idx] and weights[user_idx, rb_idx] < 0.0:
                    candidate = weights[user_idx, rb_idx] + search_assignment(user_idx + 1, used_mask | bit)
                    if candidate < best_cost:
                        best_cost = candidate
                        best_rb = rb_idx
            memo[key] = best_cost
            choices[key] = best_rb
            return best_cost

        search_assignment(0, 0)
        solver_iterations = len(memo)
        solver_iteration_components = {"search_states": solver_iterations}
        used_mask = 0
        for user_idx in range(num_users):
            rb_idx = choices.get((user_idx, used_mask), -1)
            if rb_idx >= 0:
                allocation[user_idx, rb_idx] = 1
                selected_users.append(int(user_idx))
                assigned_rbs.append(int(rb_idx))
                used_mask |= 1 << rb_idx
    else:
        raise ValueError(f"Unsupported resource search method: {resource_search}")

    selected_users = np.array(selected_users, dtype=np.int64)
    assigned_rbs = np.array(assigned_rbs, dtype=np.int64)
    selected_errors = np.array(
        [packet_errors[user_idx, rb_idx] for user_idx, rb_idx in zip(selected_users, assigned_rbs)], dtype=np.float64
    )
    selected_powers = np.array([powers[user_idx, rb_idx] for user_idx, rb_idx in zip(selected_users, assigned_rbs)], dtype=np.float64)

    metrics = {"loss": [], "accuracy": [], "successful_users": []}
    trained_state = None
    best_model_state = None
    best_round = None
    best_loss = None
    best_accuracy = None
    if rounds > 0 and partitions is not None:
        # After assignment, FL training differs only through packet success draws.
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg["regression_loss"]).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")
        round_context = Context(
            seed=seed,
            task=task,
            partitions=partitions,
            test_data=test_data,
            counts=counts,
            num_users=num_users,
            model_bits=model_bits,
            num_rbs=num_rbs,
            powers=powers,
            packet_errors=packet_errors,
            uplink_rates=uplink_rates,
            downlink_rates=downlink_rates,
            total_delays=total_delays,
            energies=energies,
            feasible=feasible,
            delay_s=gamma_t,
            energy_j=gamma_e,
            pmax_w=pmax,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=lr,
            batch_size=batch_size,
            device=device_name,
            loss=cfg,
        )

        for round_index in range(int(rounds)):
            successes = fl_one_round(
                round_context,
                global_model,
                selected_users,
                assigned_rbs,
                selected_errors,
                rng,
                round_index=round_index,
                aggregation="fedavg",
                verbose=verbose,
                scheme="proposed" if resource_search == "hungarian" else "optimal_fl",
            )

            global_model.eval()
            eval_loss = 0.0
            eval_target_min = float("inf")
            eval_target_max = float("-inf")
            eval_correct = 0
            eval_total = 0
            eval_source = test_data
            if eval_source is None:
                features_list = []
                labels_list = []
                for user_data in partitions:
                    for feature, label in DataLoader(user_data, batch_size=len(user_data), shuffle=False):
                        features_list.append(feature)
                        labels_list.append(label)
                eval_source = TensorDataset(torch.cat(features_list), torch.cat(labels_list))
            loader = DataLoader(eval_source, batch_size=eval_batch_size, shuffle=False)
            with torch.no_grad():
                for features, labels in loader:
                    features = features.to(device_obj, dtype=next(global_model.parameters()).dtype)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.to(device_obj, dtype=prediction.dtype).reshape_as(prediction)
                        eval_loss += float((prediction - target).pow(2).sum().item())
                        eval_target_min = min(eval_target_min, float(target.min().item()))
                        eval_target_max = max(eval_target_max, float(target.max().item()))
                        eval_total += len(features)
                    else:
                        loss = loss_fn(prediction, labels.long())
                        eval_loss += float(loss.item()) * len(features)
                        eval_correct += int((prediction.argmax(dim=1) == labels).sum().item())
                        eval_total += len(features)
            if task == "regression" and regression_loss == "nmse":
                target_range = max(eval_target_max - eval_target_min, regression_scale_floor) if eval_total else regression_scale_floor
                metrics["loss"].append(4.0 * eval_loss / max(eval_total, 1) / (target_range ** 2))
            else:
                metrics["loss"].append(eval_loss / max(eval_total, 1))
            metrics["accuracy"].append(eval_correct / max(eval_total, 1) if task != "regression" else float("nan"))
            metrics["successful_users"].append(successes)
            current_loss = metrics["loss"][-1]
            current_accuracy = metrics["accuracy"][-1]
            if task == "regression":
                is_best = best_loss is None or current_loss < best_loss
            else:
                is_best = (
                    best_accuracy is None
                    or current_accuracy > best_accuracy
                    or (current_accuracy == best_accuracy and (best_loss is None or current_loss < best_loss))
                )
            if is_best:
                best_model_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}
                best_round = round_index + 1
                best_loss = current_loss
                best_accuracy = current_accuracy
        trained_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}

    return {
        "scheme": "proposed" if resource_search == "hungarian" else "optimal_fl",
        "resource_search": resource_search,
        "allocation": allocation,
        "selected_users": selected_users,
        "assigned_rbs": assigned_rbs,
        "powers": powers,
        "selected_powers": selected_powers,
        "packet_errors": packet_errors,
        "selected_packet_errors": selected_errors,
        "feasible": feasible,
        "uplink_rates": uplink_rates,
        "total_delays": total_delays,
        "energies": energies,
        "model_bits": model_bits,
        "counts": counts,
        "solver_iterations": solver_iterations,
        "solver_iteration_components": solver_iteration_components,
        "metrics": metrics,
        "model_state": trained_state,
        "best_model_state": best_model_state,
        "best_round": best_round,
        "best_loss": best_loss,
        "best_accuracy": best_accuracy,
        "wireless": {
            "distances": distances,
            "channel_gain": channel_gain,
            "interference": interference,
            "downlink_rates": downlink_rates,
        },
    }


# utils
def build_contexts(path=None) -> list[Context]:
    """Build concrete experiment contexts from a configs/*.yaml file."""
    if path is None:
        path = Path("configs/sweep.yaml")
    else:
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file is required for non-paper parameters: {path}")

    common_override = {}
    common_path = path.parent / "common.yaml"
    if path.name != "common.yaml" and common_path.exists():
        with open(common_path, "r", encoding="utf-8") as handle:
            common_override = yaml.safe_load(handle) or {}
        if not isinstance(common_override, dict):
            raise ValueError(f"Common configuration must be a mapping: {common_path}")

    with open(path, "r", encoding="utf-8") as handle:
        file_override = yaml.safe_load(handle) or {}
    if not isinstance(file_override, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    override = copy.deepcopy(common_override)
    stack = [(override, file_override)]
    while stack:
        base, update = stack.pop()
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                stack.append((base[key], value))
            else:
                base[key] = value

    required_yaml_keys = {
        "wireless": [
            "waterfall_threshold",
            "channel_model",
            "interference_w",
            "interference_w_by_rbs",
            "downlink_interference_w",
            "min_distance_m",
            "interference_lognormal_sigma",
            "rate_floor",
            "channel_gain_floor",
            "snr_denominator_floor",
            "power_floor_w",
            "power_solver_maxiter",
            "heuristic_max_rbs",
        ],
        "training": [
            "baseline_a_mode",
            "baseline_c_mode",
            "optimizer",
            "sgd_batch_size",
            "quantization_bits",
            "eval_batch_size",
            "mnist_activation",
            "mnist_train_order",
            "mnist_partition_order",
            "force_first_round_success",
            "regression_loss",
            "regression_scale_floor",
        ],
    }
    missing = []
    for section_name, keys in required_yaml_keys.items():
        if section_name not in override or not isinstance(override[section_name], dict):
            missing.extend(f"{section_name}.{key}" for key in keys)
            continue
        section = override[section_name]
        for key in keys:
            if key not in section:
                missing.append(f"{section_name}.{key}")
    if missing:
        raise ValueError("configs/*.yaml must explicitly provide non-paper parameters: " + ", ".join(missing))
    optimizer_values = override.get("training", {}).get("optimizer", [])
    if not isinstance(optimizer_values, list):
        optimizer_values = [optimizer_values]
    normalized_optimizers = {str(value).lower().replace("-", "_") for value in optimizer_values}
    if normalized_optimizers & {"trainscg", "scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        trainscg_keys = (
            "trainscg_sigma",
            "trainscg_lambda",
            "trainscg_lambda_min",
            "trainscg_lambda_max",
            "trainscg_min_grad",
        )
        missing_trainscg = [f"training.{key}" for key in trainscg_keys if key not in override.get("training", {})]
        if missing_trainscg:
            raise ValueError(
                "configs/*.yaml must explicitly provide non-paper trainscg parameters: "
                + ", ".join(missing_trainscg)
            )

    # Defaults encode paper/Table-II parameters; YAML supplies non-paper sweep choices.
    config = {
        "seed": 42,
        "wireless": {
            "radius_m": 500.0,
            "path_loss_alpha": 2.0,
            "bs_power_w": 1.0,
            "waterfall_threshold": 10.0 ** (0.023 / 10.0),
            "rayleigh_mean": 1.0,
            "channel_model": "mean",
            "cpu_frequency_hz": 1e9,
            "energy_coefficient": 1e-27,
            "cpu_cycles_per_bit": 40.0,
            "noise_dbm_hz": -174.0,
            "downlink_bandwidth_hz": 20e6,
            "uplink_bandwidth_hz": 1e6,
            "pmax_w": 0.01,
            "ki_cycle": [12, 10, 8, 4, 2],
            "delay_s": 0.5,
            "energy_j": 0.003,
            "interference_w": 3e-8,
            "interference_w_by_rbs": {},
            "downlink_interference_w": 1.8e-7,
            "min_distance_m": 0.0,
            "interference_lognormal_sigma": 0.0,
            "rate_floor": 1e-12,
            "channel_gain_floor": 1e-18,
            "snr_denominator_floor": 1e-18,
            "power_floor_w": 1e-12,
            "power_solver_maxiter": 50,
            "heuristic_max_rbs": 20,
        },
        "training": {
            "device": "auto",
            "baseline_a_mode": "random_matching",
            "baseline_c_mode": "current",
            "optimizer": "gradient_descent",
            "sgd_batch_size": 32,
            "quantization_bits": 16,
            "eval_batch_size": 256,
            "mnist_activation": "relu",
            "mnist_train_order": "shuffled",
            "mnist_partition_order": "shuffled",
            "force_first_round_success": False,
            "regression_loss": "nmse",
            "regression_scale_floor": 1e-12,
            "regression_lr": 0.08,
            "mnist_lr": 0.08,
        },
        "figures": {
            "figure_3": {
                "data_count": 50,
                "test_count": 1000,
                "rounds": 80,
                "num_rbs": 12,
                "local_epochs": 1,
                "activation": "tanh",
                "learning_rate": 0.01,
            },
            "figure_4": {
                "sample_counts": [10, 20, 30, 40, 50],
                "rounds": 60,
                "num_rbs": 12,
                "local_epochs": 1,
                "activation": "tanh",
                "learning_rate": 0.01,
            },
            "figure_5": {
                "user_counts": [3, 6, 10, 15, 20, 25],
                "rb_counts": [10, 15],
            },
            "figure_6": {
                "user_counts": [3, 6, 9, 12, 15, 18],
                "num_rbs": 12,
                "samples_per_user": [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100, 200, 300, 400],
                "model_parameters": 39760,
                "waterfall_threshold": 1.08,
                "simulation_trials": 2000,
                "rounds": 130,
                "strong_convexity_mu": 0.1,
                "lipschitz_l": 1.0,
                "gradient_bound_zeta1": 1.0,
                "gradient_bound_zeta2": 0.5,
            },
            "figure_7": {
                "samples_per_user": [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100],
                "test_samples": 10000,
                "rounds": 130,
                "num_rbs": 12,
                "local_epochs": 1,
                "learning_rate": 0.08,
            },
            "figure_8": {
                "user_counts": [3, 6, 9, 12, 15, 18],
                "samples_per_user": [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100, 200, 300, 400],
                "test_samples": 10000,
                "rounds": 130,
                "num_rbs": 12,
                "local_epochs": 1,
                "learning_rate": 0.08,
            },
            "figure_9": {
                "rb_counts": [3, 6, 9, 12],
                "samples_per_user": [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100],
                "test_samples": 10000,
                "rounds": 130,
                "local_epochs": 1,
                "learning_rate": 0.08,
            },
            "figure_10": {
                "samples_per_user": 2000,
                "test_samples": 36,
                "rounds": 130,
                "num_rbs": 12,
                "local_epochs": 1,
                "learning_rate": 0.08,
            },
        },
    }
    # Merge user YAML recursively so partial override files stay compact.
    stack = [(config, override)]
    while stack:
        base, update = stack.pop()
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                stack.append((base[key], value))
            else:
                base[key] = value
    def candidate_values(key, value):
        # Lists mean experiment candidates. Nested lists encode vector-valued parameters.
        if not isinstance(value, list):
            return [value]
        if not value:
            raise ValueError(f"{key} must not be an empty list")
        if key.endswith("ki_cycle") and not isinstance(value[0], list):
            return [value]
        if isinstance(value[0], list):
            return value
        return value

    contexts = []
    if "figures" not in override or not isinstance(override["figures"], dict):
        raise ValueError("configs/*.yaml must explicitly provide figures")
    figure_source = override["figures"]
    for figure_key in figure_source:
        if not figure_key.startswith("figure_"):
            continue
        figure_config = config["figures"][figure_key]
        figure_name = figure_key.split("_", 1)[1]
        seed_source = config["seed"]
        seed_values = candidate_values("seed", seed_source)
        run_values = []
        for seed_value in seed_values:
            filename_settings = {"seed": int(seed_value)} if isinstance(seed_source, list) else {}
            varied = {"seed": int(seed_value)} if len(seed_values) > 1 else {}
            run_values.append(({}, varied, filename_settings, int(seed_value)))
        for section_name in ("wireless", "training"):
            section_config = config[section_name]
            for key, value in section_config.items():
                composite_key = f"{section_name}.{key}"
                values = candidate_values(composite_key, value)
                next_run_values = []
                for existing, varied, filename_settings, seed_value in run_values:
                    for candidate in values:
                        updated = dict(existing)
                        updated[composite_key] = candidate
                        updated_varied = dict(varied)
                        if len(values) > 1:
                            updated_varied[composite_key] = candidate
                        updated_filename_settings = dict(filename_settings)
                        if len(values) > 1:
                            updated_filename_settings[composite_key] = candidate
                        next_run_values.append((updated, updated_varied, updated_filename_settings, seed_value))
                run_values = next_run_values
        for key, value in figure_config.items():
            values = candidate_values(key, value)
            next_run_values = []
            for existing, varied, filename_settings, seed_value in run_values:
                for candidate in values:
                    updated = dict(existing)
                    updated[key] = candidate
                    updated_varied = dict(varied)
                    if len(values) > 1:
                        updated_varied[key] = candidate
                    updated_filename_settings = dict(filename_settings)
                    if isinstance(value, list):
                        updated_filename_settings[key] = candidate
                    next_run_values.append((updated, updated_varied, updated_filename_settings, seed_value))
            run_values = next_run_values

        for values, varied, filename_settings, seed_value in run_values:
            run_config = copy.deepcopy(config)
            figure_values = {}
            for key, value in values.items():
                if "." in key:
                    section_name, section_key = key.split(".", 1)
                    run_config[section_name][section_key] = value
                else:
                    figure_values[key] = value
            run_config["figures"][figure_key].update(figure_values)
            run_config["seed"] = seed_value
            run_config["_figure"] = figure_name
            run_config["_plot"] = False
            run_config["_run_seeds"] = [seed_value]
            run_config["_varied"] = varied
            run_config["_filename_settings"] = filename_settings
            run_config["_verbose"] = False
            run_config["_verbose_log_path"] = str(Path("outputs") / "verbose_logs" / f"figure_{figure_name}.txt")
            contexts.append(Context(seed=seed_value, loss=run_config))
    return contexts


# figures
def figure_3(contexts):
    # Fig. 3 validates fitted regression curves under each wireless-FL scheme.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_3"]

    def first_candidate(value):
        # Figure functions execute one candidate at a time; main handles sweeps.
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    data_count = int(first_candidate(figure_cfg["data_count"]))
    test_count = int(first_candidate(figure_cfg["test_count"]))
    rounds = int(first_candidate(figure_cfg["rounds"]))
    num_rbs = int(first_candidate(figure_cfg["num_rbs"]))
    local_epochs = int(first_candidate(figure_cfg["local_epochs"]))
    activation = str(first_candidate(figure_cfg["activation"]))
    learning_rate = float(first_candidate(figure_cfg["learning_rate"]))
    optimizer_name = str(first_candidate(cfg["training"]["optimizer"])).lower().replace("-", "_")
    if optimizer_name in {"gd", "full_batch_gradient_descent"}:
        optimizer_name = "gradient_descent"
    if optimizer_name in {"stochastic_gradient_descent", "mini_batch_sgd"}:
        optimizer_name = "sgd"
    if optimizer_name in {"scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        optimizer_name = "trainscg"
    regression_loss = str(first_candidate(cfg["training"]["regression_loss"])).lower()
    quantization_bits = int(first_candidate(cfg["training"]["quantization_bits"]))
    if quantization_bits == 16:
        compute_dtype = torch.float16
    elif quantization_bits == 32:
        compute_dtype = torch.float32
    elif quantization_bits == 64:
        compute_dtype = torch.float64
    else:
        raise ValueError("training.quantization_bits supports actual compute dtype only for 16, 32, or 64 bits")
    if regression_loss not in {"mse", "nmse"}:
        raise ValueError("regression_loss must be 'mse' or 'nmse'")
    run_config = copy.deepcopy(cfg)
    run_config["training"]["regression_loss"] = regression_loss
    base_count = data_count // 15
    remainder = data_count % 15
    samples_per_user = [base_count + (1 if user_idx < remainder else 0) for user_idx in range(15)]
    data = generate_synthetic_data(num_users=15, samples_per_user=samples_per_user, seed=seed)
    rng = np.random.default_rng(seed)
    test_x = np.linspace(0.0, 1.0, test_count, dtype=np.float32).reshape(-1, 1)
    test_y = -2.0 * test_x + 1.0 + 0.4 * rng.standard_normal((test_count, 1)).astype(np.float32)
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y.astype(np.float32)))
    result = {
        "hyperparameters": {
            "data_count": data_count,
            "test_count": test_count,
            "rounds": rounds,
            "num_rbs": num_rbs,
            "local_epochs": local_epochs,
            "optimizer": optimizer_name,
            "activation": activation,
            "learning_rate": learning_rate,
            "quantization_bits": quantization_bits,
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
            "loss_function": regression_loss,
            "optimal_resource_search": "heuristic",
        }
    }
    if optimizer_name == "sgd":
        result["hyperparameters"]["sgd_batch_size"] = int(first_candidate(cfg["training"]["sgd_batch_size"]))
    torch.manual_seed(seed)
    initial_model = RegressionFNN(activation=activation)
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    optimal_model = RegressionFNN(activation=activation)
    optimal_model.load_state_dict(copy.deepcopy(initial_state))
    optimal = proposed_algorithm(
        data["users"],
        optimal_model,
        task="regression",
        test_data=test_data,
        rounds=rounds,
        num_rbs=num_rbs,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        resource_search="heuristic",
        seed=seed,
        config=run_config,
    )
    result["optimal"] = {
        "loss": optimal["metrics"]["loss"][-1],
        "selected": len(optimal["selected_users"]),
        "solver_iterations": optimal["solver_iterations"],
        "model_state": optimal["model_state"],
    }
    for name, runner in {"proposed": proposed_algorithm, "baseline_a": baseline_a, "baseline_b": baseline_b}.items():
        model_instance = RegressionFNN(activation=activation)
        model_instance.load_state_dict(copy.deepcopy(initial_state))
        output = runner(
            data["users"],
            model_instance,
            task="regression",
            test_data=test_data,
            rounds=rounds,
            num_rbs=num_rbs,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            seed=seed,
            config=run_config,
        )
        result[name] = {"loss": output["metrics"]["loss"][-1], "selected": len(output["selected_users"]), "model_state": output["model_state"]}

    xs = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        result["x_grid"] = xs.numpy().ravel()
        result["noise_free_prediction"] = (-2.0 * xs + 1.0).numpy().ravel()
        result["samples"] = (data["x"].numpy().ravel(), data["y"].numpy().ravel())
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(result["samples"][0], result["samples"][1], marker="x", color="red", label="Data samples")
        for key, label, color, linestyle, linewidth in [
            ("proposed", "Proposed algorithm", "blue", "-", 2.0),
            ("optimal", "Optimal FL", "magenta", (0, (5, 5)), 2.0),
            ("baseline_a", "Baseline a)", "black", "-", 2.0),
            ("baseline_b", "Baseline b)", "limegreen", "-", 2.0),
        ]:
            fitted = RegressionFNN(activation=activation).to(dtype=compute_dtype)
            fitted.load_state_dict(result[key]["model_state"])
            with torch.no_grad():
                prediction = fitted(xs.to(dtype=compute_dtype)).float().numpy().ravel()
            ax.plot(result["x_grid"], prediction, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        ax.set_xlabel("Input of the FL algorithm")
        ax.set_ylabel("Output of the FL algorithm")
        ax.set_xlim(0, 1)
        ax.set_ylim(-2, 2)
        ax.set_yticks(np.arange(-2, 3, 1))
        ax.legend()
        result["figure"] = fig
    for key in ["proposed", "optimal", "baseline_a", "baseline_b"]:
        result[key].pop("model_state", None)
    return result


def figure_4(contexts):
    # Fig. 4 varies per-user regression samples and reports final training loss.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_4"]
    if len(contexts) > 1:
        merged = copy.deepcopy(cfg)
        merged_cfg = merged["figures"]["figure_4"]
        merged_cfg["sample_counts"] = sorted({int(item.loss["figures"]["figure_4"]["sample_counts"]) for item in contexts})
        merged["_run_seeds"] = sorted({int(item.seed) for item in contexts})
        return figure_4([Context(seed=seed, loss=merged)])

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    sample_counts_raw = figure_cfg["sample_counts"]
    if isinstance(sample_counts_raw, list) and sample_counts_raw and isinstance(sample_counts_raw[0], list):
        sample_counts_raw = sample_counts_raw[0]
    sample_counts = [int(value) for value in sample_counts_raw] if isinstance(sample_counts_raw, list) else [int(sample_counts_raw)]
    if not sample_counts:
        raise ValueError("figure_4.sample_counts must not be empty")
    rounds = int(first_candidate(figure_cfg["rounds"]))
    num_rbs = int(first_candidate(figure_cfg["num_rbs"]))
    local_epochs = int(first_candidate(figure_cfg["local_epochs"]))
    activation = str(first_candidate(figure_cfg["activation"]))
    noise_std = 0.4
    learning_rate = float(first_candidate(figure_cfg["learning_rate"]))
    optimizer_name = str(first_candidate(cfg["training"]["optimizer"])).lower().replace("-", "_")
    if optimizer_name in {"gd", "full_batch_gradient_descent"}:
        optimizer_name = "gradient_descent"
    if optimizer_name in {"stochastic_gradient_descent", "mini_batch_sgd"}:
        optimizer_name = "sgd"
    if optimizer_name in {"scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        optimizer_name = "trainscg"
    eval_batch_size = int(first_candidate(cfg["training"]["eval_batch_size"]))
    regression_loss = str(first_candidate(cfg["training"]["regression_loss"])).lower()
    quantization_bits = int(first_candidate(cfg["training"]["quantization_bits"]))
    if quantization_bits == 16:
        compute_dtype = torch.float16
    elif quantization_bits == 32:
        compute_dtype = torch.float32
    elif quantization_bits == 64:
        compute_dtype = torch.float64
    else:
        raise ValueError("training.quantization_bits supports actual compute dtype only for 16, 32, or 64 bits")
    if regression_loss not in {"mse", "nmse"}:
        raise ValueError("regression_loss must be 'mse' or 'nmse'")
    run_config = copy.deepcopy(cfg)
    run_config["training"]["regression_loss"] = regression_loss
    run_seeds = [int(value) for value in cfg["_run_seeds"]]

    curves_by_seed = {"proposed": [], "baseline_a": [], "baseline_b": []}
    data_seeds_by_seed = {}
    runners = {"proposed": proposed_algorithm, "baseline_a": baseline_a, "baseline_b": baseline_b}
    for run_seed in run_seeds:
        # Independent data seeds avoid reusing easier/harder samples across x-values.
        data_seed_rng = np.random.default_rng(run_seed)
        data_seeds = [int(data_seed_rng.integers(0, 2**32 - 1)) for _ in sample_counts]
        data_seeds_by_seed[int(run_seed)] = data_seeds
        seed_curves = {name: [] for name in curves_by_seed}
        for count, data_seed in zip(sample_counts, data_seeds):
            data = generate_synthetic_data(num_users=15, samples_per_user=count, seed=data_seed, noise_std=noise_std)
            train_data = TensorDataset(data["x"], data["y"])
            torch.manual_seed(run_seed)
            initial_model = RegressionFNN(activation=activation)
            initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
            for curve_name, runner in runners.items():
                model_instance = RegressionFNN(activation=activation)
                model_instance.load_state_dict(copy.deepcopy(initial_state))
                output = runner(
                    data["users"],
                    model_instance,
                    task="regression",
                    test_data=train_data,
                    rounds=rounds,
                    num_rbs=num_rbs,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    seed=run_seed,
                    config=run_config,
                )
                fitted = RegressionFNN(activation=activation).to(dtype=compute_dtype)
                fitted.load_state_dict(output["model_state"])
                fitted.eval()
                display_loss = 0.0
                display_total = 0
                with torch.no_grad():
                    for features, labels in DataLoader(train_data, batch_size=eval_batch_size, shuffle=False):
                        prediction = fitted(features.to(dtype=compute_dtype))
                        target = labels.to(dtype=prediction.dtype).reshape_as(prediction)
                        display_loss += float((prediction - target).pow(2).sum().item())
                        display_total += len(features)
                seed_curves[curve_name].append(display_loss / max(display_total, 1))
        for curve_name, values in seed_curves.items():
            curves_by_seed[curve_name].append(values)
    curves = {name: np.mean(values, axis=0).tolist() for name, values in curves_by_seed.items()}
    result = {
        "samples_per_user": sample_counts,
        "loss": curves,
        "hyperparameters": {
            "sample_counts": sample_counts,
            "rounds": rounds,
            "num_rbs": num_rbs,
            "local_epochs": local_epochs,
            "optimizer": optimizer_name,
            "activation": activation,
            "noise_std": noise_std,
            "learning_rate": learning_rate,
            "quantization_bits": quantization_bits,
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
            "training_loss_function": regression_loss,
            "loss_function": "mse",
            "loss_source": "training_mse_after_training",
            "data_sampling": "independent_per_count",
            "data_seeds_by_seed": data_seeds_by_seed,
            "seeds": run_seeds,
            "seed_runs": len(run_seeds),
        },
    }
    if optimizer_name == "sgd":
        result["hyperparameters"]["sgd_batch_size"] = int(first_candidate(cfg["training"]["sgd_batch_size"]))
    if plot:
        fig, ax = plt.subplots()
        ax.plot(sample_counts, curves["proposed"], color="blue", marker="o", linewidth=2.5, label="Proposed algorithm")
        ax.plot(sample_counts, curves["baseline_a"], color="red", linestyle=(0, (6, 4)), linewidth=2.5, label="Baseline a)")
        ax.plot(sample_counts, curves["baseline_b"], color="black", marker="s", linestyle=(0, (6, 4)), linewidth=2.5, label="Baseline b)")
        ax.set_xlabel("Number of data samples per user")
        ax.set_ylabel("Value of the loss function")
        ax.set_xticks(sample_counts)
        loss_values = [value for values in curves.values() for value in values]
        y_min = math.floor(min(loss_values) / 0.005) * 0.005
        y_max = math.ceil(max(loss_values) / 0.005) * 0.005
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.0025, 0.005))
        ax.legend()
        result["figure"] = fig
    return result


def figure_5(contexts):
    # Fig. 5 measures assignment complexity as user count grows.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_5"]
    if len(contexts) > 1:
        merged = copy.deepcopy(cfg)
        merged_cfg = merged["figures"]["figure_5"]
        merged_cfg["user_counts"] = sorted({int(item.loss["figures"]["figure_5"]["user_counts"]) for item in contexts})
        merged_cfg["rb_counts"] = sorted({int(item.loss["figures"]["figure_5"]["rb_counts"]) for item in contexts})
        merged["_run_seeds"] = sorted({int(item.seed) for item in contexts})
        return figure_5([Context(seed=seed, loss=merged)])

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    users_raw = figure_cfg["user_counts"]
    if isinstance(users_raw, list) and users_raw and isinstance(users_raw[0], list):
        users_raw = users_raw[0]
    users = [int(value) for value in users_raw] if isinstance(users_raw, list) else [int(users_raw)]
    rb_counts_raw = figure_cfg["rb_counts"]
    if isinstance(rb_counts_raw, list) and rb_counts_raw and isinstance(rb_counts_raw[0], list):
        rb_counts_raw = rb_counts_raw[0]
    rb_counts = [int(value) for value in rb_counts_raw] if isinstance(rb_counts_raw, list) else [int(rb_counts_raw)]
    run_seeds = [int(value) for value in cfg["_run_seeds"]]

    curves_by_seed = {rb_count: [] for rb_count in rb_counts}
    timings_by_seed = {rb_count: [] for rb_count in rb_counts}
    for run_seed in run_seeds:
        for rb_count in rb_counts:
            seed_curve = []
            seed_timings = []
            for user_count in users:
                started = time.perf_counter()
                output = proposed_algorithm(num_users=user_count, num_rbs=rb_count, seed=run_seed, config=cfg)
                seed_timings.append(time.perf_counter() - started)
                seed_curve.append(output["solver_iterations"])
            curves_by_seed[rb_count].append(seed_curve)
            timings_by_seed[rb_count].append(seed_timings)
    curves = {rb_count: np.mean(values, axis=0).tolist() for rb_count, values in curves_by_seed.items()}
    timings = {rb_count: np.mean(values, axis=0).tolist() for rb_count, values in timings_by_seed.items()}
    result = {
        "users": users,
        "iterations": curves,
        "edge_weight_evaluations": curves,
        "seconds": timings,
        "hyperparameters": {"user_counts": users, "rb_counts": rb_counts, "seeds": run_seeds, "seed_runs": len(run_seeds)},
    }
    if plot:
        fig, ax = plt.subplots()
        colors = ["blue", "black", "red", "green"]
        markers = ["o", "s", "^", "d"]
        for idx, rb_count in enumerate(rb_counts):
            linestyle = "-" if idx == 0 else (0, (6, 4))
            ax.plot(users, curves[rb_count], color=colors[idx % len(colors)], marker=markers[idx % len(markers)], linestyle=linestyle, linewidth=2.5, label=rf"$R={rb_count}$")
        ax.set_xlabel("Number of users")
        ax.set_ylabel("Number of iterations")
        curve_values = [value for values in curves.values() for value in values]
        ax.set_yticks(np.arange(0, math.ceil(max(curve_values) / 20) * 20 + 20, 20))
        ax.legend()
        result["figure"] = fig
    return result


def figure_6(contexts):
    # Fig. 6 compares Theorem-1 convergence-gap bound with Monte Carlo packet loss.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_6"]
    if len(contexts) > 1:
        merged = copy.deepcopy(cfg)
        merged_cfg = merged["figures"]["figure_6"]
        merged_cfg["user_counts"] = sorted({int(item.loss["figures"]["figure_6"]["user_counts"]) for item in contexts})
        merged["_run_seeds"] = sorted({int(item.seed) for item in contexts})
        return figure_6([Context(seed=seed, loss=merged)])
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    users_raw = figure_cfg["user_counts"]
    if isinstance(users_raw, list) and users_raw and isinstance(users_raw[0], list):
        users_raw = users_raw[0]
    users = [int(value) for value in users_raw] if isinstance(users_raw, list) else [int(users_raw)]
    if not users:
        raise ValueError("figure_6.user_counts must not be empty")
    num_rbs = int(first_candidate(figure_cfg["num_rbs"]))
    samples_raw = figure_cfg["samples_per_user"]
    if isinstance(samples_raw, list) and samples_raw and isinstance(samples_raw[0], list):
        samples_raw = samples_raw[0]
    samples_per_user = [int(value) for value in samples_raw]
    if len(samples_per_user) < max(users):
        raise ValueError("figure_6.samples_per_user must contain at least max(user_counts) entries")
    model_parameters = int(first_candidate(figure_cfg["model_parameters"]))
    quantization_bits = int(first_candidate(train_cfg["quantization_bits"]))
    simulation_trials = int(first_candidate(figure_cfg["simulation_trials"]))
    rounds = int(first_candidate(figure_cfg["rounds"]))
    mu = float(first_candidate(figure_cfg["strong_convexity_mu"]))
    lipschitz = float(first_candidate(figure_cfg["lipschitz_l"]))
    zeta1 = float(first_candidate(figure_cfg["gradient_bound_zeta1"]))
    zeta2 = float(first_candidate(figure_cfg["gradient_bound_zeta2"]))
    threshold = float(first_candidate(figure_cfg["waterfall_threshold"]))
    if simulation_trials <= 0:
        raise ValueError("figure_6.simulation_trials must be positive")
    if rounds <= 0:
        raise ValueError("figure_6.rounds must be positive")
    if mu <= 0.0 or lipschitz <= 0.0:
        raise ValueError("figure_6 strong_convexity_mu and lipschitz_l must be positive")

    interference_by_rbs = wireless["interference_w_by_rbs"]
    interference_profile = interference_by_rbs.get(str(num_rbs)) if isinstance(interference_by_rbs, dict) else None
    if interference_profile is None and isinstance(interference_by_rbs, dict):
        interference_profile = interference_by_rbs.get(num_rbs)
    if interference_profile is None:
        interference_profile = [float(wireless["interference_w"])] * num_rbs
    interference = np.maximum(np.asarray(interference_profile, dtype=np.float64), 0.0)
    if interference.size != num_rbs:
        raise ValueError(f"wireless.interference_w_by_rbs[{num_rbs}] must contain {num_rbs} values")

    radius = float(wireless["radius_m"])
    power = float(wireless["pmax_w"])
    noise_w = 1e-14
    downlink_interference = float(wireless["downlink_interference_w"])
    downlink_bandwidth_mhz = float(wireless["downlink_bandwidth_hz"]) / 1e6
    model_megabits = model_parameters * quantization_bits / 1024.0 / 1024.0
    local_training_energy = (
        float(wireless["energy_coefficient"])
        * float(wireless["cpu_cycles_per_bit"])
        * float(wireless["cpu_frequency_hz"]) ** 2
        * model_megabits
    )
    delay_requirement = float(wireless["delay_s"])
    energy_requirement = float(wireless["energy_j"])
    run_seeds = [int(value) for value in cfg["_run_seeds"]]

    def simulate_packet_error_bound(counts, selected_user_ids, selected_error_values, packet_rng):
        # This simulates only the wireless-miss term in Theorem 1, not full MNIST training.
        total = float(np.sum(counts))
        selected_user_ids = np.asarray(selected_user_ids, dtype=np.int64)
        selected_error_values = np.asarray(selected_error_values, dtype=np.float64)
        gaps = np.zeros(simulation_trials, dtype=np.float64)
        if selected_user_ids.size:
            selected_counts = counts[selected_user_ids].astype(np.float64)
        else:
            selected_counts = np.asarray([], dtype=np.float64)
        for _round_index in range(rounds):
            if selected_user_ids.size:
                successes = packet_rng.random((simulation_trials, selected_user_ids.size)) > selected_error_values[None, :]
                missed_weights = total - successes @ selected_counts
            else:
                missed_weights = np.full(simulation_trials, total, dtype=np.float64)
            miss_ratios = missed_weights / total
            contractions = 1.0 - mu / lipschitz + 4.0 * mu * zeta2 * miss_ratios / lipschitz
            increments = 2.0 * zeta1 * miss_ratios / lipschitz
            gaps = contractions * gaps + increments
        return float(np.mean(gaps))

    theoretical_by_seed = []
    bound_simulated_by_seed = []
    selected_by_seed = []
    miss_ratio_by_seed = []
    for run_seed in run_seeds:
        rng = np.random.default_rng(run_seed)
        distance_pool = np.maximum(rng.random(max(users)) * radius, np.finfo(np.float64).tiny)
        theoretical_seed = []
        bound_simulated_seed = []
        selected_seed = []
        miss_ratio_seed = []
        for user_count in users:
            counts = np.asarray(samples_per_user[:user_count], dtype=np.float64)
            total = float(np.sum(counts))
            distances = distance_pool[:user_count]
            path_gain = distances[:, None] ** (-float(wireless["path_loss_alpha"]))
            denominator = interference[None, :] + noise_w
            packet_errors = 1.0 - np.exp(-threshold * denominator / np.maximum(power * path_gain, np.finfo(np.float64).tiny))
            packet_errors = np.clip(packet_errors, 0.0, 1.0)
            uplink_rates = np.log2(1.0 + power * path_gain / np.maximum(denominator, np.finfo(np.float64).tiny))
            downlink_rates = downlink_bandwidth_mhz * np.log2(
                1.0 + distances ** (-float(wireless["path_loss_alpha"])) / max(downlink_interference, np.finfo(np.float64).tiny)
            )
            uplink_delays = model_megabits / np.maximum(uplink_rates, np.finfo(np.float64).tiny)
            downlink_delays = model_megabits / np.maximum(downlink_rates, np.finfo(np.float64).tiny)
            total_delays = uplink_delays + downlink_delays[:, None]
            energies = local_training_energy + power * uplink_delays
            feasible = (total_delays < delay_requirement) & (energies < energy_requirement)
            weights = np.where(feasible, counts[:, None] * (packet_errors - 1.0), 0.0)
            rows, cols = linear_sum_assignment(weights)
            final_errors = np.ones(user_count, dtype=np.float64)
            selected_users = []
            for row, col in zip(rows, cols):
                if feasible[row, col] and weights[row, col] < 0.0:
                    final_errors[row] = packet_errors[row, col]
                    selected_users.append(int(row))
            selected_users_array = np.asarray(selected_users, dtype=np.int64)
            selected_errors = final_errors[selected_users_array] if selected_users_array.size else np.asarray([], dtype=np.float64)

            miss_ratio = float(np.sum(counts * final_errors) / total)
            contraction = 1.0 - mu / lipschitz + 4.0 * mu * zeta2 * miss_ratio / lipschitz
            gap = (2.0 * zeta1 * miss_ratio / lipschitz) / max(1.0 - contraction, 1e-9)
            theoretical_seed.append(gap)
            selected_seed.append(len(selected_users))
            miss_ratio_seed.append(miss_ratio)
            packet_rng = np.random.default_rng(run_seed * 1000003 + user_count * 7919 + 101)
            bound_simulated_seed.append(simulate_packet_error_bound(counts, selected_users_array, selected_errors, packet_rng))
        theoretical_by_seed.append(theoretical_seed)
        bound_simulated_by_seed.append(bound_simulated_seed)
        selected_by_seed.append(selected_seed)
        miss_ratio_by_seed.append(miss_ratio_seed)

    theoretical = np.mean(theoretical_by_seed, axis=0).tolist()
    simulated = np.mean(bound_simulated_by_seed, axis=0).tolist()
    result = {
        "users": users,
        "theoretical_gap": theoretical,
        "simulation_gap": simulated,
        "selected_users": np.mean(selected_by_seed, axis=0).tolist(),
        "miss_ratio": np.mean(miss_ratio_by_seed, axis=0).tolist(),
        "hyperparameters": {
            "user_counts": users,
            "num_rbs": num_rbs,
            "samples_per_user": samples_per_user[: max(users)],
            "model_parameters": model_parameters,
            "quantization_bits": quantization_bits,
            "model_megabits": model_megabits,
            "waterfall_threshold": threshold,
            "simulation_trials": simulation_trials,
            "rounds": rounds,
            "strong_convexity_mu": mu,
            "lipschitz_l": lipschitz,
            "gradient_bound_zeta1": zeta1,
            "gradient_bound_zeta2": zeta2,
            "seeds": run_seeds,
            "seed_runs": len(run_seeds),
            "wireless_model": "docs/Wireless-FL/FLMIN.m",
        },
    }
    if plot:
        fig, ax = plt.subplots()
        ax.plot(users, theoretical, color="blue", marker="o", linewidth=2.5, label="Theoretical analysis")
        ax.plot(users, simulated, color="black", marker="s", linestyle=(0, (6, 4)), linewidth=2.5, label="Simulation result")
        ax.set_xlabel("Number of users")
        ax.set_ylabel("Convergence gap due to wireless factors")
        ax.set_xticks(users)
        ax.legend(loc="upper left")
        result["figure"] = fig
    return result


def figure_7(contexts):
    # Fig. 7 tracks MNIST identification accuracy over FL communication rounds.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_7"]
    if len(contexts) > 1:
        merged = copy.deepcopy(cfg)
        merged_cfg = merged["figures"]["figure_7"]
        merged_cfg["rounds"] = max(int(item.loss["figures"]["figure_7"]["rounds"]) for item in contexts)
        merged["_run_seeds"] = sorted({int(item.seed) for item in contexts})
        return figure_7([Context(seed=seed, loss=merged)])

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    samples_raw = figure_cfg["samples_per_user"]
    if isinstance(samples_raw, list) and samples_raw and isinstance(samples_raw[0], list):
        samples_raw = samples_raw[0]
    samples_per_user = [int(value) for value in samples_raw]
    num_users = len(samples_per_user)
    test_samples = int(first_candidate(figure_cfg["test_samples"]))
    rounds = int(first_candidate(figure_cfg["rounds"]))
    num_rbs = int(first_candidate(figure_cfg["num_rbs"]))
    local_epochs = int(first_candidate(figure_cfg["local_epochs"]))
    learning_rate = float(first_candidate(figure_cfg["learning_rate"]))
    quantization_bits = int(first_candidate(cfg["training"]["quantization_bits"]))
    if quantization_bits == 16:
        compute_dtype = torch.float16
    elif quantization_bits == 32:
        compute_dtype = torch.float32
    elif quantization_bits == 64:
        compute_dtype = torch.float64
    else:
        raise ValueError("training.quantization_bits supports actual compute dtype only for 16, 32, or 64 bits")
    optimizer_name = str(cfg["training"].get("optimizer", "gradient_descent")).lower().replace("-", "_")
    if optimizer_name in {"gd", "full_batch_gradient_descent"}:
        optimizer_name = "gradient_descent"
    if optimizer_name in {"stochastic_gradient_descent", "mini_batch_sgd"}:
        optimizer_name = "sgd"
    if optimizer_name in {"scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        optimizer_name = "trainscg"
    run_seeds = [int(value) for value in cfg["_run_seeds"]]

    curves = {"proposed": [], "baseline_a": [], "baseline_b": [], "baseline_c": []}
    runners = {"proposed": proposed_algorithm, "baseline_a": baseline_a, "baseline_b": baseline_b, "baseline_c": baseline_c}
    for run_seed in run_seeds:
        data = load_mnist_data(
            num_users=num_users,
            samples_per_user=samples_per_user,
            test_samples=test_samples,
            seed=run_seed,
            train_order=cfg["training"]["mnist_train_order"],
            partition_order=cfg["training"]["mnist_partition_order"],
        )
        for name, runner in runners.items():
            output = runner(
                data["users"],
                MNISTFNN,
                task="mnist",
                test_data=data["test"],
                rounds=rounds,
                num_rbs=num_rbs,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                seed=run_seed,
                config=cfg,
            )
            curves[name].append(output["metrics"]["accuracy"])
    curves = {name: np.mean(values, axis=0).tolist() for name, values in curves.items()}
    hyperparameters = {
        "samples_per_user": samples_per_user,
        "test_samples": test_samples,
        "rounds": rounds,
        "num_rbs": num_rbs,
        "local_epochs": local_epochs,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "quantization_bits": quantization_bits,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "seeds": run_seeds,
        "seed_runs": len(run_seeds),
        "baseline_a_mode": cfg["training"]["baseline_a_mode"],
        "baseline_c_mode": cfg["training"]["baseline_c_mode"],
    }
    if optimizer_name == "sgd":
        hyperparameters.update({"sgd_batch_size": int(cfg["training"]["sgd_batch_size"])})
    if optimizer_name == "adam":
        hyperparameters.update(
            {
                "adam_beta1": float(cfg["training"]["adam_beta1"]),
                "adam_beta2": float(cfg["training"]["adam_beta2"]),
                "adam_eps": float(cfg["training"]["adam_eps"]),
                "adam_weight_decay": float(cfg["training"]["adam_weight_decay"]),
                "adam_persistent_state": bool(cfg["training"]["adam_persistent_state"]),
            }
        )
    if optimizer_name == "lbfgs":
        hyperparameters.update(
            {
                "lbfgs_max_iter": int(cfg["training"]["lbfgs_max_iter"]),
                "lbfgs_max_eval": int(cfg["training"]["lbfgs_max_eval"]),
                "lbfgs_tolerance_grad": float(cfg["training"]["lbfgs_tolerance_grad"]),
                "lbfgs_tolerance_change": float(cfg["training"]["lbfgs_tolerance_change"]),
                "lbfgs_history_size": int(cfg["training"]["lbfgs_history_size"]),
                "lbfgs_line_search_fn": cfg["training"]["lbfgs_line_search_fn"],
            }
        )
    if optimizer_name == "trainscg":
        hyperparameters.update(
            {
                "trainscg_sigma": float(cfg["training"]["trainscg_sigma"]),
                "trainscg_lambda": float(cfg["training"]["trainscg_lambda"]),
                "trainscg_lambda_min": float(cfg["training"]["trainscg_lambda_min"]),
                "trainscg_lambda_max": float(cfg["training"]["trainscg_lambda_max"]),
                "trainscg_min_grad": float(cfg["training"]["trainscg_min_grad"]),
            }
        )
    result = {
        "rounds": list(range(1, rounds + 1)),
        "accuracy": curves,
        "hyperparameters": hyperparameters,
    }
    if plot:
        fig, ax = plt.subplots()
        ax.plot(result["rounds"], curves["proposed"], color="black", linewidth=2.0, label="Proposed FL")
        ax.plot(result["rounds"], curves["baseline_a"], color="blue", linestyle=(0, (6, 4)), linewidth=2.0, label="Baseline a)")
        ax.plot(result["rounds"], curves["baseline_b"], color="red", linewidth=2.0, label="Baseline b)")
        ax.plot(result["rounds"], curves["baseline_c"], color="red", linestyle=":", linewidth=2.0, label="Baseline c)")
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Identification accuracy")
        ax.set_xlim(0, rounds)
        accuracy_values = [value for values in curves.values() for value in values]
        y_max = max(0.9, math.ceil(max(accuracy_values) / 0.05) * 0.05)
        ax.set_ylim(0.1, y_max)
        ax.set_xticks(np.arange(0, rounds + 1, 20))
        ax.set_yticks(np.arange(0.1, y_max + 0.01, 0.1))
        ax.legend()
        result["figure"] = fig
    return result


def figure_8(contexts):
    # Fig. 8 varies total users while keeping RBs fixed.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_8"]
    if len(contexts) > 1:
        merged = copy.deepcopy(cfg)
        merged_cfg = merged["figures"]["figure_8"]
        merged_cfg["user_counts"] = sorted({int(item.loss["figures"]["figure_8"]["user_counts"]) for item in contexts})
        merged["_run_seeds"] = sorted({int(item.seed) for item in contexts})
        return figure_8([Context(seed=seed, loss=merged)])

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    user_counts_raw = figure_cfg["user_counts"]
    if isinstance(user_counts_raw, list) and user_counts_raw and isinstance(user_counts_raw[0], list):
        user_counts_raw = user_counts_raw[0]
    user_counts = [int(value) for value in user_counts_raw] if isinstance(user_counts_raw, list) else [int(user_counts_raw)]
    samples_raw = figure_cfg["samples_per_user"]
    if isinstance(samples_raw, list) and samples_raw and isinstance(samples_raw[0], list):
        samples_raw = samples_raw[0]
    sample_pool = [int(value) for value in samples_raw]
    if max(user_counts) > len(sample_pool):
        raise ValueError("figure_8.samples_per_user must cover the largest figure_8.user_counts value")
    test_samples = int(first_candidate(figure_cfg["test_samples"]))
    rounds = int(first_candidate(figure_cfg["rounds"]))
    num_rbs = int(first_candidate(figure_cfg["num_rbs"]))
    local_epochs = int(first_candidate(figure_cfg["local_epochs"]))
    learning_rate = float(first_candidate(figure_cfg["learning_rate"]))
    optimizer_name = str(first_candidate(cfg["training"]["optimizer"])).lower().replace("-", "_")
    if optimizer_name in {"gd", "full_batch_gradient_descent"}:
        optimizer_name = "gradient_descent"
    if optimizer_name in {"stochastic_gradient_descent", "mini_batch_sgd"}:
        optimizer_name = "sgd"
    if optimizer_name in {"scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        optimizer_name = "trainscg"
    run_seeds = [int(value) for value in cfg["_run_seeds"]]

    curves_by_seed = {"proposed": [], "baseline_a": [], "baseline_b": [], "baseline_c": []}
    runners = {"proposed": proposed_algorithm, "baseline_a": baseline_a, "baseline_b": baseline_b, "baseline_c": baseline_c}
    for run_seed in run_seeds:
        shared = load_mnist_data(
            num_users=max(user_counts),
            samples_per_user=sample_pool[:max(user_counts)],
            test_samples=test_samples,
            seed=run_seed,
            train_order=cfg["training"]["mnist_train_order"],
            partition_order=cfg["training"]["mnist_partition_order"],
        )
        test_data = shared["test"]
        seed_curves = {name: [] for name in curves_by_seed}
        for user_count in user_counts:
            samples = sample_pool[:user_count]
            data = load_mnist_data(
                num_users=user_count,
                samples_per_user=samples,
                test_samples=test_samples,
                seed=run_seed,
                train_order=cfg["training"]["mnist_train_order"],
                partition_order=cfg["training"]["mnist_partition_order"],
            )
            for name, runner in runners.items():
                output = runner(
                    data["users"],
                    MNISTFNN,
                    task="mnist",
                    test_data=test_data,
                    rounds=rounds,
                    num_rbs=num_rbs,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    seed=run_seed,
                    config=cfg,
                )
                seed_curves[name].append(output["metrics"]["accuracy"][-1])
        for name, values in seed_curves.items():
            curves_by_seed[name].append(values)
    curves = {name: np.mean(values, axis=0).tolist() for name, values in curves_by_seed.items()}
    result = {
        "users": user_counts,
        "accuracy": curves,
        "hyperparameters": {
            "user_counts": user_counts,
            "samples_per_user": sample_pool,
            "test_samples": test_samples,
            "rounds": rounds,
            "num_rbs": num_rbs,
            "local_epochs": local_epochs,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "seeds": run_seeds,
            "seed_runs": len(run_seeds),
            "baseline_a_mode": cfg["training"]["baseline_a_mode"],
            "baseline_c_mode": cfg["training"]["baseline_c_mode"],
        },
    }
    if optimizer_name == "sgd":
        result["hyperparameters"]["sgd_batch_size"] = int(cfg["training"]["sgd_batch_size"])
    if plot:
        fig, ax = plt.subplots()
        ax.plot(user_counts, curves["proposed"], color="black", linewidth=2.0, label="Proposed FL")
        ax.plot(user_counts, curves["baseline_a"], color="blue", linestyle=(0, (6, 4)), linewidth=2.0, label="Baseline a)")
        ax.plot(user_counts, curves["baseline_b"], color="red", linewidth=2.0, label="Baseline b)")
        ax.plot(user_counts, curves["baseline_c"], color="red", linestyle=":", linewidth=2.0, label="Baseline c)")
        ax.set_xlabel("Total number of users")
        ax.set_ylabel("Identification accuracy")
        ax.set_xlim(min(user_counts), max(user_counts))
        accuracy_values = [value for values in curves.values() for value in values]
        y_min = math.floor(min(accuracy_values) / 0.02) * 0.02
        y_max = math.ceil(max(accuracy_values) / 0.02) * 0.02
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(user_counts)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.02))
        ax.legend()
        result["figure"] = fig
    return result


def figure_9(contexts):
    # Fig. 9 varies the number of uplink RBs for a fixed MNIST user pool.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_9"]
    if len(contexts) > 1:
        merged = copy.deepcopy(cfg)
        merged_cfg = merged["figures"]["figure_9"]
        merged_cfg["rb_counts"] = sorted({int(item.loss["figures"]["figure_9"]["rb_counts"]) for item in contexts})
        merged["_run_seeds"] = sorted({int(item.seed) for item in contexts})
        return figure_9([Context(seed=seed, loss=merged)])

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    rb_counts_raw = figure_cfg["rb_counts"]
    if isinstance(rb_counts_raw, list) and rb_counts_raw and isinstance(rb_counts_raw[0], list):
        rb_counts_raw = rb_counts_raw[0]
    rb_counts = [int(value) for value in rb_counts_raw] if isinstance(rb_counts_raw, list) else [int(rb_counts_raw)]
    samples_raw = figure_cfg["samples_per_user"]
    if isinstance(samples_raw, list) and samples_raw and isinstance(samples_raw[0], list):
        samples_raw = samples_raw[0]
    samples_per_user = [int(value) for value in samples_raw]
    test_samples = int(first_candidate(figure_cfg["test_samples"]))
    rounds = int(first_candidate(figure_cfg["rounds"]))
    local_epochs = int(first_candidate(figure_cfg["local_epochs"]))
    learning_rate = float(first_candidate(figure_cfg["learning_rate"]))
    optimizer_name = str(first_candidate(cfg["training"]["optimizer"])).lower().replace("-", "_")
    if optimizer_name in {"gd", "full_batch_gradient_descent"}:
        optimizer_name = "gradient_descent"
    if optimizer_name in {"stochastic_gradient_descent", "mini_batch_sgd"}:
        optimizer_name = "sgd"
    if optimizer_name in {"scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        optimizer_name = "trainscg"
    runner_seed_mode = str(first_candidate(figure_cfg["runner_seed_mode"])).lower()
    if runner_seed_mode not in {"same", "rb_offset"}:
        raise ValueError("figure_9.runner_seed_mode must be 'same' or 'rb_offset'")
    run_seeds = [int(value) for value in cfg["_run_seeds"]]

    curves_by_seed = {"proposed": [], "baseline_a": [], "baseline_b": [], "baseline_c": []}
    runners = {"proposed": proposed_algorithm, "baseline_a": baseline_a, "baseline_b": baseline_b, "baseline_c": baseline_c}
    for run_seed in run_seeds:
        data = load_mnist_data(
            num_users=len(samples_per_user),
            samples_per_user=samples_per_user,
            test_samples=test_samples,
            seed=run_seed,
            train_order=cfg["training"]["mnist_train_order"],
            partition_order=cfg["training"]["mnist_partition_order"],
        )
        seed_curves = {name: [] for name in curves_by_seed}
        for rb_count in rb_counts:
            for name, runner in runners.items():
                runner_seed = run_seed if runner_seed_mode == "same" else run_seed * 1009 + int(rb_count)
                output = runner(
                    data["users"],
                    MNISTFNN,
                    task="mnist",
                    test_data=data["test"],
                    rounds=rounds,
                    num_rbs=rb_count,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    seed=runner_seed,
                    config=cfg,
                )
                seed_curves[name].append(output["metrics"]["accuracy"][-1])
        for name, values in seed_curves.items():
            curves_by_seed[name].append(values)
    curves = {name: np.mean(values, axis=0).tolist() for name, values in curves_by_seed.items()}
    result = {
        "rbs": rb_counts,
        "accuracy": curves,
        "hyperparameters": {
            "rb_counts": rb_counts,
            "samples_per_user": samples_per_user,
            "test_samples": test_samples,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "runner_seed_mode": runner_seed_mode,
            "seeds": run_seeds,
            "seed_runs": len(run_seeds),
            "baseline_a_mode": cfg["training"]["baseline_a_mode"],
            "baseline_c_mode": cfg["training"]["baseline_c_mode"],
            "baseline_c_assignment": "wireless_fl_munkres_then_random_rb" if str(cfg["training"]["baseline_c_mode"]).lower() == "wireless_fl" else "ki_agnostic_hungarian",
            "baseline_c_aggregation": "fedavg",
            "channel_model": cfg["wireless"]["channel_model"],
            "mnist_activation": cfg["training"]["mnist_activation"],
            "mnist_train_order": cfg["training"]["mnist_train_order"],
            "mnist_partition_order": cfg["training"]["mnist_partition_order"],
            "force_first_round_success": cfg["training"]["force_first_round_success"],
        },
    }
    if optimizer_name == "sgd":
        result["hyperparameters"]["sgd_batch_size"] = int(cfg["training"]["sgd_batch_size"])
    if plot:
        fig, ax = plt.subplots()
        ax.plot(rb_counts, curves["proposed"], color="black", linewidth=2.0, label="Proposed FL")
        ax.plot(rb_counts, curves["baseline_a"], color="blue", linestyle=(0, (6, 4)), linewidth=2.0, label="Baseline a)")
        ax.plot(rb_counts, curves["baseline_b"], color="red", linewidth=2.0, label="Baseline b)")
        ax.plot(rb_counts, curves["baseline_c"], color="red", linestyle=":", linewidth=2.0, label="Baseline c)")
        ax.set_xlabel("Number of RBs")
        ax.set_ylabel("Identification accuracy")
        ax.set_xlim(min(rb_counts), max(rb_counts))
        accuracy_values = [value for values in curves.values() for value in values]
        y_min = math.floor(min(accuracy_values) / 0.02) * 0.02
        y_max = math.ceil(max(accuracy_values) / 0.02) * 0.02
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(rb_counts)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.02))
        ax.legend()
        result["figure"] = fig
    return result


def figure_10(contexts):
    # Fig. 10 uses CNN predictions on a 6x6 MNIST sample grid.
    context = contexts[0]
    seed = int(context.seed)
    cfg = context.loss
    plot = bool(cfg["_plot"])
    figure_cfg = cfg["figures"]["figure_10"]

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    samples_per_user = int(first_candidate(figure_cfg["samples_per_user"]))
    test_samples = int(first_candidate(figure_cfg["test_samples"]))
    rounds = int(first_candidate(figure_cfg["rounds"]))
    num_rbs = int(first_candidate(figure_cfg["num_rbs"]))
    local_epochs = int(first_candidate(figure_cfg["local_epochs"]))
    learning_rate = float(first_candidate(figure_cfg["learning_rate"]))
    optimizer_name = str(first_candidate(cfg["training"]["optimizer"])).lower().replace("-", "_")
    if optimizer_name in {"gd", "full_batch_gradient_descent"}:
        optimizer_name = "gradient_descent"
    if optimizer_name in {"stochastic_gradient_descent", "mini_batch_sgd"}:
        optimizer_name = "sgd"
    if optimizer_name in {"scg", "scaled_conjugate_gradient", "matlab_trainscg"}:
        optimizer_name = "trainscg"
    data = load_mnist_data(
        num_users=15,
        samples_per_user=samples_per_user,
        test_samples=test_samples,
        seed=seed,
        train_order=cfg["training"]["mnist_train_order"],
        partition_order=cfg["training"]["mnist_partition_order"],
    )
    if len(data["test"]) != 36:
        raise ValueError("Fig. 10 must evaluate exactly 36 MNIST samples")
    torch.manual_seed(seed)
    initial_model = MNISTCNN()
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    proposed_model = MNISTCNN()
    proposed_model.load_state_dict(copy.deepcopy(initial_state))
    baseline_model = MNISTCNN()
    baseline_model.load_state_dict(copy.deepcopy(initial_state))
    proposed = proposed_algorithm(
        data["users"],
        proposed_model,
        task="mnist",
        test_data=data["test"],
        rounds=rounds,
        num_rbs=num_rbs,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        seed=seed,
        config=cfg,
    )
    baseline = baseline_b(
        data["users"],
        baseline_model,
        task="mnist",
        test_data=data["test"],
        rounds=rounds,
        num_rbs=num_rbs,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        seed=seed,
        config=cfg,
    )
    images, labels = data["test"].tensors
    proposed_state = proposed["best_model_state"] if proposed.get("best_model_state") is not None else proposed["model_state"]
    baseline_state = baseline["best_model_state"] if baseline.get("best_model_state") is not None else baseline["model_state"]
    proposed_model = MNISTCNN()
    proposed_model.load_state_dict(proposed_state)
    baseline_model = MNISTCNN()
    baseline_model.load_state_dict(baseline_state)
    proposed_model.eval()
    baseline_model.eval()
    with torch.no_grad():
        proposed_predictions = proposed_model(images).argmax(dim=1)
        baseline_predictions = baseline_model(images).argmax(dim=1)
    proposed_correct = int((proposed_predictions == labels).sum().item())
    baseline_correct = int((baseline_predictions == labels).sum().item())
    n_examples = len(data["test"])
    result = {
        "proposed_accuracy": proposed_correct / max(n_examples, 1),
        "baseline_b_accuracy": baseline_correct / max(n_examples, 1),
        "proposed_correct": proposed_correct,
        "baseline_b_correct": baseline_correct,
        "proposed_best_round": proposed["best_round"],
        "baseline_b_best_round": baseline["best_round"],
        "n_examples": n_examples,
        "hyperparameters": {
            "samples_per_user": samples_per_user,
            "test_samples": test_samples,
            "rounds": rounds,
            "num_rbs": num_rbs,
            "local_epochs": local_epochs,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
        },
    }
    if optimizer_name == "sgd":
        result["hyperparameters"]["sgd_batch_size"] = int(cfg["training"]["sgd_batch_size"])
    if plot:
        fig, axes = plt.subplots(6, 6, figsize=(7, 7))
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(images[idx, 0], cmap="gray")
            proposed_color = "black" if int(proposed_predictions[idx]) == int(labels[idx]) else "red"
            baseline_color = "black" if int(baseline_predictions[idx]) == int(labels[idx]) else "red"
            ax.text(0.5, 1.04, str(int(proposed_predictions[idx])), transform=ax.transAxes, color=proposed_color, fontsize=9, fontweight="bold", ha="center", va="bottom")
            ax.text(0.5, -0.08, str(int(baseline_predictions[idx])), transform=ax.transAxes, color=baseline_color, fontsize=9, fontweight="bold", ha="center", va="top")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.89, hspace=0.55, wspace=0.35)
        fig.text(0.08, 0.975, f"Proposed FL: {result['proposed_correct']}", fontsize=10, fontweight="bold", ha="left", va="top")
        fig.text(0.08, 0.948, f"Baseline b): {result['baseline_b_correct']}", fontsize=10, ha="left", va="top")
        result["figure"] = fig
    return result


# main
def main():
    # CLI entry point for paper figure reproduction and YAML sweeps.
    parser = argparse.ArgumentParser(description="Reproduce FL over wireless network experiments from Chen et al. TWC 2021.")
    parser.add_argument("--figure", choices=["3", "4", "5", "6", "7", "8", "9", "10", "all"], default="all")
    parser.add_argument("--config", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Write detailed per-round FL training diagnostics to OUTPUT_DIR/verbose_logs/*.txt.")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "font.size": 11,
            "legend.fancybox": False,
            "legend.framealpha": 1.0,
        }
    )

    calls = {
        "3": figure_3,
        "4": figure_4,
        "5": figure_5,
        "6": figure_6,
        "7": figure_7,
        "8": figure_8,
        "9": figure_9,
        "10": figure_10,
    }
    selected = calls if args.figure == "all" else {args.figure: calls[args.figure]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def contexts_for_figure(name):
        config_path = Path(args.config) if args.config is not None else Path("configs") / f"figure_{name}.yaml"
        contexts = [context for context in build_contexts(config_path) if context.loss["_figure"] == name]
        if not contexts:
            raise ValueError(f"{config_path} did not define figure_{name}")
        verbose_log_path = output_dir / "verbose_logs" / f"figure_{name}.txt"
        if args.verbose:
            verbose_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(verbose_log_path, "w", encoding="utf-8") as handle:
                handle.write(f"figure={name}\nconfig={config_path}\n")
        for context in contexts:
            if args.seed is not None:
                context.seed = args.seed
                context.loss["seed"] = args.seed
                context.loss["_run_seeds"] = [args.seed]
                context.loss["_filename_settings"]["seed"] = args.seed
                context.loss["_varied"]["seed"] = args.seed
            context.loss["_plot"] = True
            context.loss["_verbose"] = bool(args.verbose)
            context.loss["_verbose_log_path"] = str(verbose_log_path)
        return contexts

    def value_token(value):
        # Stable filename tokens make output sweeps traceable.
        if isinstance(value, list):
            raw = "_".join(str(item) for item in value)
        else:
            raw = str(value)
        raw = raw.replace("-", "m").replace(".", "p")
        token = "".join(char if char.isalnum() else "_" for char in raw).strip("_")
        if len(token) > 80:
            checksum = sum((index + 1) * ord(char) for index, char in enumerate(token)) % 100000
            token = f"{token[:56].rstrip('_')}_len{len(token)}_sum{checksum}"
        return token or "value"

    def key_token(key):
        token = "".join(char if char.isalnum() else "_" for char in str(key)).strip("_")
        return token or "key"

    def run_token(varied):
        if not varied:
            return "default"
        token = "_".join(f"{key_token(key)}_{value_token(value)}" for key, value in varied.items())
        if len(token) > 180:
            checksum = sum((index + 1) * ord(char) for index, char in enumerate(token)) % 100000
            token = f"{token[:148].rstrip('_')}_len{len(token)}_sum{checksum}"
        return token

    def format_figure(fig, name):
        # Apply a paper-like black-axis/grid style after each figure builds its data.
        for ax in fig.axes:
            if name != "10":
                ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
                ax.tick_params(direction="in", top=True, right=True)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(1.0)
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_edgecolor("black")
                legend.get_frame().set_linewidth(0.8)
                legend.get_frame().set_alpha(1.0)
        if name == "10":
            fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.89, hspace=0.55, wspace=0.35)
        else:
            fig.tight_layout()

    def to_builtin(value):
        if torch.is_tensor(value):
            detached = value.detach().cpu()
            return {
                "shape": list(detached.shape),
                "first": float(detached.reshape(-1)[0].item()) if detached.numel() else None,
            }
        if isinstance(value, dict):
            return {str(key): to_builtin(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_builtin(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    planned = {name: contexts_for_figure(name) for name in selected}
    total_runs = sum(len(contexts) for contexts in planned.values())
    print(f"planned_runs: {total_runs}")
    if total_runs >= 100:
        print(f"warning: YAML expands to {total_runs} runs; narrow candidate lists if this is unintended.")

    merge_axis_keys_by_figure = {
        "4": {"sample_counts"},
        "5": {"user_counts", "rb_counts"},
        "6": {"user_counts"},
        "8": {"user_counts"},
        "9": {"rb_counts"},
    }
    seed_averaged_figures = {"4", "5", "6", "7", "8", "9"}

    for name, func in selected.items():
        config_path = Path(args.config) if args.config is not None else Path("configs") / f"figure_{name}.yaml"
        contexts = planned[name]
        merge_axis_keys = merge_axis_keys_by_figure.get(name, set())
        grouped_contexts = {}
        for context in contexts:
            varied = context.loss.get("_varied", {})
            group_varied = {}
            for key in sorted(varied):
                if key in merge_axis_keys:
                    continue
                if name in seed_averaged_figures and key == "seed":
                    continue
                group_varied[key] = varied[key]
            group_key = tuple((key, value_token(value)) for key, value in group_varied.items())
            if group_key not in grouped_contexts:
                grouped_contexts[group_key] = {"contexts": [], "varied": group_varied}
            grouped_contexts[group_key]["contexts"].append(context)

        if len(grouped_contexts) > 1:
            context_groups = []
            for group_key in sorted(grouped_contexts):
                group = grouped_contexts[group_key]
                context_groups.append((run_token(group["varied"]), group["contexts"], group["varied"]))
        else:
            group_key = next(iter(grouped_contexts))
            group = grouped_contexts[group_key]
            context_groups = [("contexts", group["contexts"], {})]

        for group_index, (group_token, group_contexts, group_varied) in enumerate(context_groups, start=1):
            result = func(group_contexts)
            save_path = None
            settings_path = None
            save_time = datetime.now()
            if "figure" in result:
                fig = result["figure"]
                format_figure(fig, name)
                save_dir = output_dir / f"figure_{name}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_token = save_time.strftime("%Y%m%d_%H%M%S_%f")
                save_path = save_dir / f"{save_token}.png"
                settings_path = save_dir / f"{save_token}.yaml"
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            printable = {"context_count": len(group_contexts)}
            if group_varied:
                printable["group"] = group_varied
            for key, value in result.items():
                if key == "figure":
                    continue
                if isinstance(value, np.ndarray):
                    printable[key] = {"shape": value.shape, "first": float(value.ravel()[0]) if value.size else None}
                elif isinstance(value, tuple) and value and all(isinstance(item, np.ndarray) for item in value):
                    printable[key] = [{"shape": item.shape, "first": float(item.ravel()[0]) if item.size else None} for item in value]
                elif isinstance(value, dict):
                    printable[key] = {}
                    for subkey, subvalue in value.items():
                        if key == "loss" and isinstance(subvalue, list) and subvalue:
                            printable[key][subkey] = subvalue[-1]
                        else:
                            printable[key][subkey] = subvalue
                else:
                    printable[key] = value
            if settings_path is not None:
                settings_payload = {
                    "figure": f"figure_{name}",
                    "generated_at": save_time.isoformat(timespec="microseconds"),
                    "config_path": str(config_path),
                    "output_png": str(save_path),
                    "context_count": len(group_contexts),
                    "group_index": group_index,
                    "group_token": group_token,
                    "group": group_varied,
                    "hyperparameters": result.get("hyperparameters", {}),
                    "summary": printable,
                    "contexts": [context.loss for context in group_contexts],
                }
                with open(settings_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(to_builtin(settings_payload), handle, sort_keys=False, allow_unicode=True)
            label = f"figure_{name}" if len(context_groups) == 1 else f"figure_{name}[{group_token}]"
            print(f"{label}: {printable}")
            if save_path is not None:
                print(f"saved: {save_path}")
                print(f"settings: {settings_path}")


if __name__ == "__main__":
    main()
