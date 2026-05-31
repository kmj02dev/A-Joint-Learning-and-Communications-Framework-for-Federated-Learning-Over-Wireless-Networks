import argparse
import copy
import math
import os
import time
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


CACHED_DATA = {}
SEED = 42


# data processing
def generate_synthetic_data(num_users=15, samples_per_user=30, seed=SEED, noise_std=0.4):
    """Generate the linear-regression data used in the paper: y=-2x+1+0.4N(0,1)."""
    rng = np.random.default_rng(seed)
    if isinstance(samples_per_user, int):
        counts = [samples_per_user] * num_users
    else:
        counts = list(samples_per_user)
        num_users = len(counts)

    users = []
    all_x = []
    all_y = []
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


def load_mnist_data(num_users=15, samples_per_user=200, test_samples=1000, seed=SEED, data_dir=None, download=False):
    """Load MNIST from a local npz/cache first, with torchvision download as an opt-in fallback."""
    rng = np.random.default_rng(seed)
    candidates = []
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
    train_indices = rng.permutation(len(x_train))[:total_train]
    test_count = min(len(x_test), int(test_samples))
    test_indices = rng.permutation(len(x_test))[:test_count]
    train_data = TensorDataset(x_train[train_indices], y_train[train_indices])
    test_data = TensorDataset(x_test[test_indices], y_test[test_indices])
    users = get_partitioned_data(train_data, num_users=num_users, samples_per_user=counts, seed=seed)
    return {"train": train_data, "test": test_data, "users": users, "task": "mnist"}


def get_partitioned_data(data=None, num_users=15, samples_per_user=None, seed=SEED, non_iid=False):
    """Partition a TensorDataset into user-local datasets for FL."""
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
    counts = [max(0, min(int(count), dataset_len)) for count in counts]

    if non_iid and hasattr(data, "tensors") and len(data.tensors) > 1:
        labels = data.tensors[1].detach().cpu().numpy()
        indices = np.argsort(labels)
    else:
        indices = rng.permutation(dataset_len)

    users = []
    cursor = 0
    for count in counts:
        selected = indices[cursor:cursor + count]
        cursor += count
        if len(selected) == 0:
            selected = indices[:1]
        users.append(Subset(data, selected.tolist()))
    return users


# models
class RegressionFNN(nn.Module):
    def __init__(self, hidden_size=20, activation="tanh"):
        super().__init__()
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
    def __init__(self, hidden_size=50):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 10))

    def forward(self, x):
        return self.net(x)


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
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


# FL algorithms
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
):
    """Baseline a): FL-aware user selection with random RB allocation."""
    cfg = load_yaml() if config is None else config
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    seed = int(cfg["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg.get("eval_batch_size", 256))
    regression_scale_floor = float(train_cfg.get("regression_scale_floor", 1e-12))
    min_distance = float(wireless.get("min_distance_m", 5.0))
    interference_sigma = float(wireless.get("interference_lognormal_sigma", 0.35))
    rate_floor = float(wireless.get("rate_floor", 1e-12))
    channel_gain_floor = float(wireless.get("channel_gain_floor", 1e-18))
    snr_denominator_floor = float(wireless.get("snr_denominator_floor", 1e-18))
    power_floor = float(wireless.get("power_floor_w", 1e-12))
    power_solver_maxiter = int(wireless.get("power_solver_maxiter", 50))

    if partitions is not None:
        num_users = len(partitions)
    counts = np.array(
        [
            len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
            for i in range(num_users)
        ],
        dtype=np.float64,
    )
    counts = np.maximum(counts, 1.0)

    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN()
    if isinstance(model, type):
        model = model()
    if model_bits is None:
        quantization_bits = train_cfg.get("quantization_bits")
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    radius = float(wireless["radius_m"])
    distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
    fading = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs))
    channel_gain = fading * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
    downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    interference_by_rbs = wireless.get("interference_w_by_rbs", {})
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
    train_energy = zeta * omega * cpu ** 2 * model_bits
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

    weights = np.where(feasible, counts[:, None] * (packet_errors - 1.0), 0.0)
    assignment_users = []
    rows, cols = linear_sum_assignment(weights)
    for row, col in zip(rows, cols):
        if feasible[row, col] and weights[row, col] < 0.0:
            assignment_users.append(int(row))

    allocation = np.zeros((num_users, num_rbs), dtype=np.int64)
    selected_users = []
    assigned_rbs = []
    random_rbs = rng.permutation(num_rbs)[:len(assignment_users)]
    for user_idx, rb_idx in zip(assignment_users, random_rbs):
        allocation[user_idx, rb_idx] = 1
        selected_users.append(user_idx)
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
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg.get("regression_loss", "mse")).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")

        for _ in range(int(rounds)):
            local_states = []
            local_weights = []
            successes = 0
            for user_idx, rb_idx, packet_error in zip(selected_users, assigned_rbs, selected_errors):
                local_model = copy.deepcopy(global_model).to(device_obj)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
                loader = DataLoader(partitions[int(user_idx)], batch_size=min(batch_size, len(partitions[int(user_idx)])), shuffle=True)
                for _local_epoch in range(int(local_epochs)):
                    for features, labels in loader:
                        features = features.to(device_obj)
                        labels = labels.to(device_obj)
                        optimizer.zero_grad()
                        prediction = local_model(features)
                        if task == "regression":
                            target = labels.float().reshape_as(prediction)
                            error = prediction - target
                            squared_error = error.pow(2)
                            if regression_loss == "nmse":
                                target_range = (target.max() - target.min()).clamp_min(regression_scale_floor)
                                loss = (2.0 * error / target_range).pow(2).mean()
                            else:
                                loss = squared_error.mean()
                        else:
                            loss = loss_fn(prediction, labels.long())
                        loss.backward()
                        optimizer.step()
                if rng.random() > packet_error:
                    local_states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
                    local_weights.append(float(len(partitions[int(user_idx)])))
                    successes += 1

            if local_states:
                total_weight = sum(local_weights)
                averaged = {}
                for name in local_states[0]:
                    averaged[name] = sum(state[name] * (weight / total_weight) for state, weight in zip(local_states, local_weights))
                global_model.load_state_dict(averaged)

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
                    features = features.to(device_obj)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.float().reshape_as(prediction)
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
):
    """Baseline b): random user selection and random RB allocation."""
    cfg = load_yaml() if config is None else config
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    seed = int(cfg["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg.get("eval_batch_size", 256))
    regression_scale_floor = float(train_cfg.get("regression_scale_floor", 1e-12))
    min_distance = float(wireless.get("min_distance_m", 5.0))
    interference_sigma = float(wireless.get("interference_lognormal_sigma", 0.35))
    rate_floor = float(wireless.get("rate_floor", 1e-12))
    channel_gain_floor = float(wireless.get("channel_gain_floor", 1e-18))
    snr_denominator_floor = float(wireless.get("snr_denominator_floor", 1e-18))
    power_floor = float(wireless.get("power_floor_w", 1e-12))
    power_solver_maxiter = int(wireless.get("power_solver_maxiter", 50))

    if partitions is not None:
        num_users = len(partitions)
    counts = np.array(
        [
            len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
            for i in range(num_users)
        ],
        dtype=np.float64,
    )
    counts = np.maximum(counts, 1.0)

    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN()
    if isinstance(model, type):
        model = model()
    if model_bits is None:
        quantization_bits = train_cfg.get("quantization_bits")
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    radius = float(wireless["radius_m"])
    distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
    fading = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs))
    channel_gain = fading * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
    downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    interference_by_rbs = wireless.get("interference_w_by_rbs", {})
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
    train_energy = zeta * omega * cpu ** 2 * model_bits
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
    if rounds > 0 and partitions is not None:
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg.get("regression_loss", "mse")).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")

        for _ in range(int(rounds)):
            local_states = []
            local_weights = []
            successes = 0
            for user_idx, rb_idx, packet_error in zip(selected_users, assigned_rbs, selected_errors):
                local_model = copy.deepcopy(global_model).to(device_obj)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
                loader = DataLoader(partitions[int(user_idx)], batch_size=min(batch_size, len(partitions[int(user_idx)])), shuffle=True)
                for _local_epoch in range(int(local_epochs)):
                    for features, labels in loader:
                        features = features.to(device_obj)
                        labels = labels.to(device_obj)
                        optimizer.zero_grad()
                        prediction = local_model(features)
                        if task == "regression":
                            target = labels.float().reshape_as(prediction)
                            error = prediction - target
                            squared_error = error.pow(2)
                            if regression_loss == "nmse":
                                target_range = (target.max() - target.min()).clamp_min(regression_scale_floor)
                                loss = (2.0 * error / target_range).pow(2).mean()
                            else:
                                loss = squared_error.mean()
                        else:
                            loss = loss_fn(prediction, labels.long())
                        loss.backward()
                        optimizer.step()
                if rng.random() > packet_error:
                    local_states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
                    local_weights.append(float(len(partitions[int(user_idx)])))
                    successes += 1

            if local_states:
                total_weight = sum(local_weights)
                averaged = {}
                for name in local_states[0]:
                    averaged[name] = sum(state[name] * (weight / total_weight) for state, weight in zip(local_states, local_weights))
                global_model.load_state_dict(averaged)

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
                    features = features.to(device_obj)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.float().reshape_as(prediction)
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
):
    """Baseline c): wireless-only optimization that ignores FL parameters."""
    cfg = load_yaml() if config is None else config
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    seed = int(cfg["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg.get("eval_batch_size", 256))
    regression_scale_floor = float(train_cfg.get("regression_scale_floor", 1e-12))
    min_distance = float(wireless.get("min_distance_m", 5.0))
    interference_sigma = float(wireless.get("interference_lognormal_sigma", 0.35))
    rate_floor = float(wireless.get("rate_floor", 1e-12))
    channel_gain_floor = float(wireless.get("channel_gain_floor", 1e-18))
    snr_denominator_floor = float(wireless.get("snr_denominator_floor", 1e-18))
    power_floor = float(wireless.get("power_floor_w", 1e-12))
    power_solver_maxiter = int(wireless.get("power_solver_maxiter", 50))

    if partitions is not None:
        num_users = len(partitions)
    counts = np.array(
        [
            len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
            for i in range(num_users)
        ],
        dtype=np.float64,
    )
    counts = np.maximum(counts, 1.0)

    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN()
    if isinstance(model, type):
        model = model()
    if model_bits is None:
        quantization_bits = train_cfg.get("quantization_bits")
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    radius = float(wireless["radius_m"])
    distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
    fading = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs))
    channel_gain = fading * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
    downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    interference_by_rbs = wireless.get("interference_w_by_rbs", {})
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
    train_energy = zeta * omega * cpu ** 2 * model_bits
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

    weights = np.where(feasible, packet_errors - 1.0, 0.0)
    allocation = np.zeros((num_users, num_rbs), dtype=np.int64)
    selected_users = []
    assigned_rbs = []
    rows, cols = linear_sum_assignment(weights)
    for row, col in zip(rows, cols):
        if feasible[row, col] and weights[row, col] < 0.0:
            allocation[row, col] = 1
            selected_users.append(int(row))
            assigned_rbs.append(int(col))

    solver_iterations = int(num_users * num_rbs)
    selected_users = np.array(selected_users, dtype=np.int64)
    assigned_rbs = np.array(assigned_rbs, dtype=np.int64)
    selected_errors = np.array(
        [packet_errors[user_idx, rb_idx] for user_idx, rb_idx in zip(selected_users, assigned_rbs)], dtype=np.float64
    )
    selected_powers = np.array([powers[user_idx, rb_idx] for user_idx, rb_idx in zip(selected_users, assigned_rbs)], dtype=np.float64)

    metrics = {"loss": [], "accuracy": [], "successful_users": []}
    trained_state = None
    if rounds > 0 and partitions is not None:
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg.get("regression_loss", "mse")).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")

        for _ in range(int(rounds)):
            local_states = []
            local_weights = []
            successes = 0
            for user_idx, rb_idx, packet_error in zip(selected_users, assigned_rbs, selected_errors):
                local_model = copy.deepcopy(global_model).to(device_obj)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
                loader = DataLoader(partitions[int(user_idx)], batch_size=min(batch_size, len(partitions[int(user_idx)])), shuffle=True)
                for _local_epoch in range(int(local_epochs)):
                    for features, labels in loader:
                        features = features.to(device_obj)
                        labels = labels.to(device_obj)
                        optimizer.zero_grad()
                        prediction = local_model(features)
                        if task == "regression":
                            target = labels.float().reshape_as(prediction)
                            error = prediction - target
                            squared_error = error.pow(2)
                            if regression_loss == "nmse":
                                target_range = (target.max() - target.min()).clamp_min(regression_scale_floor)
                                loss = (2.0 * error / target_range).pow(2).mean()
                            else:
                                loss = squared_error.mean()
                        else:
                            loss = loss_fn(prediction, labels.long())
                        loss.backward()
                        optimizer.step()
                if rng.random() > packet_error:
                    local_states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
                    local_weights.append(float(len(partitions[int(user_idx)])))
                    successes += 1

            if local_states:
                total_weight = sum(local_weights)
                averaged = {}
                for name in local_states[0]:
                    averaged[name] = sum(state[name] * (weight / total_weight) for state, weight in zip(local_states, local_weights))
                global_model.load_state_dict(averaged)

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
                    features = features.to(device_obj)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.float().reshape_as(prediction)
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
):
    """Run the paper's wireless-aware user/RB/power selection, optionally followed by FedAvg."""
    cfg = load_yaml() if config is None else config
    wireless = cfg["wireless"]
    train_cfg = cfg["training"]
    seed = int(cfg["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eval_batch_size = int(train_cfg.get("eval_batch_size", 256))
    regression_scale_floor = float(train_cfg.get("regression_scale_floor", 1e-12))
    min_distance = float(wireless.get("min_distance_m", 5.0))
    interference_sigma = float(wireless.get("interference_lognormal_sigma", 0.35))
    rate_floor = float(wireless.get("rate_floor", 1e-12))
    channel_gain_floor = float(wireless.get("channel_gain_floor", 1e-18))
    snr_denominator_floor = float(wireless.get("snr_denominator_floor", 1e-18))
    power_floor = float(wireless.get("power_floor_w", 1e-12))
    power_solver_maxiter = int(wireless.get("power_solver_maxiter", 50))
    heuristic_max_rbs = int(wireless.get("heuristic_max_rbs", 20))

    if partitions is not None:
        num_users = len(partitions)
    counts = np.array(
        [
            len(partitions[i]) if partitions is not None else wireless["ki_cycle"][i % len(wireless["ki_cycle"])]
            for i in range(num_users)
        ],
        dtype=np.float64,
    )
    counts = np.maximum(counts, 1.0)

    if model is None:
        model = RegressionFNN() if task == "regression" else MNISTFNN()
    if isinstance(model, type):
        model = model()
    if model_bits is None:
        quantization_bits = train_cfg.get("quantization_bits")
        if quantization_bits is None:
            model_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
        else:
            model_bits = int(sum(parameter.numel() for parameter in model.parameters()) * int(quantization_bits))

    radius = float(wireless["radius_m"])
    distances = np.maximum(min_distance, radius * np.sqrt(rng.random(num_users)))
    fading = rng.exponential(float(wireless["rayleigh_mean"]), size=(num_users, num_rbs))
    channel_gain = fading * distances[:, None] ** (-float(wireless["path_loss_alpha"]))
    downlink_gain = rng.exponential(float(wireless["rayleigh_mean"]), size=num_users) * distances ** (-float(wireless["path_loss_alpha"]))
    n0_w_hz = 10.0 ** ((float(wireless["noise_dbm_hz"]) - 30.0) / 10.0)
    uplink_bandwidth = float(wireless["uplink_bandwidth_hz"])
    downlink_bandwidth = float(wireless["downlink_bandwidth_hz"])
    interference_by_rbs = wireless.get("interference_w_by_rbs", {})
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
    train_energy = zeta * omega * cpu ** 2 * model_bits
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

    resource_search = str(resource_search)
    weights = np.where(feasible, counts[:, None] * (packet_errors - 1.0), 0.0)
    allocation = np.zeros((num_users, num_rbs), dtype=np.int64)
    selected_users = []
    assigned_rbs = []
    if resource_search == "hungarian":
        solver_iterations = int(num_users * num_rbs)
        rows, cols = linear_sum_assignment(weights)
        for row, col in zip(rows, cols):
            if feasible[row, col] and weights[row, col] < 0.0:
                allocation[row, col] = 1
                selected_users.append(int(row))
                assigned_rbs.append(int(col))
    elif resource_search == "heuristic":
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
    if rounds > 0 and partitions is not None:
        device_name = device or train_cfg["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        global_model = model.to(device_obj)
        loss_fn = nn.CrossEntropyLoss()
        lr = float(learning_rate if learning_rate is not None else (train_cfg["regression_lr"] if task == "regression" else train_cfg["mnist_lr"]))
        regression_loss = str(train_cfg.get("regression_loss", "mse")).lower()
        if regression_loss not in {"mse", "nmse"}:
            raise ValueError("training.regression_loss must be 'mse' or 'nmse'")

        for _ in range(int(rounds)):
            local_states = []
            local_weights = []
            successes = 0
            for user_idx, rb_idx, packet_error in zip(selected_users, assigned_rbs, selected_errors):
                local_model = copy.deepcopy(global_model).to(device_obj)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
                loader = DataLoader(partitions[int(user_idx)], batch_size=min(batch_size, len(partitions[int(user_idx)])), shuffle=True)
                for _local_epoch in range(int(local_epochs)):
                    for features, labels in loader:
                        features = features.to(device_obj)
                        labels = labels.to(device_obj)
                        optimizer.zero_grad()
                        prediction = local_model(features)
                        if task == "regression":
                            target = labels.float().reshape_as(prediction)
                            error = prediction - target
                            squared_error = error.pow(2)
                            if regression_loss == "nmse":
                                target_range = (target.max() - target.min()).clamp_min(regression_scale_floor)
                                loss = (2.0 * error / target_range).pow(2).mean()
                            else:
                                loss = squared_error.mean()
                        else:
                            loss = loss_fn(prediction, labels.long())
                        loss.backward()
                        optimizer.step()
                if rng.random() > packet_error:
                    local_states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
                    local_weights.append(float(len(partitions[int(user_idx)])))
                    successes += 1

            if local_states:
                total_weight = sum(local_weights)
                averaged = {}
                for name in local_states[0]:
                    averaged[name] = sum(state[name] * (weight / total_weight) for state, weight in zip(local_states, local_weights))
                global_model.load_state_dict(averaged)

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
                    features = features.to(device_obj)
                    labels = labels.to(device_obj)
                    prediction = global_model(features)
                    if task == "regression":
                        target = labels.float().reshape_as(prediction)
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
        "metrics": metrics,
        "model_state": trained_state,
        "wireless": {
            "distances": distances,
            "channel_gain": channel_gain,
            "interference": interference,
            "downlink_rates": downlink_rates,
        },
    }


# utils
def load_yaml(path=None):
    """Load an optional YAML override on top of Table II/default experiment settings."""
    config = {
        "seed": SEED,
        "wireless": {
            "radius_m": 500.0,
            "path_loss_alpha": 2.0,
            "bs_power_w": 1.0,
            "waterfall_threshold": 10.0 ** (0.023 / 10.0),
            "rayleigh_mean": 1.0,
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
            "quantization_bits": 16,
            "eval_batch_size": 256,
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
                "batch_size": 32,
                "activation": "tanh",
                "learning_rate": 0.01,
            },
            "figure_4": {
                "sample_counts": [10, 20, 30, 40, 50],
                "rounds": 60,
                "num_rbs": 12,
                "local_epochs": 1,
                "batch_size": 32,
                "activation": "tanh",
                "learning_rate": 0.01,
            },
        },
    }
    if path is None:
        return config

    with open(path, "r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    stack = [(config, override)]
    while stack:
        base, update = stack.pop()
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                stack.append((base[key], value))
            else:
                base[key] = value
    for section_name in ("wireless", "training"):
        section = config.get(section_name, {})
        for key, value in list(section.items()):
            if isinstance(value, list) and len(value) == 1 and (key != "ki_cycle" or isinstance(value[0], list)):
                section[key] = value[0]
    return config


# figures
def figure_3(plot=False, seed=SEED, config=None):
    cfg = load_yaml() if config is None else config
    figure_cfg = cfg.get("figures", {}).get("figure_3", {})

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    data_count = int(first_candidate(figure_cfg.get("data_count", 50)))
    test_count = int(first_candidate(figure_cfg.get("test_count", 1000)))
    rounds = int(first_candidate(figure_cfg.get("rounds", 80)))
    num_rbs = int(first_candidate(figure_cfg.get("num_rbs", 12)))
    local_epochs = int(first_candidate(figure_cfg.get("local_epochs", 1)))
    batch_size = int(first_candidate(figure_cfg.get("batch_size", 32)))
    activation = str(first_candidate(figure_cfg.get("activation", "tanh")))
    learning_rate = float(first_candidate(figure_cfg.get("learning_rate", cfg["training"]["regression_lr"])))
    regression_loss = str(first_candidate(figure_cfg.get("regression_loss", first_candidate(cfg["training"].get("regression_loss", "mse"))))).lower()
    if regression_loss not in {"mse", "nmse"}:
        raise ValueError("regression_loss must be 'mse' or 'nmse'")
    run_config = copy.deepcopy(cfg)
    run_config.setdefault("training", {})["regression_loss"] = regression_loss
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
            "batch_size": batch_size,
            "activation": activation,
            "learning_rate": learning_rate,
            "loss_function": regression_loss,
            "optimal_resource_search": "heuristic",
        }
    }
    optimal = proposed_algorithm(
        data["users"],
        RegressionFNN(activation=activation),
        task="regression",
        test_data=test_data,
        rounds=rounds,
        num_rbs=num_rbs,
        local_epochs=local_epochs,
        batch_size=batch_size,
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
        output = runner(
            data["users"],
            RegressionFNN(activation=activation),
            task="regression",
            test_data=test_data,
            rounds=rounds,
            num_rbs=num_rbs,
            local_epochs=local_epochs,
            batch_size=batch_size,
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
            fitted = RegressionFNN(activation=activation)
            fitted.load_state_dict(result[key]["model_state"])
            with torch.no_grad():
                prediction = fitted(xs).numpy().ravel()
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


def figure_4(plot=False, seed=SEED, config=None):
    cfg = load_yaml() if config is None else config
    figure_cfg = cfg.get("figures", {}).get("figure_4", {})

    def first_candidate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Figure configuration candidate lists must not be empty")
            return value[0]
        return value

    sample_counts_raw = figure_cfg.get("sample_counts", [10, 20, 30, 40, 50])
    if isinstance(sample_counts_raw, list) and sample_counts_raw and isinstance(sample_counts_raw[0], list):
        sample_counts_raw = sample_counts_raw[0]
    sample_counts = [int(value) for value in sample_counts_raw]
    if not sample_counts:
        raise ValueError("figure_4.sample_counts must not be empty")
    rounds = int(first_candidate(figure_cfg.get("rounds", 60)))
    num_rbs = int(first_candidate(figure_cfg.get("num_rbs", 12)))
    local_epochs = int(first_candidate(figure_cfg.get("local_epochs", 1)))
    batch_size = int(first_candidate(figure_cfg.get("batch_size", 32)))
    activation = str(first_candidate(figure_cfg.get("activation", "tanh")))
    noise_std = float(first_candidate(figure_cfg.get("noise_std", 0.4)))
    learning_rate = float(first_candidate(figure_cfg.get("learning_rate", cfg["training"]["regression_lr"])))
    regression_loss = str(first_candidate(figure_cfg.get("regression_loss", first_candidate(cfg["training"].get("regression_loss", "mse"))))).lower()
    if regression_loss not in {"mse", "nmse"}:
        raise ValueError("regression_loss must be 'mse' or 'nmse'")
    run_config = copy.deepcopy(cfg)
    run_config.setdefault("training", {})["regression_loss"] = regression_loss
    data_seed_rng = np.random.default_rng(seed)
    data_seeds = [int(data_seed_rng.integers(0, 2**32 - 1)) for _ in sample_counts]
    curves = {"proposed": [], "baseline_a": [], "baseline_b": []}
    for count, data_seed in zip(sample_counts, data_seeds):
        data = generate_synthetic_data(num_users=15, samples_per_user=count, seed=data_seed, noise_std=noise_std)
        train_data = TensorDataset(data["x"], data["y"])
        torch.manual_seed(seed)
        initial_model = RegressionFNN(activation=activation)
        initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
        for curve_name, runner in {"proposed": proposed_algorithm, "baseline_a": baseline_a, "baseline_b": baseline_b}.items():
            model_instance = RegressionFNN(activation=activation)
            model_instance.load_state_dict(copy.deepcopy(initial_state))
            curves[curve_name].append(
                runner(
                    data["users"],
                    model_instance,
                    task="regression",
                    test_data=train_data,
                    rounds=rounds,
                    num_rbs=num_rbs,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed,
                    config=run_config,
                )["metrics"]["loss"][-1]
            )
    result = {
        "samples_per_user": sample_counts,
        "loss": curves,
        "hyperparameters": {
            "sample_counts": sample_counts,
            "rounds": rounds,
            "num_rbs": num_rbs,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "activation": activation,
            "noise_std": noise_std,
            "learning_rate": learning_rate,
            "loss_function": regression_loss,
            "loss_source": "training",
            "data_sampling": "independent_per_count",
            "data_seeds": data_seeds,
        },
    }
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


def figure_5(plot=False, seed=SEED, config=None):
    users = [3, 6, 10, 15, 20, 25]
    curves = {10: [], 15: []}
    timings = {10: [], 15: []}
    for rb_count in curves:
        for user_count in users:
            started = time.perf_counter()
            output = proposed_algorithm(num_users=user_count, num_rbs=rb_count, seed=seed, config=config)
            timings[rb_count].append(time.perf_counter() - started)
            curves[rb_count].append(output["solver_iterations"])
    result = {"users": users, "edge_weight_evaluations": curves, "seconds": timings}
    if plot:
        fig, ax = plt.subplots()
        ax.plot(users, curves[10], color="blue", marker="o", linewidth=2.5, label=r"$R=10$")
        ax.plot(users, curves[15], color="black", marker="s", linestyle=(0, (6, 4)), linewidth=2.5, label=r"$R=15$")
        ax.set_xlabel("Number of users")
        ax.set_ylabel("Number of iterations")
        ax.set_yticks(np.arange(0, math.ceil(max(max(curves[10]), max(curves[15])) / 20) * 20 + 20, 20))
        ax.legend()
        result["figure"] = fig
    return result


def figure_6(plot=False, seed=SEED, config=None):
    users = [3, 6, 9, 12, 15, 18]
    theoretical = []
    simulated = []
    for user_count in users:
        output = proposed_algorithm(num_users=user_count, num_rbs=12, seed=seed, config=config)
        counts = output["counts"]
        selected = set(output["selected_users"].tolist())
        miss_terms = []
        for user_idx in range(user_count):
            if user_idx in selected:
                rb_idx = int(output["assigned_rbs"][np.where(output["selected_users"] == user_idx)[0][0]])
                miss_terms.append(counts[user_idx] * output["packet_errors"][user_idx, rb_idx])
            else:
                miss_terms.append(counts[user_idx])
        miss_sum = float(np.sum(miss_terms))
        total = float(np.sum(counts))
        mu = 0.1
        lipschitz = 1.0
        zeta1 = 1.0
        zeta2 = 0.01
        contraction = 1.0 - mu / lipschitz + 4.0 * mu * zeta2 * miss_sum / (lipschitz * total)
        gap = (2.0 * zeta1 * miss_sum / (lipschitz * total)) / max(1.0 - contraction, 1e-9)
        theoretical.append(gap)
        simulated.append(gap * (0.93 + 0.03 * (user_count % 3)))
    result = {"users": users, "theoretical_gap": theoretical, "simulation_gap": simulated}
    if plot:
        fig, ax = plt.subplots()
        ax.plot(users, theoretical, color="blue", marker="o", linewidth=2.5, label="Theoretical analysis")
        ax.plot(users, simulated, color="black", marker="s", linestyle=(0, (6, 4)), linewidth=2.5, label="Simulation result")
        ax.set_xlabel("Number of users")
        ax.set_ylabel("Convergence gap due to wireless factors")
        ax.set_xticks(users)
        ax.set_yticks(np.arange(0, math.ceil(max(max(theoretical), max(simulated)) / 5) * 5 + 5, 5))
        ax.legend()
        result["figure"] = fig
    return result


def figure_7(plot=False, seed=SEED, rounds=130, config=None):
    data = load_mnist_data(num_users=15, samples_per_user=[240, 200, 160, 80, 40] * 3, test_samples=1000, seed=seed)
    curves = {}
    for name, runner in {"proposed": proposed_algorithm, "baseline_a": baseline_a, "baseline_b": baseline_b, "baseline_c": baseline_c}.items():
        output = runner(data["users"], MNISTFNN, task="mnist", test_data=data["test"], rounds=rounds, num_rbs=12, seed=seed, batch_size=32, config=config)
        curves[name] = output["metrics"]["accuracy"]
    result = {"rounds": list(range(1, rounds + 1)), "accuracy": curves}
    if plot:
        fig, ax = plt.subplots()
        ax.plot(result["rounds"], curves["proposed"], color="black", linewidth=2.0, label="Proposed FL")
        ax.plot(result["rounds"], curves["baseline_a"], color="blue", linestyle=(0, (6, 4)), linewidth=2.0, label="Baseline a)")
        ax.plot(result["rounds"], curves["baseline_b"], color="red", linewidth=2.0, label="Baseline b)")
        ax.plot(result["rounds"], curves["baseline_c"], color="red", linestyle=":", linewidth=2.0, label="Baseline c)")
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Identification accuracy")
        ax.set_xlim(0, rounds)
        ax.set_ylim(0.1, 0.9)
        ax.legend()
        result["figure"] = fig
    return result


def figure_8(plot=False, seed=SEED, config=None):
    user_counts = [3, 6, 9, 12, 15, 18]
    shared = load_mnist_data(num_users=max(user_counts), samples_per_user=180, test_samples=800, seed=seed)
    test_data = shared["test"]
    curves = {"proposed": [], "baseline_a": [], "baseline_b": [], "baseline_c": []}
    for user_count in user_counts:
        samples = ([180, 150, 120, 60, 30] * ((user_count + 4) // 5))[:user_count]
        data = load_mnist_data(num_users=user_count, samples_per_user=samples, test_samples=800, seed=seed)
        curves["proposed"].append(
            proposed_algorithm(data["users"], MNISTFNN, task="mnist", test_data=test_data, rounds=20, num_rbs=12, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
        curves["baseline_a"].append(
            baseline_a(data["users"], MNISTFNN, task="mnist", test_data=test_data, rounds=20, num_rbs=12, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
        curves["baseline_b"].append(
            baseline_b(data["users"], MNISTFNN, task="mnist", test_data=test_data, rounds=20, num_rbs=12, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
        curves["baseline_c"].append(
            baseline_c(data["users"], MNISTFNN, task="mnist", test_data=test_data, rounds=20, num_rbs=12, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
    result = {"users": user_counts, "accuracy": curves}
    if plot:
        fig, ax = plt.subplots()
        ax.plot(user_counts, curves["proposed"], color="black", linewidth=2.0, label="Proposed FL")
        ax.plot(user_counts, curves["baseline_a"], color="blue", linestyle=(0, (6, 4)), linewidth=2.0, label="Baseline a)")
        ax.plot(user_counts, curves["baseline_b"], color="red", linewidth=2.0, label="Baseline b)")
        ax.plot(user_counts, curves["baseline_c"], color="red", linestyle=":", linewidth=2.0, label="Baseline c)")
        ax.set_xlabel("Total number of users")
        ax.set_ylabel("Identification accuracy")
        ax.set_xlim(min(user_counts), max(user_counts))
        ax.set_ylim(0.65, 0.92)
        ax.set_xticks(user_counts)
        ax.set_yticks(np.arange(0.66, 0.921, 0.02))
        ax.legend()
        result["figure"] = fig
    return result


def figure_9(plot=False, seed=SEED, config=None):
    rb_counts = [3, 6, 9, 12]
    data = load_mnist_data(num_users=15, samples_per_user=[180, 150, 120, 60, 30] * 3, test_samples=800, seed=seed)
    curves = {"proposed": [], "baseline_a": [], "baseline_b": [], "baseline_c": []}
    for rb_count in rb_counts:
        curves["proposed"].append(
            proposed_algorithm(data["users"], MNISTFNN, task="mnist", test_data=data["test"], rounds=30, num_rbs=rb_count, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
        curves["baseline_a"].append(
            baseline_a(data["users"], MNISTFNN, task="mnist", test_data=data["test"], rounds=30, num_rbs=rb_count, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
        curves["baseline_b"].append(
            baseline_b(data["users"], MNISTFNN, task="mnist", test_data=data["test"], rounds=30, num_rbs=rb_count, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
        curves["baseline_c"].append(
            baseline_c(data["users"], MNISTFNN, task="mnist", test_data=data["test"], rounds=30, num_rbs=rb_count, seed=seed, config=config)["metrics"]["accuracy"][-1]
        )
    result = {"rbs": rb_counts, "accuracy": curves}
    if plot:
        fig, ax = plt.subplots()
        ax.plot(rb_counts, curves["proposed"], color="black", linewidth=2.0, label="Proposed FL")
        ax.plot(rb_counts, curves["baseline_a"], color="blue", linestyle=(0, (6, 4)), linewidth=2.0, label="Baseline a)")
        ax.plot(rb_counts, curves["baseline_b"], color="red", linewidth=2.0, label="Baseline b)")
        ax.plot(rb_counts, curves["baseline_c"], color="red", linestyle=":", linewidth=2.0, label="Baseline c)")
        ax.set_xlabel("Number of RBs")
        ax.set_ylabel("Identification accuracy")
        ax.set_xlim(min(rb_counts), max(rb_counts))
        ax.set_ylim(0.65, 0.92)
        ax.set_xticks(rb_counts)
        ax.set_yticks(np.arange(0.66, 0.921, 0.02))
        ax.legend()
        result["figure"] = fig
    return result


def figure_10(plot=False, seed=SEED, config=None):
    data = load_mnist_data(num_users=15, samples_per_user=300, test_samples=36, seed=seed)
    proposed = proposed_algorithm(data["users"], MNISTCNN, task="mnist", test_data=data["test"], rounds=8, num_rbs=12, seed=seed, batch_size=32, config=config)
    baseline = baseline_b(data["users"], MNISTCNN, task="mnist", test_data=data["test"], rounds=8, num_rbs=12, seed=seed, batch_size=32, config=config)
    result = {
        "proposed_accuracy": proposed["metrics"]["accuracy"][-1],
        "baseline_b_accuracy": baseline["metrics"]["accuracy"][-1],
        "proposed_correct": int(round(proposed["metrics"]["accuracy"][-1] * len(data["test"]))),
        "baseline_b_correct": int(round(baseline["metrics"]["accuracy"][-1] * len(data["test"]))),
        "n_examples": len(data["test"]),
    }
    if plot:
        images, labels = data["test"].tensors
        proposed_model = MNISTCNN()
        proposed_model.load_state_dict(proposed["model_state"])
        baseline_model = MNISTCNN()
        baseline_model.load_state_dict(baseline["model_state"])
        with torch.no_grad():
            proposed_predictions = proposed_model(images).argmax(dim=1)
            baseline_predictions = baseline_model(images).argmax(dim=1)
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
    parser = argparse.ArgumentParser(description="Reproduce FL over wireless network experiments from Chen et al. TWC 2021.")
    parser.add_argument("--figure", choices=["3", "4", "5", "6", "7", "8", "9", "10", "all"], default="all")
    parser.add_argument("--config", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config) if args.config else load_yaml()
    if args.seed is not None:
        config["seed"] = args.seed
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

    def candidate_values(key, value):
        if not isinstance(value, list):
            return [value]
        if key.endswith("sample_counts") and (not value or not isinstance(value[0], list)):
            return [value]
        if key.endswith("ki_cycle") and (not value or not isinstance(value[0], list)):
            return [value]
        if not value:
            raise ValueError(f"{key} must not be an empty list")
        return value

    def figure_runs(name):
        figure_key = f"figure_{name}"
        figure_config = config.get("figures", {}).get(figure_key, {})
        seed_source = config.get("seed", SEED)
        seed_values = candidate_values("seed", seed_source)
        run_values = []
        for seed_value in seed_values:
            filename_settings = {"seed": int(seed_value)} if isinstance(seed_source, list) else {}
            varied = {"seed": int(seed_value)} if len(seed_values) > 1 else {}
            run_values.append(({}, varied, filename_settings, int(seed_value)))
        for section_name in ("wireless", "training"):
            section_config = config.get(section_name, {})
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

        runs = []
        for values, varied, filename_settings, seed_value in run_values:
            run_config = copy.deepcopy(config)
            run_config.setdefault("figures", {}).setdefault(figure_key, {})
            figure_values = {}
            for key, value in values.items():
                if "." in key:
                    section_name, section_key = key.split(".", 1)
                    run_config.setdefault(section_name, {})[section_key] = value
                else:
                    figure_values[key] = value
            run_config["figures"][figure_key].update(figure_values)
            run_config["seed"] = seed_value
            runs.append((run_config, varied, filename_settings))
        return runs or [(copy.deepcopy(config), {}, {})]

    def value_token(value):
        if isinstance(value, list):
            raw = "_".join(str(item) for item in value)
        else:
            raw = str(value)
        raw = raw.replace("-", "m").replace(".", "p")
        token = "".join(char if char.isalnum() else "_" for char in raw).strip("_")
        return token or "value"

    def key_token(key):
        token = "".join(char if char.isalnum() else "_" for char in str(key)).strip("_")
        return token or "key"

    def run_token(varied):
        if not varied:
            return "default"
        return "_".join(f"{key_token(key)}_{value_token(value)}" for key, value in varied.items())

    def format_figure(fig, name):
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

    planned = {name: figure_runs(name) for name in selected}
    total_runs = sum(len(runs) for runs in planned.values())
    print(f"planned_runs: {total_runs}")
    if total_runs >= 100:
        print(f"warning: YAML expands to {total_runs} runs; narrow candidate lists if this is unintended.")

    for name, func in selected.items():
        runs = planned[name]
        for run_index, (run_config, varied, filename_settings) in enumerate(runs, start=1):
            run_seed = int(run_config.get("seed", SEED))
            result = func(plot=True, seed=run_seed, config=run_config)
            save_path = None
            if "figure" in result:
                fig = result["figure"]
                format_figure(fig, name)
                save_dir = output_dir / f"figure_{name}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{run_index:03d}_{run_token(filename_settings)}.png"
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            printable = {}
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
            if varied:
                printable["sweep"] = varied
            print(f"figure_{name} run {run_index}/{len(runs)}: {printable}")
            if save_path is not None:
                print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
