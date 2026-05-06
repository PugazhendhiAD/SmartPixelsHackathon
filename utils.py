import argparse
import os
import numpy as np
import torch
from typing import List, Tuple


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable directory".format(prospective_dir)
            )


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid file".format(prospective_file)
            )
        else:
            setattr(namespace, self.dest, prospective_file)


class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # folder_name = folder_name.rstrip(os.sep)
        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)


class SymLogTransform:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): how many epochs to wait after last improvement.
            min_delta (float): minimum change in the monitored quantity to count as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
    

def get_device(device_str):
    device = torch.device(device_str)
    
    # Optional: validate availability
    if device.type == "mps":
        assert torch.backends.mps.is_available(), "MPS device is not available."
    elif device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA device is not available."
    
    return device


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_label_index(name: str, label_names: List[str]) -> int:
    if name in label_names:
        return label_names.index(name)
    if name in LABEL_ORDER:
        for n in LABEL_ORDER:
            if n in label_names and n == name:
                return label_names.index(n)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def true_class_from_signed_pt(pt_signed: np.ndarray, pt_boundary: float) -> np.ndarray:
    c = np.empty_like(pt_signed, dtype=np.int64)
    c[(pt_signed >= 0.0) & (pt_signed <= pt_boundary)] = 0
    c[(pt_signed < 0.0) & (pt_signed >= -pt_boundary)] = 1
    c[(pt_signed > pt_boundary) | (pt_signed < -pt_boundary)] = 2
    return c


def confusion_matrix_3(true_c: np.ndarray, pred_c: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(true_c, pred_c):
        cm[int(t), int(p)] += 1
    return cm


def roc_curve_binary(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y = y_true[order].astype(np.int32)
    s = score[order]
    P = max(int(y.sum()), 1)
    N = max(int((1 - y).sum()), 1)
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / P
    fpr = fps / N
    keep = np.r_[True, np.diff(s) != 0]
    return fpr[keep], tpr[keep]


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))
