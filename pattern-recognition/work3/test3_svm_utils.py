import contextlib
import io
import os
import sys

import numpy as np
import scipy.io as sio
from libsvm.svmutil import svm_parameter, svm_predict, svm_problem, svm_train


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _suppress_stdout():
    return contextlib.redirect_stdout(io.StringIO())


def _load_mat(path):
    mat = sio.loadmat(path)
    key = [k for k in mat if not k.startswith("_")][0]
    return np.asarray(mat[key])


def load_data(data_dir=DATA_DIR):
    files = {
        "train_data": os.path.join(data_dir, "train_data.mat"),
        "train_label": os.path.join(data_dir, "train_label.mat"),
        "test_data": os.path.join(data_dir, "test_data.mat"),
    }
    for name, path in files.items():
        if not os.path.isfile(path):
            sys.exit(f"[error] missing {path}; put {name}.mat under work3/")

    Xtr = _load_mat(files["train_data"]).astype(np.float64)
    ytr = _load_mat(files["train_label"]).astype(np.float64).ravel()
    Xte = _load_mat(files["test_data"]).astype(np.float64)

    if Xtr.shape[0] == 361 and Xtr.shape[1] != 361:
        Xtr = Xtr.T
    if Xte.shape[0] == 361 and Xte.shape[1] != 361:
        Xte = Xte.T

    print(f"[load] Xtrain={Xtr.shape}, ytrain={ytr.shape}, Xtest={Xte.shape}")
    print(f"[load] train +1={(ytr == 1).sum()}, -1={(ytr == -1).sum()}")
    return Xtr, ytr, Xte


def spatial_stats(X):
    imgs = X.reshape(-1, 19, 19)
    row_sum = imgs.sum(axis=2)
    col_sum = imgs.sum(axis=1)
    mean = imgs.mean(axis=(1, 2)).reshape(-1, 1)
    std = imgs.std(axis=(1, 2)).reshape(-1, 1)
    center = imgs[:, 5:14, 5:14].sum(axis=(1, 2)).reshape(-1, 1)
    border = (
        imgs[:, 0, :].sum(axis=1)
        + imgs[:, -1, :].sum(axis=1)
        + imgs[:, :, 0].sum(axis=1)
        + imgs[:, :, -1].sum(axis=1)
    ).reshape(-1, 1)
    return np.hstack([row_sum, col_sum, mean, std, center, border])


def hog_features(X, n_row=3, n_col=3, n_bins=9, signed=False, norm="l2"):
    N = X.shape[0]
    imgs = X.reshape(N, 19, 19)

    pad = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode="edge")
    dx = pad[:, 1:20, 2:21] - pad[:, 1:20, 0:19]
    dy = pad[:, 2:21, 1:20] - pad[:, 0:19, 1:20]

    magnitude = np.sqrt(dx**2 + dy**2)
    angle_range = 360.0 if signed else 180.0
    orientation = np.degrees(np.arctan2(dy, dx)) % angle_range
    bin_width = angle_range / n_bins
    bin_idx = np.clip(np.floor(orientation / bin_width).astype(int), 0, n_bins - 1)

    row_edges = np.round(np.linspace(0, 19, n_row + 1)).astype(int)
    col_edges = np.round(np.linspace(0, 19, n_col + 1)).astype(int)

    blocks = []
    for i in range(n_row):
        for j in range(n_col):
            r0, r1 = row_edges[i], row_edges[i + 1]
            c0, c1 = col_edges[j], col_edges[j + 1]
            cell_mag = magnitude[:, r0:r1, c0:c1]
            cell_bin = bin_idx[:, r0:r1, c0:c1]

            hist = np.zeros((N, n_bins), dtype=np.float64)
            for b in range(n_bins):
                hist[:, b] = (cell_mag * (cell_bin == b)).sum(axis=(1, 2))

            if norm == "l1sqrt":
                hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-6)
                hist = np.sqrt(hist)
                hist = hist / np.sqrt((hist**2).sum(axis=1, keepdims=True) + 1e-6)
            elif norm == "l1":
                hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-6)
            else:
                hist = hist / np.sqrt((hist**2).sum(axis=1, keepdims=True) + 1e-6)

            blocks.append(hist)

    return np.hstack(blocks)


def base_feature_groups(X, hog_kwargs=None):
    if hog_kwargs is None:
        hog_kwargs = {}
    return {
        "pixel": X,
        "stat": spatial_stats(X),
        "hog": hog_features(X, **hog_kwargs),
    }


class MinMaxScaler:
    def __init__(self, lo=-1.0, hi=1.0):
        self.lo = lo
        self.hi = hi

    def fit(self, X):
        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)
        self.rng = self.xmax - self.xmin
        self.rng[self.rng == 0] = 1.0
        return self

    def transform(self, X):
        Xs = (X - self.xmin) / self.rng
        return Xs * (self.hi - self.lo) + self.lo


class StandardScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def scale_train_test(Xtr, Xte, scaler_cls=MinMaxScaler):
    scaler = scaler_cls().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), scaler


def scale_groups(train_groups, test_groups, scaler_cls=MinMaxScaler):
    scaled_train = {}
    scaled_test = {}
    scalers = {}
    for name in train_groups:
        Xtr_s, Xte_s, scaler = scale_train_test(train_groups[name], test_groups[name], scaler_cls)
        scaled_train[name] = Xtr_s
        scaled_test[name] = Xte_s
        scalers[name] = scaler
    return scaled_train, scaled_test, scalers


def concat_weighted_groups(groups, weights, order=("pixel", "stat", "hog")):
    return np.hstack([groups[name] * weights.get(name, 1.0) for name in order if name in groups])


def cv_accuracy_rbf(y, X, C, gamma, nfold=5):
    param = f"-s 0 -t 2 -c {C} -g {gamma} -v {nfold} -q"
    with _suppress_stdout():
        return svm_train(y.tolist(), X.tolist(), param)


def _is_better(acc, C, gamma, best):
    if acc > best["acc"]:
        return True
    if acc == best["acc"] and best["C"] is not None:
        return C < best["C"] or (C == best["C"] and gamma < best["g"])
    return False


def grid_search_rbf(
    y,
    X,
    C_exps=range(-5, 16, 2),
    g_exps=range(-15, 4, 2),
    nfold=5,
    tag="grid",
    verbose=True,
):
    C_list = list(C_exps)
    g_list = list(g_exps)
    best = {"acc": -1, "C": None, "g": None, "ce": None, "ge": None}

    if verbose:
        print(f"\n=== {tag} ===")
        print(f"    C exps: {C_list}")
        print(f"    g exps: {g_list}")
        print(f"    CV: {nfold}-fold")

    for ce in C_list:
        for ge in g_list:
            C = 2.0**ce
            gamma = 2.0**ge
            acc = cv_accuracy_rbf(y, X, C, gamma, nfold)
            flag = ""
            if _is_better(acc, C, gamma, best):
                best.update(acc=acc, C=C, g=gamma, ce=ce, ge=ge)
                flag = "  <-- best"
            if verbose:
                print(
                    f"    C=2^{ce:+6.2f}={C:10.4g}  "
                    f"g=2^{ge:+6.2f}={gamma:10.4g}  acc={acc:6.3f}%{flag}"
                )

    if verbose:
        print(
            f"[{tag}] best C=2^{best['ce']:+.2f}={best['C']}, "
            f"g=2^{best['ge']:+.2f}={best['g']}, CV={best['acc']:.3f}%"
        )
    return best


def fine_grid_around(best, y, X, step=0.25, span=1.0, nfold=5, tag="fine"):
    C_exps = np.arange(best["ce"] - span, best["ce"] + span + 1e-9, step)
    g_exps = np.arange(best["ge"] - span, best["ge"] + span + 1e-9, step)
    return grid_search_rbf(y, X, C_exps=C_exps, g_exps=g_exps, nfold=nfold, tag=tag)


def train_final_rbf(y, X, C, gamma):
    prob = svm_problem(y.tolist(), X.tolist())
    param = svm_parameter(f"-s 0 -t 2 -c {C} -g {gamma} -q")
    with _suppress_stdout():
        return svm_train(prob, param)


def predict(model, X, y_dummy=None):
    if y_dummy is None:
        y_dummy = [0.0] * X.shape[0]
    p_lab, _, _ = svm_predict(y_dummy, X.tolist(), model, "-q")
    return np.asarray(p_lab)


def store_txt(labels, result_txt):
    with open(result_txt, "w") as w:
        for index, lab in enumerate(labels, 1):
            w.write(f"{index} {int(lab)}\n")


def train_predict_save(ytr, Xtr, Xte, best, out_name):
    print("\n=== train final model ===")
    print(f"    C={best['C']}, g={best['g']}")
    model = train_final_rbf(ytr, Xtr, best["C"], best["g"])
    p_train = predict(model, Xtr, ytr.tolist())
    print(f"[train acc] {(p_train == ytr).mean() * 100:.3f}%")

    y_pred = predict(model, Xte)
    out_path = os.path.join(DATA_DIR, out_name)
    store_txt(y_pred, out_path)
    print(f"[output] {len(y_pred)} predictions saved to {out_path}")
    print(f"[output] +1={(y_pred == 1).sum()}, -1={(y_pred == -1).sum()}")
