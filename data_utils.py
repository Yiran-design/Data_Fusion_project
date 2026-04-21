import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

VIEW_NAMES = ["clear", "shift_blur", "occluded"]
COSTS = np.array([2.0, 1.0, 1.0], dtype=float)


def clip_img(img):
    return np.clip(img, 0, 16)


def blur3x3(img):
    """Simple 3x3 mean blur without external dependencies."""
    padded = np.pad(img, 1, mode="edge")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = padded[i:i+3, j:j+3].mean()
    return out


def view_clear(img, rng):
    return clip_img(img + rng.normal(0, 0.3, img.shape))


def view_shift_blur(img, rng):
    shifted = np.roll(img, shift=1, axis=1)
    return clip_img(blur3x3(shifted))


def view_occluded(img, rng):
    out = img.copy()
    out[:3, :3] = 0
    out = np.roll(out, shift=1, axis=0)
    out = out + rng.normal(0, 0.2, img.shape)
    return clip_img(out)


VIEW_FNS = [view_clear, view_shift_blur, view_occluded]


def make_augmented_training_set(X, y, rng):
    X_aug, y_aug = [], []
    for x_flat, label in zip(X, y):
        img = x_flat.reshape(8, 8)
        for fn in VIEW_FNS:
            X_aug.append(fn(img, rng).reshape(-1))
            y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)


def load_and_train_model(seed=42):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.30,
        random_state=seed,
        stratify=digits.target,
    )

    rng = np.random.default_rng(seed)
    X_aug, y_aug = make_augmented_training_set(X_train, y_train, rng)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed),
    )
    clf.fit(X_aug, y_aug)
    return clf, X_test, y_test
