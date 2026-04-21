import numpy as np

from data_utils import VIEW_FNS, COSTS
from scheduler import (
    semantic_scores_from_probs,
    select_random,
    select_aoi_only,
    select_semantic_only,
    select_joint,
)


def fuse_cached_probs(cached_probs, cached_sem, ages, n_classes=10):
    fused = np.zeros(n_classes, dtype=float)
    weight_sum = 0.0

    for i in range(len(cached_probs)):
        if cached_probs[i] is None:
            continue
        # Stale cache contributes less
        w = float(cached_sem[i]) / float(ages[i])
        fused += w * cached_probs[i]
        weight_sum += w

    if weight_sum == 0:
        return None
    return fused / weight_sum


def run_policy(clf, X_test, y_test, budget, method,
               n_episodes=200, episode_len=5, seed=42):
    rng = np.random.default_rng(seed)

    all_acc = []
    all_aoi = []
    all_cost = []

    V = len(VIEW_FNS)
    n_classes = len(np.unique(y_test))

    for _ in range(n_episodes):
        idx = rng.integers(0, len(X_test))
        base_img = X_test[idx].reshape(8, 8)
        label = y_test[idx]

        cached_probs = [None] * V
        cached_sem = np.zeros(V, dtype=float)
        ages = np.full(V, 3.0)  # start with stale cache

        for _ in range(episode_len):
            views = [fn(base_img, rng).reshape(-1) for fn in VIEW_FNS]
            views_np = np.array(views)

            probs = clf.predict_proba(views_np)
            sem_scores = semantic_scores_from_probs(probs)

            if method == "random":
                selected = select_random(COSTS, budget, rng)
            elif method == "aoi":
                selected = select_aoi_only(ages, COSTS, budget)
            elif method == "semantic":
                selected = select_semantic_only(sem_scores, COSTS, budget)
            elif method == "joint":
                selected = select_joint(views_np, sem_scores, ages, COSTS, budget)
            else:
                raise ValueError(f"Unknown method: {method}")

            selected_mask = np.zeros(V, dtype=bool)
            for i in selected:
                selected_mask[i] = True

            for i in range(V):
                if selected_mask[i]:
                    cached_probs[i] = probs[i]
                    cached_sem[i] = max(float(sem_scores[i]), 1e-6)
                    ages[i] = 1.0
                else:
                    ages[i] += 1.0

            fused = fuse_cached_probs(cached_probs, cached_sem, ages, n_classes=n_classes)
            pred = int(np.argmax(np.mean(probs, axis=0))) if fused is None else int(np.argmax(fused))

            all_acc.append(int(pred == label))
            all_aoi.append(float(np.mean(ages)))
            all_cost.append(float(np.sum(COSTS[selected])))

    return {
        "accuracy": float(np.mean(all_acc)),
        "avg_aoi": float(np.mean(all_aoi)),
        "avg_cost": float(np.mean(all_cost)),
    }


def run_all_experiments(clf, X_test, y_test, budgets, methods,
                        n_episodes=200, episode_len=5, seed=42):
    results = {m: {"accuracy": [], "avg_aoi": [], "avg_cost": []} for m in methods}
    for budget in budgets:
        for method in methods:
            r = run_policy(
                clf=clf,
                X_test=X_test,
                y_test=y_test,
                budget=budget,
                method=method,
                n_episodes=n_episodes,
                episode_len=episode_len,
                seed=seed + int(budget),
            )
            results[method]["accuracy"].append(r["accuracy"])
            results[method]["avg_aoi"].append(r["avg_aoi"])
            results[method]["avg_cost"].append(r["avg_cost"])
    return results
