import numpy as np


def semantic_scores_from_probs(probs):
    """Semantic score = top1 probability - top2 probability."""
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def feasible_greedy(order, costs, budget):
    selected = []
    used = 0.0
    for i in order:
        if used + costs[i] <= budget:
            selected.append(int(i))
            used += costs[i]
    return selected


def diversity_bonus(view_flat, selected_views):
    if len(selected_views) == 0:
        return 0.0
    dists = [np.linalg.norm(view_flat - sv) / len(view_flat) for sv in selected_views]
    return float(np.mean(dists))


def select_random(costs, budget, rng):
    order = rng.permutation(len(costs))
    return feasible_greedy(order, costs, budget)


def select_aoi_only(ages, costs, budget):
    # Update the stalest cached views first
    order = np.argsort(-ages)
    return feasible_greedy(order, costs, budget)


def select_semantic_only(sem_scores, costs, budget):
    order = np.argsort(-(sem_scores / costs))
    return feasible_greedy(order, costs, budget)


def select_joint(views_flat, sem_scores, ages, costs, budget,
                 alpha=0.45, beta=0.35, gamma=0.25, delta=0.15):
    """Greedy selection under budget with a joint utility."""
    selected = []
    selected_views = []
    used = 0.0
    remaining = set(range(len(costs)))

    age_norm = ages / max(float(np.max(ages)), 1.0)
    cost_norm = costs / float(np.max(costs))

    while True:
        best_i = None
        best_score = -1e9

        for i in remaining:
            if used + costs[i] > budget:
                continue

            div = diversity_bonus(views_flat[i], selected_views)
            score = alpha * sem_scores[i] + beta * age_norm[i] + gamma * div - delta * cost_norm[i]
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        selected.append(int(best_i))
        selected_views.append(views_flat[best_i])
        used += costs[best_i]
        remaining.remove(best_i)

    return selected
