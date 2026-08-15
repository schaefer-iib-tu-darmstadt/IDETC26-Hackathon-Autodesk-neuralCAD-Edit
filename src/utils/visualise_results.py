from src.utils.args import parse_args
from src.utils.process_config import load_config
import os
import os.path as osp
import json
from src.utils.db import DatabaseManager
import numpy as np
import matplotlib.pyplot as plt


def parse_rating(rating: dict) -> dict:
    """
    Parses a rating dictionary to extract relevant information.

    Args:
        rating (dict): The rating dictionary to parse.

    Returns:
        dict: A dictionary of metric: score containing the parsed information.
    """

    if rating["user"] == "similarity_eval":
        return_dict = {}
        if "volume f1 gt" in rating:
            return_dict["volume_f1"] = rating["volume f1 gt"]
        if "chamfer similarity norm gt" in rating:
            return_dict["chamfer_similarity_norm"] = rating["chamfer similarity norm gt"]
        if "diff f1 gt" in rating:
            return_dict["diff_f1"] = rating["diff f1 gt"]
        return return_dict

    return None


def display_rating_results(config: dict, dbm: DatabaseManager, difficulty: str = "all", request_fields={"eval_vis_multi": True, "eval_geometric": True}, request_type="edit"):

    request_fields["request_type"] = request_type

    if difficulty == "all":
        pass
    else:
        request_fields["difficulty"] = difficulty

    request_ids = dbm.requests.find(request_fields)
    request_ids = [request["_id"] for request in request_ids]
    request_ids.sort()

    scores = {}

    ratings_iterator = dbm.ratings.find()
    for rating in ratings_iterator:

        edit = dbm.edits.find_one({"_id": rating["edit"]})

        if not edit:
            continue

        if "request" not in edit:
            continue
        edit_request_id = edit["request"]
        request = dbm.requests.find_one({"_id": edit_request_id})

        if not request:
            continue

        user = dbm.users.find_one({"_id": edit["user"]})

        if edit_request_id not in request_ids:
            continue

        if request["request_type"] != request_type:
            continue

        valid_user = False
        if user["_id"] in config["benchmark_eval_users"][request_type]:
            valid_user = True
            valid_user_id = user["_id"]
        if "other human" in config["benchmark_eval_users"][request_type] and user.get("is_human", True) and edit["user"] != request["user"]:
            valid_user = True
            valid_user_id = "other human"
        if "gt human" in config["benchmark_eval_users"][request_type] and edit["user"] == request["user"]:
            valid_user = True
            valid_user_id = "gt human"

        if not valid_user:
            continue

        print(rating)

        metrics = parse_rating(rating)

        if not metrics:
            continue

        for k, v in metrics.items():
            user_scores = scores.get(valid_user_id, {})
            scores_dict = user_scores.get(k, {})
            scores_dict[edit_request_id] = v
            user_scores[k] = scores_dict
            scores[valid_user_id] = user_scores

    all_metrics = set()
    for user_scores in scores.values():
        all_metrics.update(user_scores.keys())

    print(all_metrics)

    for request_id in request_ids:
        for user_id in config["benchmark_eval_users"][request_type]:
            if user_id not in scores:
                print(f"User {user_id} not in scores, adding placeholder.")
                scores[user_id] = {}
            for metric in all_metrics:
                if metric not in scores.get(user_id, {}):
                    scores[user_id][metric] = {}
                if request_id not in scores.get(user_id, {}).get(metric, {}):
                    scores[user_id][metric][request_id] = None

    return scores


METRIC_DISPLAY_NAMES = {
    "chamfer_similarity_norm": "Chamfer similarity (norm)",
    "diff_f1": "Diff F1",
    "volume_f1": "Volume F1",
}

BASELINE_MODELS = {
    "gemini-3-pro_cadquery-script",
    "gpt-5.2_cadquery-script",
    "claude-sonnet-4.5_cadquery-script",
}
BASELINE_COLOR = "#999999"
NEW_MODEL_COLOR = "#1f77b4"

HUMAN_BASELINE_KEY = "other human"
HUMAN_BASELINE_LABEL = "human baseline"
HUMAN_BASELINE_COLOR = "#444444"


def model_color(model, index=0):
    return BASELINE_COLOR if model in BASELINE_MODELS else NEW_MODEL_COLOR


def _bar_models(config, request_type):
    """Models rendered as bars: config order, excluding humans (gt/baseline)."""
    return [
        m for m in config["benchmark_eval_users"][request_type]
        if m not in ("gt human", HUMAN_BASELINE_KEY)
    ]


# Note: Check whether a model has evaluation data to omit empty bars in plots
def _has_model_data(results: dict, model: str) -> bool:
    """Check if a model has at least one non-None score in the results dict."""
    for task_data in results.values():
        if model in task_data:
            for metric_data in task_data[model].values():
                if isinstance(metric_data, dict):
                    if any(v is not None for v in metric_data.values()):
                        return True
                elif isinstance(metric_data, list):
                    if any(v is not None for v in metric_data):
                        return True
                elif metric_data is not None:
                    return True
    return False


def _score_or_zero(v):
    """Map missing or failed scores to 0.0 so models are penalized for failures."""
    if v is None or v != v:
        return 0.0
    return float(v)


def _aggregate_scores_by_metric(results: dict):
    """Collapse a results dict (task -> model -> metric -> {edit_id: score}) into
    metric -> model -> list_of_scores, pooling across all tasks/difficulties.

    Missing scores (``None`` placeholders for failed/unrated edits) and NaN
    values (e.g. from ``diff_f1`` on unloadable meshes) are counted as 0.0
    rather than dropped, so a model is penalized for failed runs. This
    matches the leaderboard notebook, which averages every edit with a 0.0
    default for missing metrics.
    """
    metric_model_scores = {}
    for _task, model_data in results.items():
        for model, metric_data in model_data.items():
            for metric, score_dict in metric_data.items():
                if isinstance(score_dict, dict):
                    values = [_score_or_zero(v) for v in score_dict.values()]
                elif isinstance(score_dict, list):
                    values = [_score_or_zero(v) for v in score_dict]
                else:
                    values = [_score_or_zero(score_dict)]
                metric_model_scores.setdefault(metric, {}).setdefault(model, []).extend(values)
    return metric_model_scores


def plot_metric_facets(
    means_by_metric,
    models,
    metrics=None,
    baseline_means=None,
    axes=None,
    show_titles=True,
    row_label=None,
    metric_titles=None,
    suptitle=None,
):
    """Draw one bar facet per metric from pre-computed mean scores.

    Args:
        means_by_metric: ``{metric: {model: mean_score}}``.
        models: model ids rendered as bars, in display order.
        metrics: facet order; defaults to ``means_by_metric`` keys.
        baseline_means: optional ``{metric: mean}`` for a dashed reference line.
        axes: optional row of matplotlib Axes (for multi-row grids).
        show_titles: whether to set a title on each facet.
        row_label: optional y-axis prefix for the first facet in a grid row.
        metric_titles: optional ``{metric: display_title}`` overrides.
        suptitle: optional figure title.

    Returns:
        tuple: (fig, axes)
    """
    if metrics is None:
        metrics = list(means_by_metric.keys())
    if metric_titles is None:
        metric_titles = METRIC_DISPLAY_NAMES

    if axes is None:
        fig, axes_2d = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 6.5), squeeze=False)
        axes = axes_2d[0]
    else:
        fig = axes[0].figure

    x = np.arange(len(models))
    colors = [model_color(m, i) for i, m in enumerate(models)]
    baseline_present = False

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        model_means = means_by_metric.get(metric, {})
        means = [float(model_means.get(m, 0.0)) for m in models]
        ax.bar(x, means, color=colors)

        if baseline_means and metric in baseline_means:
            baseline_mean = baseline_means[metric]
            if baseline_mean is not None and baseline_mean == baseline_mean:
                ax.axhline(
                    float(baseline_mean),
                    linestyle="--",
                    linewidth=1.5,
                    color=HUMAN_BASELINE_COLOR,
                    label=HUMAN_BASELINE_LABEL,
                )
                baseline_present = True

        if show_titles:
            ax.set_title(metric_titles.get(metric, metric), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean score" if i == 0 and row_label is None else "")
        for xi, mean in zip(x, means):
            ax.text(xi, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    if row_label is not None:
        axes[0].set_ylabel(f"{row_label}\nMean score")

    if baseline_present:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout()
    return fig, axes


def faceted_bar_plot(config: dict, results: dict, request_type: str = "edit", metrics=None, save=True):
    """
    Bar chart with one facet (subplot) per metric, showing the mean score for
    every model. The human baseline is drawn as a dashed reference line.
  """
    metric_model_scores = _aggregate_scores_by_metric(results)

    if metrics is None:
        metrics = [m for m in ["chamfer_similarity_norm", "volume_f1", "diff_f1"] if m in metric_model_scores]
        metrics += [m for m in metric_model_scores if m not in metrics]

    models = [m for m in _bar_models(config, request_type) if _has_model_data(results, m)]
    means_by_metric = {}
    baseline_means = {}
    for metric in metrics:
        means_by_metric[metric] = {}
        for model in models:
            values = metric_model_scores.get(metric, {}).get(model, [])
            means_by_metric[metric][model] = float(np.mean(values)) if values else 0.0
        baseline_values = metric_model_scores.get(metric, {}).get(HUMAN_BASELINE_KEY, [])
        baseline_means[metric] = float(np.mean(baseline_values)) if baseline_values else None

    fig, axes = plot_metric_facets(
        means_by_metric,
        models,
        metrics=metrics,
        baseline_means=baseline_means,
        suptitle=f"Metric comparison across models ({request_type})",
    )

    if save:
        out_dir = osp.join(config["storage_dir"]["path"], "results")
        os.makedirs(out_dir, exist_ok=True)
        fig_fn = osp.join(out_dir, "metric_bar_facets.png")
        plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
        print(f"Saved faceted bar plot to {fig_fn}")

    return fig, axes


def cost_barplot(config: dict, dbm: DatabaseManager, request_type: str = "edit", save=True):
    """
    Bar plot of the mean per-edit cost estimate for each model, with error bars
    (standard deviation across that model's edits). Cost is read from each
    edit's ``token_counts.cost_estimate``.

    Only models present in ``benchmark_eval_users`` that have cost data (i.e.
    non-human harness runs) are shown.

    Note: this iterates every edit in the database with no request_type,
    difficulty, or latest-edit-per-user filter. Stale or duplicate edits can
    skew the mean. The standard pipeline calls
    ``clean_db_single_edit_per_user_per_request()`` in ``run_all_benchmarks.py``
    before plotting to keep cost stats honest.
    """
    models = _bar_models(config, request_type)

    model_costs = {}
    for edit in dbm.edits.find({}):
        user = edit.get("user")
        if user not in models:
            continue
        token_counts = edit.get("token_counts") or {}
        cost = token_counts.get("cost_estimate")
        if cost is None:
            continue
        model_costs.setdefault(user, []).append(float(cost))

    ordered_models = [m for m in models if model_costs.get(m)]
    means = [float(np.mean(model_costs[m])) for m in ordered_models]
    stds = [float(np.std(model_costs[m])) for m in ordered_models]
    colors = [model_color(m, i) for i, m in enumerate(ordered_models)]

    fig, ax = plt.subplots(figsize=(3.2 * max(1, len(ordered_models)), 6.5))

    if ordered_models:
        x = np.arange(len(ordered_models))
        ax.bar(x, means, yerr=stds, color=colors, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_models, rotation=45, ha="right", fontsize=8)
        for xi, mean in zip(x, means):
            ax.text(xi, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No cost data available", ha="center", va="center", transform=ax.transAxes)

    ax.set_ylabel("Estimated cost per edit ($)")
    ax.set_title(f"Mean cost per edit ({request_type})", fontsize=10)
    plt.tight_layout()

    if save:
        out_dir = osp.join(config["storage_dir"]["path"], "results")
        os.makedirs(out_dir, exist_ok=True)
        fig_fn = osp.join(out_dir, "cost_barplot.png")
        plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
        print(f"Saved cost bar plot to {fig_fn}")

    return fig, ax


def main():
    args = parse_args()
    config = load_config(args.config)
    dbm = DatabaseManager(config)
    dbm.print_db_summary()
    display_rating_results(config, dbm, difficulty="all", request_type="edit")

    out_dir = osp.join(config["storage_dir"]["path"], "results")
    result_path = osp.join(out_dir, "all_results.json")
    with open(result_path, "r") as f:
        results = json.load(f)

    faceted_bar_plot(config=config, results=results, save=True)
    cost_barplot(config=config, dbm=dbm, save=True)


if __name__ == "__main__":
    main()
