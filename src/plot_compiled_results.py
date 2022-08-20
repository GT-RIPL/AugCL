import os
import os.path as osp
import argparse

try:
    import wandb
except ImportError:
    wandb = None
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MARKER_ORDER = ["^", "<", "v", "d", "s", "x", "o", ">"]


def fig_save(
    save_dir,
    save_name,
    fig,
    is_high_quality=True,
    verbose=True,
    clear=False,
    log_wandb=False,
    wandb_name=None,
) -> str:
    """
    :param save_dir: Directory to save file to. Directory is created if it does not exist.
    :param save_name: No file extension included in name.
    :returns: The saved full file path.
    """
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    plt.tight_layout()
    if is_high_quality:
        full_path = osp.join(save_dir, f"{save_name}.pdf")
        fig.savefig(full_path, bbox_inches="tight", dpi=100)
    else:
        full_path = osp.join(save_dir, f"{save_name}.png")
        fig.savefig(full_path)

    if verbose:
        print(f"Saved to {full_path}")
    if clear:
        plt.close(fig)
        plt.clf()

    if log_wandb:
        if wandb_name is None:
            raise ValueError("Must specify wandb log name as well")
        if wandb is None:
            raise ValueError("Wandb is not installed.")
        wandb.log({wandb_name: wandb.Image(full_path)})
    return full_path


def smooth_arr(scalars: List[float], weight: float) -> List[float]:
    """
    Taken from the answer here https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
    :param weight: Between 0 and 1.
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def line_plot(
    plot_df,
    x_name: str,
    y_name: str,
    avg_key: str,
    group_key: str,
    smooth_factor: Union[Dict[str, float], float] = 1,
    ax: Optional[matplotlib.axes.Axes] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    y_disp_bounds: Optional[Tuple[float, float]] = None,
    x_disp_bounds: Optional[Tuple[float, float]] = None,
    group_colors: Optional[Dict[str, int]] = None,
    xtick_fn=None,
    ytick_fn=None,
    legend=True,
    rename_map: Optional[Dict[str, str]] = None,
    title=None,
    axes_font_size=14,
    title_font_size=18,
    legend_font_size="x-large",
    method_idxs: Optional[Dict[str, int]] = None,
    num_marker_points: Optional[Dict[str, int]] = None,
    line_styles: Optional[Dict[str, str]] = None,
    tight=False,
    nlegend_cols=1,
    fetch_std=False,
    ax_dims: Tuple[int, int] = (5, 4),
):
    """
    :param avg_key: This is typically the seed.
    :param group_key: These are the different lines.
    :param smooth_factor: Can specify a different smooth factor per method if desired.
    :param y_bounds: What the data plot values are clipped to.
    :param y_disp_bounds: What the plotting is stopped at.
    :param ax: If not specified, one is automatically created, with the specified dimensions under `ax_dims`
    :param group_colors: If not specified defaults to `method_idxs`.
    :param num_marker_points: Key maps method name to the number of markers drawn on the line, NOT the
      number of points that are plotted! By default this is 8.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=ax_dims)
    if rename_map is None:
        rename_map = {}
    if line_styles is None:
        line_styles = {}
    if num_marker_points is None:
        num_marker_points = {}

    if method_idxs is None:
        method_idxs = {k: i for i, k in enumerate(plot_df[group_key].unique())}

    plot_df = plot_df.copy()
    if tight:
        plt.tight_layout(pad=2.2)
    if group_colors is None:
        group_colors = method_idxs

    colors = sns.color_palette()
    group_colors = {k: colors[i] for k, i in group_colors.items()}

    avg_y_df = plot_df.groupby([group_key, x_name]).mean()
    std_y_df = plot_df.groupby([group_key, x_name]).std()

    if y_name in plot_df.columns and y_name not in avg_y_df.columns:
        raise ValueError(
            f"Desired column {y_name} lost in the grouping. Make sure it is a numeric type"
        )

    method_runs = plot_df.groupby(group_key)[avg_key].unique()
    if fetch_std:
        y_std = y_name + "_std"
        new_df = []
        for k, sub_df in plot_df.groupby([group_key]):
            where_matches = avg_y_df.index.get_level_values(0) == k

            use_df = avg_y_df[where_matches]
            if np.isnan(sub_df.iloc[0][y_std]):
                use_df["std"] = std_y_df[where_matches][y_name]
            else:
                use_df["std"] = avg_y_df[where_matches][y_std]
            new_df.append(use_df)
        avg_y_df = pd.concat(new_df)
    else:
        avg_y_df["std"] = std_y_df[y_name]

    lines = []
    names = []

    # Update the legend info with any previously plotted lines
    if ax.get_legend() is not None:
        all_lines = ax.get_lines()

        for i, n in enumerate(ax.get_legend().get_texts()):
            names.append(n.get_text())
            lines.append((all_lines[i * 2 + 1], all_lines[i * 2]))

    for name, sub_df in avg_y_df.groupby(level=0):
        names.append(name)
        # sub_df = smooth_data(sub_df, smooth_factor, y_name, [group_key, avg_key])
        x_vals = sub_df.index.get_level_values(x_name).to_numpy()
        y_vals = sub_df[y_name].to_numpy()

        if x_disp_bounds is not None:
            use_y_vals = sub_df[
                sub_df.index.get_level_values(x_name) < x_disp_bounds[1]
            ][y_name].to_numpy()
        else:
            use_y_vals = y_vals
        print(
            f"{name}: n_seeds: {len(method_runs[name])} (from WB run IDs {list(method_runs[name])})",
            max(use_y_vals),
            use_y_vals[-1],
        )
        y_std = sub_df["std"].fillna(0).to_numpy()
        if isinstance(smooth_factor, dict):
            use_smooth_factor = (
                smooth_factor[name]
                if name in smooth_factor
                else smooth_factor["default"]
            )
        else:
            use_smooth_factor = smooth_factor
        if use_smooth_factor != 0.0:
            y_vals = np.array(smooth_arr(y_vals, use_smooth_factor))
            y_std = np.array(smooth_arr(y_std, use_smooth_factor))

        add_kwargs = {}
        if name in line_styles:
            add_kwargs["linestyle"] = line_styles[name]
        line_to_add = ax.plot(x_vals, y_vals, **add_kwargs)
        sel_vals = [
            int(x)
            for x in np.linspace(0, len(x_vals) - 1, num=num_marker_points.get(name, 8))
        ]
        # midx = method_idxs[name] % len(MARKER_ORDER)
        ladd = ax.plot(
            x_vals[sel_vals],
            y_vals[sel_vals],
            # MARKER_ORDER[midx],
            label=rename_map.get(name, name),
            color=group_colors[name],
            markersize=8,
        )

        lines.append((ladd[0], line_to_add[0]))

        plt.setp(line_to_add, linewidth=2, color=group_colors[name])
        min_y_fill = y_vals - y_std
        max_y_fill = y_vals + y_std

        if y_bounds is not None:
            min_y_fill = np.clip(min_y_fill, y_bounds[0], y_bounds[1])
            max_y_fill = np.clip(max_y_fill, y_bounds[0], y_bounds[1])

        ax.fill_between(
            x_vals, min_y_fill, max_y_fill, alpha=0.2, color=group_colors[name]
        )
    if y_disp_bounds is not None:
        ax.set_ylim(*y_disp_bounds)
    if x_disp_bounds is not None:
        ax.set_xlim(*x_disp_bounds)

    if xtick_fn is not None:
        plt.xticks(ax.get_xticks(), [xtick_fn(t) for t in ax.get_xticks()])
    if ytick_fn is not None:
        plt.yticks(ax.get_yticks(), [ytick_fn(t) for t in ax.get_yticks()])

    if legend:
        labs = [(i, line_to_add[0].get_label()) for i, line_to_add in enumerate(lines)]
        labs = sorted(labs, key=lambda x: method_idxs[names[x[0]]])
        plt.legend(
            [lines[i] for i, _ in labs],
            [x[1] for x in labs],
            fontsize=legend_font_size,
            ncol=nlegend_cols,
        )

    ax.grid(b=True, which="major", color="lightgray", linestyle="--")

    ax.set_xlabel(rename_map.get(x_name, x_name), fontsize=axes_font_size)
    ax.set_ylabel(rename_map.get(y_name, y_name), fontsize=axes_font_size)
    if title is not None and title != "":
        ax.set_title(title, fontsize=title_font_size)
    return fig  # noqa: R504


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled_csv_path", type=str, required=True)
    parser.add_argument("--y_axis_title", type=str, required=True)
    parser.add_argument("--fig_save_name", type=str, default="no_name")
    parser.add_argument("--smooth_factor", type=float, default=0.0)
    parser.add_argument("--legend", default=True, type=bool)
    parser.add_argument("--title", default=None, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.compiled_csv_path)
    fig = line_plot(
        df, "step", args.y_axis_title, "seed", "method_name", legend=args.legend, smooth_factor=args.smooth_factor, title=args.title
    )
    fig_save("./", args.fig_save_name, fig)
