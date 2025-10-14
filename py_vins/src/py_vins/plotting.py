# py_vins/analyze_results.py
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from evo.core import trajectory
from evo.core.metrics import Unit
from evo.tools import file_interface, plot, log
from evo.tools.settings import SETTINGS

def plot_trajectories(ref_traj, est_traj, title="Trajectory Comparison"):
    """
    Plot trajectories using EVO's internal plotting functions.
    """
    plot_collection = plot.PlotCollection(title)

    # Prepare figures
    fig_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))
    ax_traj = plot.prepare_axis(fig_traj, plot.PlotMode.xy, length_unit=Unit.meter)

    fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))
    fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))

    # Color management
    color_palette = itertools.cycle(sns.color_palette())
    ref_color = next(color_palette)
    est_color = next(color_palette)

    # Plot reference
    plot.traj(ax_traj, plot.PlotMode.xy, ref_traj, color=ref_color, label="Reference")
    plot.traj_xyz(axarr_xyz, ref_traj, color=ref_color, label="Reference")
    plot.traj_rpy(axarr_rpy, ref_traj, color=ref_color, label="Reference")

    # Plot estimated
    plot.traj(ax_traj, plot.PlotMode.xy, est_traj, color=est_color, label="Estimate")
    plot.traj_xyz(axarr_xyz, est_traj, color=est_color, label="Estimate")
    plot.traj_rpy(axarr_rpy, est_traj, color=est_color, label="Estimate")

    # Add figures to collection
    plot_collection.add_figure("2D_trajectory", fig_traj)
    plot_collection.add_figure("XYZ", fig_xyz)
    plot_collection.add_figure("RPY", fig_rpy)

    # Show plots
    plot_collection.show()


