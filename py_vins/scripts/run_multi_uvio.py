#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import yaml

# ============================================================
# CONFIGURATION
# ============================================================
DATASET = "miluv"   # Options: "iros" or "miluv"
NUM_RUNS = 5        # Number of repeated runs
# ============================================================


def load_config(dataset_name):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        script_dir, "..", "config", f"{dataset_name}_config.yaml"
    )

    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_bag_file(config):
    base_path = os.path.join(config["base_path"], config["config"])
    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])

    bag_path = os.path.join(base_path, config["bag"])
    if not os.path.exists(bag_path):
        print(f"[ERROR] Bag file not found:\n  {bag_path}")
        sys.exit(1)


def make_results_dir(base_path, filename_est):
    results_dir = os.path.join(
        base_path, "results_uvio", filename_est
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def run_uvio(config, results_dir, run_id):
    launch_cmd = ["roslaunch", "uvio", "miluv.launch"]

    base_path = os.path.join(config["base_path"], config["config"])
    path_gt = config["path_gt"]

    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])
        path_gt = os.path.join(base_path, "results_uvio", config["path_gt"])

    bag = os.path.join(base_path, config["bag"])

    path_est = os.path.join(
        results_dir, f"run_{run_id:02d}.txt"
    )
    path_time = os.path.join(
        results_dir, f"run_{run_id:02d}_timing.txt"
    )

    args = [
        "max_cameras:=" + str(config["max_cameras"]),
        "use_stereo:=" + str(config["use_stereo"]).lower(),
        "config:=" + config["config"],
        "bag:=" + bag,
        "bag_start:=" + str(config["bag_start"]),
        "dosave:=" + str(config["dosave"]).lower(),
        "dotime:=" + str(config["dotime"]).lower(),
        "path_est:=" + path_est,
        "path_time:=" + path_time,
        "path_gt:=" + path_gt,
        "verbosity:=" + config["verbosity"],
        "num_pts:=" + str(config["num_pts"]),
        "config_uwb:=" + config["config_uwb"],
        "uwb_anchors:=" + config["uwb_anchors"],
        "spline_fname:=" + config["spline_fname"],
    ]

    launch_cmd.extend(args)

    print("\n========================================")
    print(f"Run ID   : {run_id:02d}")
    print(f"Output   : {results_dir}")
    print("========================================\n")

    bash_command = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/catkin_ws/devel/setup.bash && "
        + " ".join(launch_cmd)
    )

    logfile = os.path.join(
        results_dir, f"roslaunch_run_{run_id:02d}.log"
    )

    try:
        with open(logfile, "w") as log:
            subprocess.run(
                bash_command,
                shell=True,
                executable="/bin/bash",
                stdout=log,
                stderr=log,
                check=True,
            )
    except subprocess.CalledProcessError:
        print(f"[ERROR] roslaunch failed (see log): {logfile}")
        sys.exit(1)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    config = load_config(DATASET)
    check_bag_file(config)

    base_path = os.path.join(config["base_path"], config["config"])
    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])

    results_dir = make_results_dir(base_path, config["filename_est"])

    # Save config ONCE
    config_file = os.path.join(results_dir, "config_used.yaml")
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            yaml.dump(config, f)

    for run_id in range(NUM_RUNS):
        print(f"\n========== RUN {run_id + 1}/{NUM_RUNS} ==========\n")
        run_uvio(config, results_dir, run_id)

    print("\nAll runs completed successfully")
