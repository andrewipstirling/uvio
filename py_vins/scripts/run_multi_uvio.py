#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import yaml
import argparse

# ============================================================
# CONFIGURATION
# ============================================================
DATASET = "miluv"  # Options: "iros" or "miluv"
NUM_RUNS = 5
RUN_UVIO = True
# ============================================================


def load_config(dataset_name):
    """Load the config file for the given dataset."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "..", "config", f"{dataset_name}_config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_bag_file(config):
    """Verify that the bag file exists before running launch."""
    base_path = os.path.join(config["base_path"], config["config"])
    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])
    bag_path = os.path.join(base_path, config["bag"])
    if not os.path.exists(bag_path):
        print(f"[ERROR] Bag file not found:\n  {bag_path}")
        print("Please check your dataset path or mount the correct folder.")
        sys.exit(1)


def run_uvio_single(config, exp_dir, run_id):
    """
    Run one instance of UVIO, saving results into exp_dir.

    Parameters
    ----------
    config  : dict
        Loaded YAML config.
    exp_dir : str
        Path to the experiment output directory.
    run_id  : int
        Zero-based run index.

    Returns
    -------
    success : bool
        True if the run completed successfully.
    yaw_lines : list[str]
        Terminal output lines containing the word 'yaw'.
    """
    base_path = os.path.join(config["base_path"], config["config"])
    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])

    # Ground-truth path (shared across runs, not written into run_dir)
    path_gt = config["path_gt"]
    if DATASET == "miluv":
        path_gt = os.path.join(base_path, "results_uvio", config["path_gt"])

    # Per-run output paths
    path_est = os.path.join(exp_dir, f"run_{run_id:02d}.txt")
    path_est_global = os.path.join(exp_dir, f"run_{run_id:02d}_global.txt")
    path_time = os.path.join(exp_dir, f"run_{run_id:02d}_timing.txt")
    bag             = os.path.join(base_path, config["bag"])

    launch_cmd = ["roslaunch", "uvio", "miluv.launch"]
    args = [
        "max_cameras:="    + str(config["max_cameras"]),
        "use_stereo:="     + str(config["use_stereo"]).lower(),
        "config:="         + config["config"],
        "bag:="            + bag,
        "bag_start:="      + str(config["bag_start"]),
        "dosave:="         + str(config["dosave"]).lower(),
        "dosave_global:="  + str(config["dosave_global"]).lower(),
        "dotime:="         + str(config["dotime"]).lower(),
        "path_est:="       + path_est,
        "path_est_global:="+ path_est_global,
        "path_time:="      + path_time,
        "path_gt:="        + path_gt,
        "verbosity:="      + config["verbosity"],
        "num_pts:="        + str(config["num_pts"]),
        "config_uwb:="     + config["config_uwb"],
        "uwb_anchors:="    + config["uwb_anchors"],
        "spline_fname:="   + config["spline_fname"],
    ]
    launch_cmd.extend(args)

    print(f"\n{'='*52}")
    print(f"  Run {run_id + 1}/{NUM_RUNS}  ->  {exp_dir}")
    print(f"  Bag : {bag}")
    print(f"{'='*52}\n")

    init_log_file = os.path.join(exp_dir, "initialization_results_log.txt")
    log_file = os.path.join(exp_dir, f"temp_stdout_{run_id:02d}.log")

    bash_command = (
    "source /opt/ros/noetic/setup.bash && "
    "source ~/catkin_ws/devel/setup.bash && "
    + " ".join(launch_cmd))

    env = os.environ.copy()

    # IMPORTANT: isolate ROS logs per run (prevents cross-run corruption)
    env["ROS_LOG_DIR"] = os.path.join(exp_dir, f"roslog_run_{run_id:02d}")

    os.makedirs(env["ROS_LOG_DIR"], exist_ok=True)

    try:
        # ----------------------------
        # 1. Run and dump FULL output
        # ----------------------------
        # subprocess.run(
        #         bash_command,
        #         shell=True,
        #         executable="/bin/bash",
        #         check=True,
        #     )
        with open(log_file, "w") as f:
            result = subprocess.run(
                bash_command,
                shell=True,
                executable="/bin/bash",
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                check=True,
            )

        # # ----------------------------
        # # 2. Parse offline
        # # ----------------------------
        with open(log_file, "r") as f_in, open(init_log_file, "a") as f_out:

            f_out.write(f"\n--- Results for Run {run_id} ---\n")

            found_data = False

            for line in f_in:
                if "yaw" in line.lower():   # more robust than "yaw ="
                    f_out.write(line)
                    found_data = True

            if not found_data:
                f_out.write("No yaw data found in this run.\n")

            f_out.write("\n")

        print(f"[OK] Processed results for Run {run_id + 1}")

        return result

    except subprocess.CalledProcessError as e:
        print(f"[WARN] Run {run_id + 1} failed: {e}")
        return False


def run_experiment(config, exp_name, num_runs):
    """
    Execute `num_runs` back-to-back UVIO runs and collect results under
    a single experiment directory named `exp_name`.
    """

    base_path = os.path.join(config["base_path"], config["config"])
    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])

    exp_dir = os.path.join(base_path, "results_uvio", exp_name)
    if not RUN_UVIO:
        # Running standard vio / openvins
        exp_dir = os.path.join(base_path, "results_uvio", "standard_ov")

    os.makedirs(exp_dir, exist_ok=True)

    print(f"\nExperiment : {exp_name}")
    print(f"Output dir : {exp_dir}")
    print(f"Runs       : {num_runs}\n")

    # ----------------------------
    # Save full config snapshot
    # ----------------------------
    config_snapshot = os.path.join(exp_dir, "config_used.yaml")
    with open(config_snapshot, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


    successful = 0

    # ----------------------------
    # Run experiments
    # ----------------------------
    for run_id in range(num_runs):
        if RUN_UVIO:
            success = run_uvio_single(config, exp_dir, run_id)
        else:
            success = run_openvins_sub(config, exp_dir, run_id)
        
        if success:
            successful += 1

    # ----------------------------
    # Final summary
    # ----------------------------
    print(f"\n{'='*52}")
    print(f"  Finished: {successful}/{num_runs} runs succeeded")
    print(f"  Results : {exp_dir}")
    print(f"{'='*52}\n")

def run_openvins_sub(config, results_dir, run_id):
    """Run OpenVINS subscribe.launch multiple times."""

    launch_cmd = ["roslaunch", "ov_msckf", "subscribe.launch"]

    base_path = os.path.join(config["base_path"], config["config"])
    path_gt = config["path_gt"]

    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])
        path_gt = os.path.join(base_path, "results", config["path_gt"])

    bag = os.path.join(base_path, config["bag"])

    # Unique filenames for each run
    path_est = os.path.join(results_dir, f"run_{run_id:02d}.txt")
    path_time = os.path.join(results_dir, f"run_{run_id:02d}_timing.txt")

    args = [
        "max_cameras:=" + str(config["max_cameras"]),
        "use_stereo:=" + str(config["use_stereo"]).lower(),
        "config:=" + config["config"],
        "dobag:=" + str(config["dobag"]).lower(),
        "bag:=" + bag,
        "bag_start:=" + str(config["bag_start"]),
        "dosave:=" + str(config["dosave"]).lower(),
        "dotime:=" + str(config["dotime"]).lower(),
        "path_est:=" + path_est,
        "path_time:=" + path_time,
        "path_gt:=" + path_gt,
        "verbosity:=" + config["verbosity"],
        "num_pts:=" + str(config["num_pts"]),
    ]

    launch_cmd.extend(args)

    print("\n========================================")
    print(f"Run {run_id + 1}/{NUM_RUNS}")
    print(f"Dataset: {config['dataset']}")
    print(f"Bag file: {config['bag']}")
    print(f"Output: {results_dir}")
    print("========================================\n")

    bash_command = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/catkin_ws/devel/setup.bash && "
        + " ".join(launch_cmd)
    )

    try:
        result = subprocess.run(
            bash_command,
            shell=True,
            executable="/bin/bash",
            check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] roslaunch failed: {e}")
        return False
    
    return result
if __name__ == "__main__":

    config = load_config(DATASET)
    check_bag_file(config)
    run_experiment(config, config["filename_est"], NUM_RUNS)