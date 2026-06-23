#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import yaml
import time
import glob
import shutil

# ============================================================
# CONFIGURATION
# ============================================================
DATASET = "miluv"   # Options: "iros" or "miluv"
NUM_RUNS = 5        # Number of repeated runs
RUN_VIO = False
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

def load_uwb_config(dataset_name: str, uwb_config_name: str):
    """Load the config file for the given dataset."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config",dataset_name, uwb_config_name)
    
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


def make_results_dir(base_path, filename_est):
    """Create results directory for multiple runs."""
    results_dir = os.path.join(base_path, "results_uvio", filename_est)
    if RUN_VIO:
        results_dir = os.path.join(base_path, "results_uvio", "standard_ov")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def cleanup_ros():
    """Kill UVIO nodes but leave the RViz launcher and Master alone."""
    print("[INFO] Cleaning up UVIO...")
    current_pid = os.getpid()
    
    # We want to kill the specific launch file for miluv, 
    # but NOT the rviz.launch.
    # We use grep -v to exclude our script, rviz, and the master.
    exclude = f"rviz|roscore|rosmaster|{current_pid}"
    
    # This targets the UVIO processes specifically
    cmd = f"ps -ef | grep -E 'uvio|miluv' | grep -vE '{exclude}' | awk '{{print $2}}' | xargs kill -9"
    
    subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(1)


def run_uvio(config, results_dir, run_id):
    """Run the open_vins uvio miluv.launch using provided config."""
    
    launch_cmd = ["roslaunch", "uvio", "miluv.launch"]
    
    base_path = os.path.join(config["base_path"], config["config"])
    path_gt = config["path_gt"]
    
    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])
        path_gt = os.path.join(base_path, "results_uvio", config["path_gt"])
    
    bag = os.path.join(base_path, config["bag"])
    
    # Unique filenames for each run
    path_est = os.path.join(results_dir, f"run_{run_id:02d}.txt")
    path_est_global = os.path.join(results_dir, f"run_{run_id:02d}_global.txt")
    path_time = os.path.join(results_dir, f"run_{run_id:02d}_timing.txt")
    
    # ROS parameters as arguments
    args = [
        "max_cameras:=" + str(config["max_cameras"]),
        "use_stereo:=" + str(config["use_stereo"]).lower(),
        "config:=" + config["config"],
        "bag:=" + bag,
        "bag_start:=" + str(config["bag_start"]),
        "dosave:=" + str(config["dosave"]).lower(),
        "dosave_global:=" + str(config["dosave_global"]).lower(),
        "dotime:=" + str(config["dotime"]).lower(),
        "path_est:=" + path_est,
        "path_est_global:=" + path_est_global,
        "path_time:=" + path_time,
        "path_gt:=" + path_gt,
        "verbosity:=" + config["verbosity"],
        "num_pts:=" + str(config["num_pts"]),
        "config_uwb:=" + config["config_uwb"],
        "uwb_anchors:=" + config["uwb_anchors"],
        "spline_fname:=" + config["spline_fname"]
    ]
    
    launch_cmd.extend(args)
    
    print("\n========================================")
    print(f"Run {run_id + 1}/{NUM_RUNS}")
    print(f"Dataset: {config['dataset']}")
    print(f"Bag file: {config['bag']}")
    print(f"Output: {results_dir}")
    print("========================================\n")
    
    # Source ROS environment, launch rviz and run roslaunch
    bash_command = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/catkin_ws/devel/setup.bash && "
        + " ".join(launch_cmd)
    )

    log_file = os.path.join(results_dir, f"initialization_results_log.txt")
    match_string = "yaw ="
    # Log file
    temp_run_log = os.path.join(results_dir, f"temp_stdout_{run_id:02d}.log")
    
    # Run roslaunch
    # Modify bash_command to pipe output to tee and grep
    bash_command = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/catkin_ws/devel/setup.bash && "
        + " ".join(launch_cmd)
        ) 

    try:
        # Redirect stdout and stderr directly to a file at the OS level
        with open(temp_run_log, "w") as f_temp:
            subprocess.run(
                bash_command, 
                shell=True, 
                executable="/bin/bash", 
                stdout=f_temp, 
                stderr=subprocess.STDOUT, 
                check=True
            )
        
        # Now parse the file we just created
        with open(temp_run_log, "r") as f_in, open(log_file, "a") as f_out:
            f_out.write(f"\n--- Results for Run {run_id} ---\n")
            found_data = False
            for line in f_in:
                if "yaw =" in line:
                    f_out.write(line)
                    found_data = True
            
            if not found_data:
                f_out.write("No 'yaw =' data found in this run.\n")
            
        print(f"Processed results for Run {run_id+1}")

        # Cleanup the temporary terminal log
        if os.path.exists(temp_run_log):
            os.remove(temp_run_log)
        
        

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] roslaunch failed: {e}")

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
        subprocess.run(
            bash_command,
            shell=True,
            executable="/bin/bash",
            check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] roslaunch failed: {e}")
        raise
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"OpenVINS Multi-Run Script")
    print(f"Dataset: {DATASET}")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"{'='*60}\n")
    
    # Load dataset-specific configuration
    config = load_config(DATASET)
    uwb_config = load_uwb_config(DATASET, config["config_uwb"])
    full_config = {**config, "uwb_settings": uwb_config}
    
    # Ensure bag file exists
    check_bag_file(config)
    
    # Create results directory
    base_path = os.path.join(config["base_path"], config["config"])
    if DATASET == "miluv":
        base_path = os.path.join(base_path, config["dataset"])
    
    results_dir = make_results_dir(base_path, config["filename_est"])
    
    # Save config once
    config_file = os.path.join(results_dir, "config_used.yaml")
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False)
        print(f"[INFO] Config saved to: {config_file}\n")
    
    if not RUN_VIO:
        # Save initialization data
        log_file = os.path.join(results_dir, f"initialization_results_log.txt")
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"[INFO] Cleared old initialization log file: {log_file}")
    
    # Run multiple times 
    successful_runs = 0
    for run_id in range(NUM_RUNS):
        try:
            if RUN_VIO:
                run_openvins_sub(config, results_dir, run_id)
            else:
                run_uvio(config, results_dir, run_id)

            successful_runs += 1
        except Exception as e:
            print(f"\n[WARNING] Run {run_id + 1} failed: {e}")
    #     finally:
    #         # Clean up between runs (except after the last one)
    #         if run_id < NUM_RUNS - 1:
    #             cleanup_ros()
    #             time.sleep(1)
    
    # # Final cleanup
    # cleanup_ros()
    
    print(f"\n{'='*60}")
    print(f"Completed {successful_runs}/{NUM_RUNS} runs successfully")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")