#!/usr/bin/env python3
import os
import sys
import subprocess
import yaml

# ============================================================
# CONFIGURATION — change this one line to switch datasets!
# ============================================================
DATASET = "decar_husky"  # Options: "decar_husky" or "miluv"
SERIAL = False
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


def check_bag_file():
    """Verify that the bag file exists before running launch."""
    base_path = config["base_path"]
    base_path = os.path.join(base_path, config["config"])
    
    if DATASET == "miluv" or DATASET == "decar_husky":

        base_path = os.path.join(base_path, config["dataset"])
    
    bag_path = os.path.join(base_path, config["bag"])

    if not os.path.exists(bag_path):
        print(f"[ERROR] Bag file not found:\n  {bag_path}")
        print("Please check your dataset path or mount the correct folder.")
        sys.exit(1)


def run_openvins_sub(config):
    """Run the open_vins serial.launch using provided config."""
    # Construct the roslaunch command
    # Base roslaunch command
    launch_cmd = ["roslaunch", "ov_msckf", "subscribe.launch"]

    base_path = config["base_path"]
    base_path = os.path.join(base_path, config["config"])
    path_gt = config["path_gt"]
    
    if DATASET == "miluv" or DATASET == "decar_husky":
        base_path = os.path.join(base_path, config["dataset"])
        path_gt = os.path.join(base_path, "results", config["path_gt"])
        print("Ground-Truth Path: ", path_gt)
    # Just overwrite the estimated path name from the config file
    path_est = os.path.join(base_path, "results", "standard_vio.txt")
    path_time = os.path.join(base_path, "results" ,"standard_vio_timing.txt")
    bag = os.path.join(base_path, config["bag"])

     

    # ROS parameters as arguments
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
        "num_pts:=" + str(config["num_pts"])
    ]

    launch_cmd.extend(args)

    print("\n========================================")
    print(f"Launching OpenVINS for dataset: {config['dataset']}")
    print(f"Bag file: {config['bag']}")
    print("========================================\n")

    # Source ROS environment and run roslaunch
    bash_command = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/catkin_ws/devel/setup.bash && "
        + " ".join(launch_cmd)
    )

    # Run roslaunch
    try:
        subprocess.run(bash_command, shell=True, executable="/bin/bash", check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] roslaunch failed: {e}")
        sys.exit(1)

def run_openvins_serial(config):
    """Run the open_vins serial.launch using provided config."""
    # Construct the roslaunch command
    # Base roslaunch command
    launch_cmd = ["roslaunch", "ov_msckf", "serial.launch"]

    base_path = config["base_path"]
    base_path = os.path.join(base_path, config["config"])
    path_gt = config["path_gt"]
    
    if DATASET == "miluv"or DATASET == "decar_husky":
        base_path = os.path.join(base_path, config["dataset"])
        path_gt = os.path.join(base_path, "results", config["path_gt"])

    path_est = os.path.join(base_path, "results", "traj_estimate_serial.txt")
    path_time = os.path.join(base_path, "results", "traj_timing_serial.txt")
    bag = os.path.join(base_path, config["bag"])
    

    # ROS parameters as arguments
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
    ]

    launch_cmd.extend(args)

    print("\n========================================")
    print(f"Launching OpenVINS for dataset: {config['dataset']}")
    print(f"Bag file: {config['bag']}")
    print("========================================\n")

    # Source ROS environment and run roslaunch
    bash_command = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/catkin_ws/devel/setup.bash && "
        + " ".join(launch_cmd)
    )

    # Run roslaunch
    try:
        subprocess.run(bash_command, shell=True, executable="/bin/bash", check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] roslaunch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Load dataset-specific configuration
    config = load_config(DATASET)

    # Ensure bag file exists
    check_bag_file()

    # Run OpenVINS
    if SERIAL:
        run_openvins_serial(config)
    else:
        run_openvins_sub(config)
