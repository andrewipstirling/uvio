import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Define all parameters
params = {
    'mathtext.default': 'regular',  # Use regular font for math text
    'font.family': 'serif',         # Set font family
    'font.size': 14,                # Set font size
    'lines.linewidth': 2,           # Set default line width
    'axes.grid': True,              # Enable grid by default
    'grid.linestyle': '--'          # Set grid line style
}

# Update rcParams with all settings at once
plt.rcParams.update(params)

config_path = "/root/catkin_ws/src/uvio/config/miluv"
anchor_init_file = "uwb_anchors_miluv_init.yaml"
anchor_gt_file   = "uwb_anchors.yaml"

init_yaml_path  = os.path.join(config_path, anchor_init_file)
truth_yaml_path = os.path.join(config_path, anchor_gt_file)

def load_clean_yaml(path):
    """Loads YAML while skipping ROS-generated header lines."""
    clean_lines = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("YAML:") or line.startswith("%"):
                continue
            clean_lines.append(line)
    return yaml.safe_load("".join(clean_lines))


init_data  = load_clean_yaml(init_yaml_path)
truth_data = load_clean_yaml(truth_yaml_path)

# -------------------------------
# Extract anchor values
# -------------------------------

init_results = {}
truth_results = {}

for key, value in init_data.items():
    if key.startswith("anchor"):
        init_results[key[-1]] = {
            "p_AinG": np.array(value["p_AinG"], float),
            "cov": float(value["prior_p_AinG_cov"])
        }

for key, value in truth_data.items():
    if key.startswith("anchor"):
        truth_results[key[-1]] = {
            "p_AinG": np.array(value["p_AinG"], float)
        }

# -------------------------------
# Compute per-axis errors and 3σ
# -------------------------------

anchors = sorted(init_results.keys())
errors = []
rmse = []
sigma3 = []

for a in anchors:
    p_init  = init_results[a]["p_AinG"]
    p_truth = truth_results[a]["p_AinG"]
    cov     = init_results[a]["cov"]
    e = p_init - p_truth
    errors.append(e)
    rmse.append(np.linalg.norm(e))
    sigma3.append(3 * np.sqrt(cov))

errors = np.array(errors)   # shape (N,3)
sigma3 = np.array(sigma3)   # shape (N,)

x = np.arange(len(anchors))

# -------------------------------
# Plot per-axis errors + 3σ bounds
# -------------------------------

plt.figure(figsize=(12, 6))

for i, axis in enumerate(["x", "y", "z"]):
    plt.scatter(x, errors[:, i], label=fr'$\delta\xi_{{{axis}}}$')

plt.fill_between(x, -sigma3, sigma3,alpha = 0.3, color = "tab:purple", label=r"3$\sigma$ Bound")

plt.xticks(x, anchors)
plt.xlabel("Anchor")
plt.ylabel(r"Error, $\delta\mathbf{\xi}$ (m)")
plt.title(r"Anchor Initialization Errors with 3$\sigma$ Bounds")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

