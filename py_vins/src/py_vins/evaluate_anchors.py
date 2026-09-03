import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# Define all parameters
plt.style.use('seaborn-v0_8-whitegrid')
params = {
    'text.usetex': False,           # Use matplotlib's internal mathtext (no LaTeX install needed)
    'mathtext.fontset': 'cm',       # Computer Modern - mimics LaTeX's default math font
    'font.family': 'serif',         # Set font family
    'font.size': 14,                # Set font size
    'lines.linewidth': 2,           # Set default line width
    'axes.grid': True,              # Enable grid by default
    'axes.labelsize': 18,           # Controls x/y/z label size
    'grid.linestyle': '--'          # Set grid line style
}

# Update rcParams with all settings at once
plt.rcParams.update(params)

config_path = "/workspace/catkin_ws/src/uvio/config/miluv"
anchor_init_file = "uwb_anchors_slam_0.yaml"
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
def parse_cov(raw_cov):
    """
    Handles prior_p_AinG_cov as:
      - a scalar (isotropic variance, same on all 3 axes)
      - a 3-element list (per-axis variance, i.e. diagonal of the cov matrix)
      - a 9-element flat list or 3x3 nested list (full covariance matrix)
    Always returns a (3, 3) covariance matrix.
    """
    arr = np.array(raw_cov, dtype=float) if isinstance(raw_cov, (list, tuple, np.ndarray)) else None

    if arr is None:
        # scalar -> isotropic covariance
        return np.eye(3) * float(raw_cov)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    if arr.shape == (3,):
        return np.diag(arr)
    raise ValueError(f"Unrecognized covariance shape: {arr.shape}")


def cov_to_ellipsoid_axes(cov, n_sigma=3.0):
    """
    Eigendecomposes a 3x3 covariance matrix and returns:
      - radii: the ellipsoid semi-axis lengths (n_sigma * sqrt(eigenvalues))
      - R: rotation matrix whose columns are the corresponding eigenvectors,
           i.e. the ellipsoid's principal axes in world-frame coordinates.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)   # eigh: cov is symmetric PSD
    eigvals = np.clip(eigvals, 0, None)      # guard against tiny negative numerical noise
    radii = n_sigma * np.sqrt(eigvals)
    return radii, eigvecs


init_results = {}
truth_results = {}

for key, value in init_data.items():
    if key.startswith("anchor"):
        init_results[key[-1]] = {
            "p_AinG": np.array(value["p_AinG"], float),
            "cov": parse_cov(value["prior_p_AinG_cov"])   # always shape (3, 3)
        }

for key, value in truth_data.items():
    if key.startswith("anchor"):
        truth_results[key[-1]] = {
            "p_AinG": np.array(value["p_AinG"], float)
        }

# -------------------------------
# Compute per-axis errors and 3-sigma
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
    sigma3.append(3 * np.sqrt(np.diag(cov)))   # per-axis 3-sigma, for the 2D error plot below

errors = np.array(errors)   # shape (N,3)
sigma3 = np.array(sigma3)   # shape (N,3)

x = np.arange(len(anchors))

# -------------------------------
# Plot per-axis errors + 3-sigma bounds
# -------------------------------

plt.figure(figsize=(12, 6))

for i, axis in enumerate(["x", "y", "z"]):
    plt.scatter(x, errors[:, i], label=fr'$\delta\xi_{{{axis}}}$')

plt.fill_between(x, -sigma3[:, 0], sigma3[:, 0], alpha=0.3, color="tab:purple", label=r"3$\sigma$ Bound")

plt.xticks(x, anchors)
plt.xlabel("Anchor")
plt.ylabel(r"Error, $\delta\mathbf{\xi}$ (m)")
# plt.title(r"Anchor Initialization Errors with 3$\sigma$ Bounds")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# Plot init vs. GT anchor positions in 3D, with 3-sigma
# covariance ellipsoids around the init positions
# -------------------------------

def plot_ellipsoid(ax, center, radii, R, color="tab:red", alpha=0.15,
                    n_theta=24, n_phi=12, phi_max=np.pi):
    """
    Draws a partially-visible ellipsoid surface centered at `center`.

    `radii` are the semi-axis lengths (n_sigma * sqrt(eigenvalues) of the
    covariance matrix) and `R` is the rotation matrix whose columns are the
    corresponding eigenvectors, so the ellipsoid is oriented along the true
    principal axes of the (possibly correlated) covariance rather than
    assuming it's axis-aligned with x/y/z.

    phi_max < pi clips the ellipsoid so the far side stays visible
    (a "cut-away" look rather than a fully opaque blob).
    """
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, phi_max, n_phi)
    theta, phi = np.meshgrid(theta, phi)

    # Unit sphere in the eigenbasis, scaled by the semi-axis lengths
    xs = radii[0] * np.sin(phi) * np.cos(theta)
    ys = radii[1] * np.sin(phi) * np.sin(theta)
    zs = radii[2] * np.cos(phi)

    # Rotate into world frame: [x,y,z]_world = R @ [x,y,z]_eigenbasis, then translate
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=0)   # (3, n_theta*n_phi)
    pts_world = R @ pts
    xs = pts_world[0].reshape(xs.shape) + center[0]
    ys = pts_world[1].reshape(ys.shape) + center[1]
    zs = pts_world[2].reshape(zs.shape) + center[2]

    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha,
                     linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(xs, ys, zs, color=color, alpha=alpha + 0.15, linewidth=0.4)


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

for a in anchors:
    p_init  = init_results[a]["p_AinG"]
    p_truth = truth_results[a]["p_AinG"]
    cov     = init_results[a]["cov"]
    radii, R = cov_to_ellipsoid_axes(cov, n_sigma=3.0)

    # Ground-truth position (green), with z-line to the ground
    ax.scatter(*p_truth, s=80, color="tab:green", label="Ground-Truth" if a == anchors[0] else None)
    ax.text(p_truth[0], p_truth[1], p_truth[2], f"  {a}", fontsize=12)
    ax.plot([p_truth[0], p_truth[0]], [p_truth[1], p_truth[1]], [0, p_truth[2]],
            color="tab:green", linestyle="--", linewidth=1.5)

    # Initialized position (red), with its own z-line to the ground
    ax.scatter(*p_init, s=80, color="tab:red", label="SLAM Estimate" if a == anchors[0] else None)
    ax.plot([p_init[0], p_init[0]], [p_init[1], p_init[1]], [0, p_init[2]],
            color="tab:red", linestyle=":", linewidth=1.5)

    # Partial 3-sigma covariance ellipsoid around the init position,
    # oriented along the eigenvectors of the estimate's covariance matrix
    plot_ellipsoid(ax, p_init, radii, R, color="tab:red", alpha=0.15)

ax.set_xlabel(r"$x \; [\mathrm{m}]$", labelpad=8)
ax.set_ylabel(r"$y \; [\mathrm{m}]$", labelpad=8)
ax.set_zlabel(r"$z \; [\mathrm{m}]$", labelpad=8)
ax.set_zbound(0, 2.0)
ax.zaxis.set_major_locator(MultipleLocator(0.5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.set_title(r"Anchor Init vs. GT Positions with 3$\sigma$ Covariance Ellipsoids")
ax.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# Plot GT anchor positions in 3D
# -------------------------------

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# for a in anchors:
#     p = truth_results[a]["p_AinG"]
#     ax.scatter(p[0], p[1], p[2], s=80, color="tab:blue")
#     ax.text(p[0], p[1], p[2], f"  {a}", fontsize=12)
#     ax.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color="tab:blue", linestyle="--", linewidth=1.5)

# ax.set_xlabel(r"$x \; [\mathrm{m}]$", labelpad=8)
# ax.set_ylabel(r"$y \; [\mathrm{m}]$", labelpad=8)
# ax.set_zlabel(r"$z \; [\mathrm{m}]$", labelpad=8)
# ax.set_zbound(0, 2.0)
# ax.zaxis.set_major_locator(MultipleLocator(0.5))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# plt.tight_layout()
# plt.show()

