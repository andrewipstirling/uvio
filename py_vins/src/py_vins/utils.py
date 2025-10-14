import yaml
import navlie as nav
import numpy as np

from pathlib import Path
from typing import List
from pymlg import SO3, SE3
from navlie.types import StateWithCovariance
from navlie.lib.states import SE3State
from navlie.utils.common import load_tum_trajectory


def load_config(dataset: str):
    # __file__ -> .../py_vins/src/py_vins/utils.py
    # go up 3 levels to py_vins
    config_file = Path(__file__).resolve().parents[2] / "config" / f"{dataset}_config.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_tum_trajectory(fpath: str) -> List[SE3State]:
    """Loads a TUM trajectory file into a list of SE3State objects.
    
    Each row in the file should have 8 entries separated by spaces with the 
    following format:
    timestamp px py pz qx qy qz qw

    where px, py, and pz are the position components of the robot pose,
    and qx, qy, qz, and qw are the quaternion components of the robot pose 
    corresponding to the DCM C_ab.
    """
    txt_file = np.loadtxt(fpath, delimiter=" ", comments="#")
    
    if txt_file.shape[1] < 8:
        raise ValueError("TUM trajectory file must have at least 8 columns")

    pose_list: List[SE3State] = []
    for i in range(txt_file.shape[0]):
        data_row = txt_file[i, :]
        position = data_row[1:4]
        quat = data_row[4:8]
        C_ab = SO3.from_quat(quat, order="xyzw")
        pose_list.append(SE3State(
            value=SE3.from_components(C_ab, position),
              stamp=data_row[0]))
    return pose_list


def load_tum_covar_trajectory(fpath: str) -> List[StateWithCovariance]:
    """
    Loads a TUM trajectory file into a list of SE3StateWithCovariance objects.
    
    Each row in the file should have 8 entries separated by spaces with the 
    following format:
    timestamp px py pz qx qy qz qw

    where px, py, and pz are the position components of the robot pose,
    and qx, qy, qz, and qw are the quaternion components of the robot pose 
    corresponding to the DCM C_ab.
    """
    txt_file = np.loadtxt(fpath, delimiter=" ", comments="#")
    
    if txt_file.shape[1] != 20:
        raise ValueError("TUM trajectory with covar file must have 20 columns")
    
    pose_covar_list: List[StateWithCovariance] = []
    for i in range(txt_file.shape[0]):
        data_row = txt_file[i, :]
        position = data_row[1:4]
        quat = data_row[4:8]
        covar_ori = data_row[8:14]
        covar_pos = data_row[14:]
        C_ab = SO3.from_quat(quat, order="xyzw")
        pose = SE3State(
            value=SE3.from_components(C_ab, position),
            stamp=data_row[0])
        total_covar = np.zeros((6,6))
        # Fill in orientation covar
        total_covar[0,0] = covar_ori[0]
        total_covar[0,1] = covar_ori[1]
        total_covar[0,2] = covar_ori[2]
        total_covar[1,0] = covar_ori[1]
        total_covar[1,1] = covar_ori[3]
        total_covar[1,2] = covar_ori[4]
        total_covar[2,0] = covar_ori[2]
        total_covar[2,1] = covar_ori[4]
        total_covar[2,2] = covar_ori[5]
        # Fill in position covar
        total_covar[3,3] = covar_pos[0]
        total_covar[3,4] = covar_pos[1]
        total_covar[3,5] = covar_pos[2]
        total_covar[4,3] = covar_pos[1]
        total_covar[4,4] = covar_pos[3]
        total_covar[4,5] = covar_pos[4]
        total_covar[5,3] = covar_pos[2]
        total_covar[5,4] = covar_pos[4]
        total_covar[5,5] = covar_pos[5]
        pose_covar = StateWithCovariance(state=pose, covariance=total_covar)
        pose_covar_list.append(pose_covar)
    return pose_covar_list
        
def copy_covariance(covar: List[StateWithCovariance], traj: List[SE3State]) -> List[StateWithCovariance]:
    """
    Copies over the covaraicnes of one trajectory to another.
    """
    copied_traj_covar = []
    traj_stamps = [x.stamp for x in traj]
    covar_stamps = [x.state.stamp for x in covar]
    indices = nav.associate_stamps(traj_stamps, covar_stamps)
    for i_state, i_cov in indices:
        cov = covar[i_cov].covariance.copy()
        newstate_covar = StateWithCovariance(traj[i_state].copy(), cov)
        copied_traj_covar.append(newstate_covar)

    return copied_traj_covar

