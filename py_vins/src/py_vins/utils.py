import yaml
import navlie as nav
import numpy as np

from pathlib import Path
from typing import List, Optional
from pymlg import SO3, SE3
from navlie.types import StateWithCovariance
from navlie.lib.states import SE3State, CompositeState, State, SO3State, VectorState
from navlie.utils.common import load_tum_trajectory
from navlie.utils.alignment import associate_and_align_trajectories

class OVState(CompositeState):
    """
    A custom Open Vins "composite" state object intended to hold a list of State objects as a single conceptual "state" of type (SO(3) x R(3)).
    """

    def __init__(self, state_list:List[State], stamp = None, state_id=None):
        if len(state_list) != 2:
            raise TypeError("OVState must be a 2 state composite state.")

        if not isinstance(state_list[0], SO3State):
            raise TypeError("First state of an OVState must be a SO3State")
        
        if not isinstance(state_list[1], VectorState):
            raise TypeError("Second state must be a VectorState of R(3)")
        
        super().__init__(state_list, stamp, state_id)

    @property
    def position(self):
        return self.value[1].value
    
    def get_position_as_state(self):
        return self.value[1]
    
    @property
    def attitude(self):
        return self.value[0].value
    
    def get_attitude_as_state(self):
        return self.value[0]
    
    def copy(self) -> "OVState":
        return super().copy()


def load_config(dataset: str):
    # __file__ -> .../py_vins/src/py_vins/utils.py
    # go up 3 levels to py_vins
    config_file = Path(__file__).resolve().parents[2] / "config" / f"{dataset}_config.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_tum_trajectory(fpath: str, direction = "right") -> List[OVState]:
    """
    Loads a TUM trajectory file into a list of Composite(SO(3) x R(3)) or SE3State objects.
    
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
        
        ori = SO3State(value=C_ab, stamp=data_row[0], direction=direction)
        pos = VectorState(value=position, stamp=data_row[0])
        state = OVState(state_list=[ori, pos], stamp=data_row[0])
        
        pose_list.append(state)

    return pose_list


def load_tum_covar_trajectory(fpath: str, direction = "right") -> List[StateWithCovariance]:
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
        
        ori = SO3State(value=C_ab, stamp=data_row[0], direction = direction)
        pos = VectorState(value=position, stamp=data_row[0])
        state = OVState(state_list=[ori, pos], stamp=data_row[0])

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
        
        pose_covar = StateWithCovariance(state=state, covariance=total_covar)
        assert isinstance(pose_covar.state, OVState)
        pose_covar_list.append(pose_covar)
    return pose_covar_list
        
def copy_covariance(covar: List[StateWithCovariance], traj: List[SE3State]) -> List[StateWithCovariance]:
    """
    Copies over the covariances of one trajectory to another.
    """
    copied_traj_covar = []
    traj_stamps = [x.stamp for x in traj]
    covar_stamps = [x.state.stamp for x in covar]
    indices = nav.associate_stamps(traj_stamps, covar_stamps)
    for i_state, i_cov in indices:
        cov = covar[i_cov].covariance.copy()
        state = traj[i_state].copy()
        if isinstance(state, SE3State):
            pos_state = VectorState(value=state.position, stamp=state.stamp, state_id="r"+str(i_state))
            ori_state = SO3State(value=state.attitude, stamp=state.stamp,state_id="C"+str(i_state))
            state = OVState(state_list=[ori_state, pos_state], stamp=state.stamp, state_id="x"+str(i_state))

        state.stamp = covar[i_cov].stamp
        newstate_covar = StateWithCovariance(state, cov)
        copied_traj_covar.append(newstate_covar)

    return copied_traj_covar




