"""
simulation/sim.py — MuJoCo driver for the Wasabi-MK1 simulated arm.

Two drive modes:
  dynamics (default) : joint targets feed position actuators and physics is
                       stepped in real time — the realistic preview of how a
                       real servo arm will behave (including servo lag).
  kinematic          : joint angles are written directly into qpos, giving a
                       zero-lag, mathematically exact rendering of the
                       retargeted pose — use this to judge tracking precision.

The red mocap sphere shows the raw IK target so you can see exactly how well
the end-effector follows your hand.
"""

from __future__ import annotations

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

# World position of the shoulder joint — must match arm.xml.
SHOULDER_POS = np.array([0.0, 0.0, 0.66])

_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm.xml")

_JOINTS = ("base_yaw", "shoulder_pitch", "elbow", "wrist_roll",
           "finger_l", "finger_r")
_FINGER_CLOSED = 0.7   # finger joint angle (rad) at grip = 0


class ArmSim:
    """Owns the MuJoCo model, data, and passive viewer window."""

    def __init__(self, kinematic: bool = False):
        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.kinematic = kinematic

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self._qadr = [self.model.jnt_qposadr[self.model.joint(n).id]
                      for n in _JOINTS]
        self._mocap_id = self.model.body("target").mocapid[0]

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._last_step = time.perf_counter()

    def update(self, targets) -> None:
        """Push one frame of JointTargets into the sim and advance it."""
        finger = _FINGER_CLOSED * (1.0 - targets.grip)
        q = [targets.base, targets.shoulder, targets.elbow,
             targets.wrist_roll, finger, finger]

        if targets.target_xyz is not None:
            self.data.mocap_pos[self._mocap_id] = SHOULDER_POS + targets.target_xyz

        if self.kinematic:
            for adr, val in zip(self._qadr, q):
                self.data.qpos[adr] = val
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        else:
            self.data.ctrl[:len(q)] = q
            now = time.perf_counter()
            n_steps = int((now - self._last_step) / self.model.opt.timestep)
            self._last_step = now
            for _ in range(min(max(n_steps, 1), 100)):
                mujoco.mj_step(self.model, self.data)

        self.viewer.sync()

    @property
    def alive(self) -> bool:
        return self.viewer.is_running()

    def close(self) -> None:
        if self.viewer.is_running():
            self.viewer.close()
