"""
mujoco_sim.py — Self-contained 3-DoF MuJoCo arm simulator for Wasabi-MK1.

The arm is built from MuJoCo primitive geometries (capsules, cylinders, boxes)
so no external mesh or URDF files are required.

Joint layout (all revolute):
  joint_base      — pan   around world Z   →  mimics Base Pan servo
  joint_shoulder  — tilt  around local Y   →  mimics Shoulder Tilt servo
  joint_elbow     — bend  around local Y   →  mimics Elbow Tilt servo

Control mode: KINEMATIC (direct qpos assignment + mj_forward).
Writing joint angles directly to qpos and calling mj_forward gives instant,
zero-lag tracking — ideal for mirroring human arm motion in real time.
No actuators or physics stepping are used.

Coordinate convention (MuJoCo / world):
  +X  right   +Y  forward   +Z  up

Link lengths (tuneable — edit the fromto attributes in _XML):
  Upper arm  0.22 m    Forearm  0.19 m
"""

from __future__ import annotations

import mujoco
import numpy as np

# ── MuJoCo model description ──────────────────────────────────────────────────

_XML = """
<mujoco model="wasabi_mk1_3dof">

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.35 0.35 0.35"
               diffuse="0.85 0.85 0.85"
               specular="0.10 0.10 0.10"/>
    <map stiffness="100" shadowscale="0.4"/>
    <quality shadowsize="2048"/>
  </visual>

  <worldbody>

    <light name="top"  pos="0 -1.0 2.0" dir=" 0  1 -1"
           diffuse="0.90 0.90 0.90" specular="0.30 0.30 0.30"/>
    <light name="fill" pos="0  1.0 2.0" dir=" 0 -1 -1"
           diffuse="0.55 0.55 0.55" specular="0.10 0.10 0.10"/>

    <!-- Ground plane -->
    <geom name="floor" type="plane" size="2 2 0.1"
          rgba="0.20 0.20 0.20 1" contype="1" conaffinity="1"/>

    <!-- Pedestal -->
    <geom name="pedestal" type="box" size="0.07 0.07 0.045"
          pos="0 0 0.045" rgba="0.12 0.12 0.12 1"/>

    <!-- BASE BODY — pans around world Z -->
    <body name="base_body" pos="0 0 0.09">
      <joint name="joint_base" type="hinge" axis="0 0 1"
             range="-1.5708 1.5708"/>
      <geom name="base_disc" type="cylinder" size="0.048 0.022"
            rgba="0.18 0.18 0.18 1"/>
      <geom name="shoulder_bracket" type="box"
            size="0.018 0.038 0.030" pos="0 0 0.025"
            rgba="0.25 0.25 0.25 1"/>

      <!-- UPPER ARM BODY — shoulder tilt around local Y -->
      <body name="upper_arm_body" pos="0 0 0.052">
        <joint name="joint_shoulder" type="hinge" axis="0 1 0"
               range="0 1.5708"/>
        <geom name="upper_arm" type="capsule"
              fromto="0 0 0  0 0 0.22"
              size="0.022" rgba="0.18 0.45 0.85 1"/>
        <geom name="elbow_knuckle" type="sphere" size="0.028"
              pos="0 0 0.22" rgba="0.10 0.10 0.10 1"/>

        <!-- FOREARM BODY — elbow bend around local Y -->
        <body name="forearm_body" pos="0 0 0.22">
          <joint name="joint_elbow" type="hinge" axis="0 1 0"
                 range="0 2.2689"/>
          <geom name="forearm" type="capsule"
                fromto="0 0 0  0 0 0.19"
                size="0.018" rgba="0.18 0.75 0.45 1"/>
          <geom name="wrist_knuckle" type="sphere" size="0.022"
                pos="0 0 0.19" rgba="0.10 0.10 0.10 1"/>

          <!-- END EFFECTOR -->
          <body name="hand_body" pos="0 0 0.19">
            <geom name="palm" type="box"
                  size="0.025 0.012 0.030"
                  rgba="0.90 0.65 0.15 1"/>
            <site name="ee_site" pos="0 0 0.035" size="0.009"
                  rgba="1.0 0.15 0.15 1"/>
          </body>

        </body>
      </body>
    </body>

  </worldbody>

  <!-- No actuators: kinematic control sets qpos directly -->

</mujoco>
"""

# qpos indices — match order joints appear in the XML (breadth-first)
_IDX_BASE     = 0
_IDX_SHOULDER = 1
_IDX_ELBOW    = 2


class ArmSimulator:
    """
    MuJoCo 3-DoF arm driven by direct kinematic control.

    set_target() writes joint angles straight into qpos; step() calls
    mj_forward() to propagate FK so the viewer sees the new pose instantly.

    Typical usage
    -------------
    sim = ArmSimulator()
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            sim.set_target(base_rad, shoulder_rad, elbow_rad)
            sim.step()
            viewer.sync()
    """

    def __init__(self, substeps: int = 1) -> None:
        # substeps kept for API compatibility but unused in kinematic mode
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data  = mujoco.MjData(self.model)
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ── Control ───────────────────────────────────────────────────────────────

    def set_target(self,
                   base_rad:     float,
                   shoulder_rad: float,
                   elbow_rad:    float) -> None:
        """
        Directly set joint positions (radians).

        Joint ranges:
          base_rad     : [-π/2, π/2]   pan left / right
          shoulder_rad : [0,    π/2]   arm up (0) → arm horizontal (π/2)
          elbow_rad    : [0,    2.27]  arm straight (0) → fully bent (2.27)
        """
        self.data.qpos[_IDX_BASE]     = base_rad
        self.data.qpos[_IDX_SHOULDER] = shoulder_rad
        self.data.qpos[_IDX_ELBOW]    = elbow_rad

    def step(self) -> None:
        """Propagate forward kinematics so the viewer reflects the new pose."""
        mujoco.mj_forward(self.model, self.data)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def ee_position(self) -> np.ndarray:
        """End-effector position [x, y, z] in world frame (metres)."""
        return self.data.site_xpos[self._ee_site_id].copy()

    @property
    def joint_angles_deg(self) -> tuple[float, float, float]:
        """Current joint positions in degrees (base, shoulder, elbow)."""
        q = self.data.qpos
        return (
            float(np.degrees(q[_IDX_BASE])),
            float(np.degrees(q[_IDX_SHOULDER])),
            float(np.degrees(q[_IDX_ELBOW])),
        )
