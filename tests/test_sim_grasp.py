"""
tests/test_sim_grasp.py — Headless pick-up test for the MuJoCo scene.

Scripts the exact sequence a teleoperator performs: reach to the cube with the
gripper open, pinch, lift. Verifies the cube actually rises with the hand —
i.e., contacts, friction, and actuator strength are tuned well enough for a
real grasp, not just a pretty render.

Run with:  python tests/test_sim_grasp.py   (or pytest)
"""

import math
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retarget import solve_ik, RetargetConfig
from simulation.sim import SHOULDER_POS

_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "simulation", "arm.xml")


def _settle(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def _reach(data, target_world, grip_ctrl, cfg):
    rel = np.asarray(target_world) - SHOULDER_POS
    q1, q2, q3, _ = solve_ik(rel, cfg.l1, cfg.l2, prev_yaw=0.0,
                             lock_lo=cfg.yaw_lock_lo, lock_hi=cfg.yaw_lock_hi)
    data.ctrl[:6] = [q1, q2, q3, 0.0, grip_ctrl, grip_ctrl]


def test_pick_up_cube():
    cfg = RetargetConfig()
    model = mujoco.MjModel.from_xml_path(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    cube_qadr = model.jnt_qposadr[model.joint("cube_free").id]

    def cube_pos():
        return data.qpos[cube_qadr:cube_qadr + 3].copy()

    start = cube_pos()
    assert abs(start[2] - 0.369) < 0.01, f"cube not on stand: {start}"

    # 1. Reach: EE to the cube center, fingers open.
    _reach(data, start, grip_ctrl=0.0, cfg=cfg)
    _settle(model, data, 2.0)
    ee = data.site("ee").xpos
    assert np.linalg.norm(ee - cube_pos()) < 0.03, \
        f"EE did not reach cube: ee={ee}, cube={cube_pos()}"
    assert abs(cube_pos()[2] - start[2]) < 0.02, "cube knocked off during approach"

    # 2. Pinch.
    data.ctrl[4] = data.ctrl[5] = 0.7
    _settle(model, data, 0.8)

    # 3. Lift 15 cm and hold.
    _reach(data, start + np.array([0.0, 0.0, 0.15]), grip_ctrl=0.7, cfg=cfg)
    _settle(model, data, 1.5)

    lifted = cube_pos()
    assert lifted[2] > start[2] + 0.08, \
        f"cube did not lift: started z={start[2]:.3f}, now z={lifted[2]:.3f}"
    print(f"  grasp OK: cube lifted {100 * (lifted[2] - start[2]):.1f} cm")


if __name__ == "__main__":
    test_pick_up_cube()
    print("\n1/1 tests passed")
