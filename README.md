# Wasabi-MK1

Real-time **right-arm teleoperation**: a webcam tracks your arm and hand with
MediaPipe and drives a simulated 3-DoF robotic arm (+ wrist roll + gripper) in
MuJoCo. Built as the precision groundwork for driving the real arm later.
Inspired by the YouTube channel "Build Some Stuff".

```
Camera ─► MediaPipe (pose + right hand) ─► One-Euro filter ─► torso frame
       ─► analytic IK retargeting ─► MuJoCo sim  (+ optional UDP telemetry)
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Two windows open: the camera preview (with skeleton overlay and live joint
readout) and the MuJoCo arm. Stand or sit **facing the camera** with your whole
right arm visible; hips in frame improves accuracy but is not required.
The red sphere in the sim is the raw tracking target — the gripper chasing it
shows you the system's precision directly. Press **Q**/**ESC** to quit.

```bash
python main.py --kinematic         # zero-lag exact rendering (precision check)
python main.py --mode joint        # mimic your joint angles instead of hand position
python main.py --pose-model heavy  # most accurate pose model (slower, ~1x CPU)
python main.py --no-mirror         # disable mirror-image mapping
python main.py --udp 192.168.1.100:5005   # stream servo packets to the real arm
python main.py --no-sim            # camera/overlay only
```

Run the math test-suite (no camera needed):

```bash
python tests/test_retarget.py
```

## How precision is achieved

| Technique | Where | Why it matters |
|---|---|---|
| **Torso-anchored frame** | `retarget.py` | All angles are measured relative to your own shoulders/hips/nose, so leaning, turning, or standing off-axis no longer corrupts the output. |
| **Analytic IK (default mode)** | `retarget.py` | Your wrist position (scaled by an auto-calibrated arm-length ratio) is solved exactly for base-yaw + shoulder + elbow. The sim hand goes *where your hand is*, not where three independently-noisy angles happen to point. |
| **One-Euro filtering of landmarks** | `filters.py` | Adaptive low-pass tuned for teleop: still hand = rock solid, fast motion = no lag. Applied to 3-D positions *before* any angle math. |
| **Visibility gating** | `retarget.py` | Landmarks MediaPipe is guessing at (occluded arm, hips out of frame) are rejected instead of poisoning the output. |
| **Right-hand selection by proximity** | `tracking.py` | The hand detector result is matched to the pose model's right wrist, so it can never lock onto your left hand. |
| **Yaw singularity hold** | `retarget.py` | With the arm hanging near-vertical, azimuth is mathematically undefined; the base angle freezes smoothly instead of spinning. |
| **Wrist roll about the forearm axis** | `retarget.py` | Pronation/supination measured as a signed rotation about the actual forearm direction — camera-invariant, with unwrapping at ±180°. |
| **Slew-rate limiting** | `filters.py` | Hard cap on joint velocity — a safety layer that matters the day this drives real servos. |

## Files

| File | Role |
|---|---|
| `main.py` | Entry point, orchestration, HUD, UDP telemetry |
| `tracking.py` | Camera + MediaPipe Tasks (pose + hands), right-hand selection |
| `retarget.py` | Torso frame, IK/joint retargeting, wrist roll, grip, gating |
| `filters.py` | One-Euro filter, slew-rate limiter |
| `simulation/arm.xml` | MuJoCo model: yaw + shoulder + elbow + wrist roll + 2-finger gripper |
| `simulation/sim.py` | Sim driver (dynamics or zero-lag kinematic mode) |
| `tests/test_retarget.py` | Synthetic-pose tests pinning down every sign convention |

## Tuning

All knobs live in `RetargetConfig` (`retarget.py`):

* `min_cutoff` / `beta` — One-Euro smoothness vs responsiveness.
* `base_sign`, `roll_sign` — flip to ±1 if a joint feels inverted on your setup.
* `mirror` — mirror-image mapping (default; feels natural facing a camera).
* `l1`, `l2`, joint ranges — robot geometry; must match `simulation/arm.xml`.
* `pinch_closed` / `pinch_open` — gripper pinch calibration.

## Toward the real arm

`main.py --udp HOST:PORT` already emits the serial-style packet
`<BASE,SHOULDER,ELBOW,WRIST_P,WRIST_R,GRIPPER>\n` in integer servo degrees
(see `servo_packet()` for the angle mapping). The slew limiter and joint
clamps in `retarget.py` are the same safety layer the hardware will need.
