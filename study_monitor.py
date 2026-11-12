#!/usr/bin/env python3
"""Study Monitor v3.0: individual eye calibration + moving-mean EAR + head nods.

The Raspberry Pi does the sensing and JSON construction; MQTT is PRESERVED
but only connects when --mqtt-host is specified.  No UNO Q is needed for tests.

  python study_monitor.py --source 0
  python study_monitor.py --source 0 --no-display --duration 65
  python study_monitor.py --source 0 --no-display --duration 65 --record-video runs/review.mp4
  python study_monitor.py --report runs/study_YYYYMMDD_HHMMSS.jsonl

The initial ~10 s samples BOTH eyes while normally open and estimates a neutral
head pitch.  Head motion by itself never proves drowsiness: a stationary bowed
head is logged but not interpreted as sleepy.  Repeat nods are logged, and only
corroborating eye evidence may change StudentState to Dozzy.

StudentState protocol is unchanged: Empty / Study / Dozzy.
"""

import argparse
from collections import Counter, deque
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import fmean, median
import time

VERSION = "3.0"
EYE_LEFT = (33, 160, 158, 133, 153, 144)
EYE_RIGHT = (362, 385, 387, 263, 373, 380)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_ear(points, ids, width, height):
    """Pixel-based EAR (not distorted by image aspect ratio)."""
    pts = [(points[i].x * width, points[i].y * height) for i in ids]
    horizontal = distance(pts[0], pts[3])
    if horizontal < 6.0:
        return None
    ratio = (distance(pts[1], pts[5]) + distance(pts[2], pts[4])) / (2 * horizontal)
    return ratio if math.isfinite(ratio) and 0.015 <= ratio <= 0.70 else None


def extract_pose(matrix):
    """Return pitch/yaw/roll degrees from MediaPipe's facial pose transform.

    Matrix is a rotation (canonical face coordinates -> camera coordinates).
    Relative pitch changes, not absolute angles, drive nod tracking.  Some
    camera installations have the opposite pitch direction, controllable via
    --nod-down-sign.  When no matrix is provided, nod tracking is unavailable.
    """
    if matrix is None:
        return None
    try:
        import numpy as np
        a = np.asarray(matrix, dtype=float).reshape((4, 4))
        if not np.isfinite(a).all():
            return None
        r = a[:3, :3]
        pitch = math.degrees(math.atan2(float(r[2, 1]), float(r[2, 2])))
        yaw = math.degrees(math.asin(max(-1., min(1., float(-r[2, 0])))))
        roll = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
        return pitch, yaw, roll
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def observe(landmarks, width, height, matrix=None):
    """Keep 'face found' separate from 'eye EAR reliable'."""
    pose = extract_pose(matrix)
    if not landmarks:
        return dict(face_present=False, status="NO_FACE", left=None, right=None,
                    pitch=None, yaw=None, roll=None)
    points = landmarks[0]
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    face_width = (max(xs) - min(xs)) * width
    obs = dict(face_present=True, status="OK", left=None, right=None,
               pitch=pose[0] if pose else None,
               yaw=pose[1] if pose else None,
               roll=pose[2] if pose else None,
               FaceWidth=round(face_width, 1))
    if face_width < 85:
        obs["status"] = "FACE_TOO_SMALL"
        return obs
    if min(xs) < 0.005 or max(xs) > 0.995 or min(ys) < 0.005 or max(ys) > 0.995:
        obs["status"] = "FACE_PARTIALLY_OUT_OF_FRAME"
        return obs
    if pose is not None and abs(obs["yaw"]) > 30:
        obs["status"] = "FACE_TURNED"
        return obs
    obs["left"] = eye_ear(points, EYE_LEFT, width, height)
    obs["right"] = eye_ear(points, EYE_RIGHT, width, height)
    if obs["left"] is None or obs["right"] is None:
        obs["status"] = "EYES_UNRELIABLE"
    return obs


class Calibrator:
    """Personalize both EAR baselines and the neutral head pose on every boot."""
    def __init__(self, seconds=10.0, fraction=0.70, min_samples=25):
        self.seconds, self.fraction, self.min_samples = seconds, fraction, min_samples
        self.reset()

    def reset(self):
        self.start = None
        self.last_sample = None
        self.eyes = []
        self.pitches = []
        self.yaws = []
        self.profile = None

    def elapsed(self, t):
        return 0.0 if self.start is None else max(0., t - self.start)

    def add(self, t, obs):
        if self.profile is not None:
            return
        if obs["status"] != "OK":
            if self.last_sample is not None and t - self.last_sample > 2.0:
                self.reset()
            return
        if self.start is None:
            self.start = t
        self.last_sample = t
        self.eyes.append((obs["left"], obs["right"]))
        if obs.get("pitch") is not None and (obs.get("yaw") is None or abs(obs["yaw"]) <= 20):
            self.pitches.append(obs["pitch"])
            self.yaws.append(obs["yaw"])
        if self.elapsed(t) < self.seconds or len(self.eyes) < self.min_samples:
            return
        baselines = []
        for ix in (0, 1):
            values = sorted(x[ix] for x in self.eyes)
            baselines.append(float(median(values[len(values) // 2:])))
        self.profile = {
            "version": VERSION,
            "EAR_coordinate_system": "pixel",
            "baseline_left": round(baselines[0], 5),
            "baseline_right": round(baselines[1], 5),
            "threshold_left": round(baselines[0] * self.fraction, 5),
            "threshold_right": round(baselines[1] * self.fraction, 5),
            "close_ratio": self.fraction,
            "calibration_seconds": self.seconds,
            "valid_samples": len(self.eyes),
            "baseline_pitch": round(float(median(self.pitches)), 4)
                              if len(self.pitches) >= 10 else None,
            "baseline_yaw": round(float(median(self.yaws)), 4)
                            if len(self.yaws) >= 10 else None,
            "pose_samples": len(self.pitches),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }


def read_profile(path):
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
        if profile.get("EAR_coordinate_system") != "pixel":
            raise ValueError("profile uses old EAR coordinate system")
        for name in ("threshold_left", "threshold_right"):
            value = float(profile[name])
            if not 0.015 < value < 0.70:
                raise ValueError(name + " outside accepted range")
            profile[name] = value
        # Older v2 profile has no pitch baseline: eye detection still works.
        if profile.get("baseline_pitch") is not None:
            profile["baseline_pitch"] = float(profile["baseline_pitch"])
        return profile
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"Invalid profile {path}: {e}") from e


class NodDetector:
    """Detect brief head-down-and-back cycles, not a fixed downward posture.

    Relative pitch is smoothed with a 5-point MOVING MEAN.  To avoid counting
    slow, deliberate reading/pondering posture, excursions held >2.5s do not
    count as a nod, and the head must return near its baseline.
    """
    def __init__(self, pitch_baseline=None, down_sign=-1, angle=13.0,
                 return_angle=6.0, window_seconds=20.0):
        self.base = pitch_baseline
        self.down_sign = down_sign
        self.angle = angle
        self.return_angle = return_angle
        self.window_seconds = window_seconds
        self.values = deque(maxlen=5)
        self.stage = "LEVEL"
        self.started = None
        self.peak = 0.0
        self.events = deque()
        self.all_nods = 0
        self.last_pose_t = None
        self.filtered_pitch = None
        self.down_angle = None
        self.movement = "POSE_UNAVAILABLE" if pitch_baseline is None else "STABLE"

    def update(self, t, pitch, yaw=None):
        while self.events and self.events[0] < t - self.window_seconds:
            self.events.popleft()
        if self.base is None or pitch is None or (yaw is not None and abs(yaw) > 30):
            self.values.clear()
            self.stage = "LEVEL"
            self.started = None
            self.last_pose_t = None
            self.filtered_pitch = None
            self.down_angle = None
            self.movement = "POSE_UNAVAILABLE" if pitch is None or self.base is None else "FACE_TURNED"
            return self.info()
        if self.last_pose_t is not None and t - self.last_pose_t > 0.8:
            self.values.clear()
            self.stage = "LEVEL"
            self.started = None
        self.last_pose_t = t
        self.values.append(float(pitch))
        self.filtered_pitch = fmean(self.values)
        self.down_angle = self.down_sign * (self.filtered_pitch - self.base)
        a = self.down_angle
        self.movement = "STABLE"
        if self.stage == "LEVEL":
            if a >= self.angle:
                self.stage = "DIPPING"
                self.started = t
                self.peak = a
                self.movement = "HEAD_DIPPING"
            elif a > self.return_angle:
                self.movement = "HEAD_DOWN_STATIC_OR_MOVING"
        elif self.stage == "DIPPING":
            self.peak = max(self.peak, a)
            elapsed = t - self.started
            if elapsed > 2.5:
                self.stage = "HELD"
                self.movement = "HEAD_DOWN_HELD_NOT_NOD"
            elif a <= self.return_angle:
                if 0.20 <= elapsed <= 2.5 and self.peak >= self.angle:
                    self.events.append(t)
                    self.all_nods += 1
                    self.movement = "NOD_DETECTED"
                self.stage = "LEVEL"
                self.started = None
            else:
                self.movement = "HEAD_DIPPING"
        else:  # HELD: do not count a prolonged reading/thinking posture.
            self.movement = "HEAD_DOWN_HELD_NOT_NOD"
            if a <= self.return_angle:
                self.stage = "LEVEL"
                self.started = None
                self.movement = "STABLE"
        return self.info()

    def info(self):
        return dict(
            HeadPitch=round(self.filtered_pitch, 2) if self.filtered_pitch is not None else None,
            HeadDownAngle=round(self.down_angle, 2) if self.down_angle is not None else None,
            HeadMovement=self.movement,
            NodCount=len(self.events),
            NodTotal=self.all_nods,
        )


class StudyMonitor:
    def __init__(self, profile, *, nod_sign=-1, close_seconds=1.2,
                 assisted_close_seconds=0.65, recovery_seconds=2.0,
                 no_face_timeout=4.0, nod_angle=13.0):
        self.profile = profile
        self.close_seconds = close_seconds
        self.assisted_close_seconds = assisted_close_seconds
        self.recovery_seconds = recovery_seconds
        self.no_face_timeout = no_face_timeout
        self.state = "Empty"
        self.blinks = 0
        self.last_face = None
        self.last_good = None
        self.last_valid_frame = None
        self.left_mean_window = deque(maxlen=3)
        self.right_mean_window = deque(maxlen=3)
        self.history = deque()
        self.previous_eye = None
        self.closed_since = None
        self.open_since = None
        self.perclos = 0.0
        self.doze_time = 0.0
        self.nods = NodDetector(profile.get("baseline_pitch"), down_sign=nod_sign,
                                angle=nod_angle)
        self.last_movement = self.nods.info()

    def _eye_reset(self):
        self.left_mean_window.clear()
        self.right_mean_window.clear()
        self.previous_eye = None
        self.closed_since = None
        self.open_since = None

    def snapshot(self, t, status, valid, reason, eye_status="UNKNOWN",
                 left=None, right=None, face=False):
        average_ear = (left + right) / 2 if left is not None and right is not None else None
        rec = {
            "StudentState": self.state,
            "DozeTime": round(self.doze_time, 1) if self.state == "Dozzy" else 0.0,
            "BlinkCount": self.blinks,
            "EAR": round(average_ear, 3) if average_ear is not None else None,
            "PERCLOS": round(self.perclos, 3),
            "EAR_Left": round(left, 3) if left is not None else None,
            "EAR_Right": round(right, 3) if right is not None else None,
            "EAR_Threshold_Left": self.profile["threshold_left"],
            "EAR_Threshold_Right": self.profile["threshold_right"],
            "EyeFilter": "moving_mean_3",
            "HeadFilter": "moving_mean_5",
            "EyeStatus": eye_status,
            "TrackingStatus": status,
            "DecisionValid": bool(valid),
            "DecisionReason": reason,
            "FaceDetected": bool(face),
        }
        rec.update(self.last_movement)
        return rec

    def update(self, t, obs):
        face = bool(obs["face_present"])
        status = obs["status"]
        if face:
            self.last_face = t
        # Pose processing is independent of eye reliability.
        self.last_movement = self.nods.update(t, obs.get("pitch"), obs.get("yaw"))

        if status != "OK":
            # Eyes cannot be measured, but a visible face is NOT an empty seat.
            self._eye_reset()
            self.last_valid_frame = None
            if not face and (self.last_face is None or t - self.last_face >= self.no_face_timeout):
                self.state = "Empty"
                self.doze_time = 0.0
                self.history.clear()
                self.perclos = 0.0
                return self.snapshot(t, "NO_FACE_TIMEOUT", False,
                                     "No face observed for 4s; cannot verify actual absence", face=False)
            if self.state == "Empty" and face:
                # A visible face confirms presence but not adequate eye evidence.
                self.state = "Study"
            reason = "Eyes/face unavailable; state held, decision NOT verified"
            if self.last_movement["NodCount"] >= 3:
                reason += "; repeat nod motion suspected"
            return self.snapshot(t, status, False, reason, face=face)

        left, right = obs["left"], obs["right"]
        if self.last_good is not None and t - self.last_good > 0.8:
            self._eye_reset()
        self.last_good = t
        self.left_mean_window.append(left)
        self.right_mean_window.append(right)
        filtered_l = fmean(self.left_mean_window)
        filtered_r = fmean(self.right_mean_window)
        th_l = self.profile["threshold_left"]
        th_r = self.profile["threshold_right"]
        closed = filtered_l <= th_l and filtered_r <= th_r
        opened = filtered_l >= th_l * 1.10 and filtered_r >= th_r * 1.10
        eye_status = "CLOSED" if closed else "OPEN" if opened else "AMBIGUOUS"
        if self.state == "Empty":
            self.state = "Study"
            self.doze_time = 0.0
        if not (closed or opened):
            self.previous_eye = None
            self.closed_since = None
            self.open_since = None
            self.last_valid_frame = None
            return self.snapshot(t, "EYE_STATE_AMBIGUOUS", False,
                                 "One eye closed or EAR near threshold", eye_status,
                                 filtered_l, filtered_r, True)

        # Valid consecutive samples only; do not call dropped frames eyelid closure.
        if self.previous_eye is not None:
            previous_t, previous_closed = self.previous_eye
            dt = t - previous_t
            if 0 < dt <= 0.25:
                self.history.append((t, dt, previous_closed))
        self.previous_eye = (t, closed)
        while self.history and self.history[0][0] < t - 60.0:
            self.history.popleft()
        sample_seconds = sum(dt for _, dt, _ in self.history)
        closed_seconds = sum(dt for _, dt, is_closed in self.history if is_closed)
        self.perclos = closed_seconds / sample_seconds if sample_seconds > 0 else 0.0

        if closed:
            if self.closed_since is None:
                self.closed_since = t
            self.open_since = None
        else:
            if self.closed_since is not None:
                period = t - self.closed_since
                if 0.08 <= period <= 0.50:
                    self.blinks += 1
                self.closed_since = None
            if self.open_since is None:
                self.open_since = t
        closed_for = (t - self.closed_since) if self.closed_since is not None else 0.0
        recent_nods = self.last_movement["NodCount"]
        assisted = recent_nods >= 1 and closed_for >= self.assisted_close_seconds
        high_perclos = closed and sample_seconds >= 10.0 and self.perclos >= 0.35
        drowsy = (closed_for >= self.close_seconds or assisted or high_perclos)
        reason = "Eyes open / monitoring" if opened else "Eyes closed; evaluating duration"

        # A stationary bowed head is NOT sleep evidence. Repeated nods alone
        # are likewise insufficient to tell deliberate nods from microsleep.
        if self.state == "Study" and drowsy:
            self.state = "Dozzy"
            self.doze_time = 0.0
            if closed_for >= self.close_seconds:
                reason = "Eyes closed longer than threshold"
            elif assisted:
                reason = "Eye closure supported by head nod"
            else:
                reason = "High PERCLOS and currently closed eyes"
        elif self.state == "Dozzy":
            if opened and self.open_since is not None and t - self.open_since >= self.recovery_seconds:
                self.state = "Study"
                self.doze_time = 0.0
                reason = "Eyes reopened for recovery interval"
            else:
                reason = "Dozzy; waiting for sustained recovery"
        elif self.last_movement["HeadMovement"] == "HEAD_DOWN_HELD_NOT_NOD":
            reason = "Static head lowered (thinking/reading possible); not drowsy by posture"
        elif recent_nods >= 2 and not drowsy:
            reason = "Repeated nods observed; eye evidence insufficient for Dozzy"

        if self.state == "Dozzy" and self.last_valid_frame is not None:
            valid_dt = t - self.last_valid_frame
            if 0 < valid_dt <= 0.25:
                self.doze_time += valid_dt
        self.last_valid_frame = t
        return self.snapshot(t, "OK", True, reason, eye_status,
                             filtered_l, filtered_r, True)


def summarize(path):
    path = Path(path)
    states, tracking, movement, reasons = Counter(), Counter(), Counter(), Counter()
    valid = total = 0
    max_doze = 0.0
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if "StudentState" not in row:
            continue
        total += 1
        states[row.get("StudentState")] += 1
        tracking[row.get("TrackingStatus")] += 1
        movement[row.get("HeadMovement")] += 1
        reasons[row.get("DecisionReason")] += 1
        valid += bool(row.get("DecisionValid"))
        max_doze = max(max_doze, float(row.get("DozeTime") or 0))
    print(f"[REPORT] {path} | {total} observations", flush=True)
    print(f"  DecisionValid: {valid}/{total} ({100*valid/total if total else 0:.1f}%)")
    print(f"  StudentState: {dict(states)}")
    print(f"  TrackingStatus: {dict(tracking)}")
    print(f"  HeadMovement: {dict(movement)}")
    print(f"  DecisionReason: {dict(reasons)}")
    print(f"  Maximum DozeTime (observed, excluding tracking gaps): {max_doze:.1f}s")
    print("  This is a diagnostic log, NOT an accuracy score without labelled trials.")


def overlay(frame, cv2, lines):
    for index, line in enumerate(lines):
        pt = (10, 27 + index * 25)
        cv2.putText(frame, line, pt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, pt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 255, 255), 1, cv2.LINE_AA)


def parse_args():
    p = argparse.ArgumentParser(description="Study Monitor v3; camera only by default")
    p.add_argument("--version", action="version", version=f"study_monitor.py v{VERSION}")
    p.add_argument("--source", default="0", help="Camera index or video file")
    p.add_argument("--model", default="face_landmarker.task")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Stop automatically after N seconds; 0 means indefinite")
    p.add_argument("--record-video", default=None, help="Optional annotated MP4 output")
    p.add_argument("--log-jsonl", default=None, help="Log JSONL file (default runs/study_*.jsonl)")
    p.add_argument("--report", default=None, help="Summarize a prior JSONL log")
    p.add_argument("--profile", default="ear_profile.json")
    p.add_argument("--no-calibration", action="store_true",
                   help="Use prior valid profile rather than recalibrating")
    p.add_argument("--calibration-seconds", type=float, default=10.0)
    p.add_argument("--ear-close-ratio", type=float, default=0.70)
    p.add_argument("--threshold", type=float, default=0.22,
                   help="Fallback EAR for --no-calibration with no saved profile")
    p.add_argument("--nod-down-sign", type=int, choices=(-1, 1), default=-1,
                   help="Sign of pitch when the head nods down; flip if test shows wrong sign")
    p.add_argument("--nod-angle", type=float, default=13.0)
    p.add_argument("--close-seconds", type=float, default=1.2)
    p.add_argument("--assisted-close-seconds", type=float, default=0.65)
    p.add_argument("--no-face-timeout", type=float, default=4.0)
    # Keep the original UNO Q MQTT/serial interface unchanged.
    p.add_argument("--mqtt-host", default=None,
                   help="Existing MQTT preserved; omit for local-only test")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--mqtt-topic", default="student/monitor")
    p.add_argument("--serial", default=None)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--simulate", action="store_true", help="No camera synthetic state-machine check")
    args = p.parse_args()
    if args.calibration_seconds <= 0 or not 0 < args.ear_close_ratio < 1:
        p.error("invalid calibration duration or EAR close ratio")
    if args.no_face_timeout <= 0 or args.close_seconds <= 0 or args.nod_angle <= 0:
        p.error("timeouts and nod angle must be positive")
    return args


def main():
    args = parse_args()
    if args.report:
        summarize(args.report)
        return
    log_path = Path(args.log_jsonl or "runs/study_" +
                    datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[START] study_monitor.py v{VERSION} | MQTT={'ON' if args.mqtt_host else 'OFF'} "
          f"| display={'OFF' if args.no_display else 'ON'}", flush=True)
    print(f"[LOG] {log_path}", flush=True)
    sender = None
    serial_port = None
    cap = None
    writer = None
    detector = None
    calibrator = None
    monitor = None
    profile = None
    last_sent = None
    last_notice = -1e9
    try:
        if args.mqtt_host:
            from mqtt_sender import MQTTSender
            sender = MQTTSender(host=args.mqtt_host, port=args.mqtt_port,
                                topic=args.mqtt_topic)
            sender.connect()
        if args.serial:
            import serial
            serial_port = serial.Serial(args.serial, args.baud, timeout=0.1)

        if args.no_calibration or args.simulate:
            if not args.simulate and Path(args.profile).exists():
                profile = read_profile(args.profile)
                print(f"[CALIBRATION] Loaded {args.profile}", flush=True)
            else:
                profile = {"threshold_left": args.threshold,
                           "threshold_right": args.threshold,
                           "baseline_pitch": None}
                print("[WARNING] Fixed EAR: NOT personalized; head-nod detection disabled",
                      flush=True)
            monitor = StudyMonitor(profile, nod_sign=args.nod_down_sign,
                                   nod_angle=args.nod_angle,
                                   close_seconds=args.close_seconds,
                                   assisted_close_seconds=args.assisted_close_seconds,
                                   no_face_timeout=args.no_face_timeout)
        else:
            calibrator = Calibrator(args.calibration_seconds, args.ear_close_ratio)
            print(f"[CALIBRATION] Look straight at camera with both eyes naturally OPEN "
                  f"for about {args.calibration_seconds:g} seconds", flush=True)

        with log_path.open("w", encoding="utf-8", buffering=1) as logfile:
            def emit(t, rec):
                nonlocal last_sent
                if last_sent is not None and t - last_sent < 0.5:
                    return
                record = dict(rec)
                record["LoggedAt"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
                msg = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                logfile.write(msg + "\n")
                print(msg, flush=True)
                if sender:
                    sender.send(record)
                if serial_port:
                    serial_port.write((msg + "\n").encode("utf-8"))
                last_sent = t

            if args.simulate:
                demo = StudyMonitor(dict(threshold_left=0.14, threshold_right=0.105,
                                         baseline_pitch=0), nod_sign=-1)
                for i in range(200):
                    t = i * 0.1
                    left, right = (0.10, 0.07) if 5 < t < 16 else (0.25, 0.19)
                    obs = dict(face_present=t >= 2, status="OK" if t >= 2 else "NO_FACE",
                               left=left if t >= 2 else None,
                               right=right if t >= 2 else None,
                               pitch=0., yaw=0.)
                    emit(t, demo.update(t, obs))
                print("[SIMULATION DONE]", flush=True)
                return

            import cv2
            import mediapipe as mp
            if not Path(args.model).exists():
                raise FileNotFoundError(f"Missing model: {args.model}; run inside ~/mediapipe_test")
            source = int(args.source) if args.source.isdigit() else args.source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Camera/video source failed: {args.source}")
            is_file = not isinstance(source, int)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1 <= fps <= 120 else 15.0
            frame_index = 0
            started_real = time.monotonic()
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=args.model),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while True:
                    if args.duration > 0 and time.monotonic() - started_real >= args.duration:
                        print("[STOP] Duration reached", flush=True)
                        break
                    ok, frame = cap.read()
                    if not ok:
                        print("[STOP] Camera read failed or video ended", flush=True)
                        break
                    t = frame_index / fps if is_file else time.monotonic()
                    frame_index += 1
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                        data=rgb))
                    matrices = getattr(detected, "facial_transformation_matrixes", None)
                    matrix = matrices[0] if matrices is not None and len(matrices) else None
                    obs = observe(detected.face_landmarks, w, h, matrix)
                    if calibrator is not None and calibrator.profile is None:
                        calibrator.add(t, obs)
                        line = (f"CALIBRATING {calibrator.elapsed(t):.1f}/"
                                f"{args.calibration_seconds:.1f}s samples={len(calibrator.eyes)} "
                                f"{obs['status']}")
                        if t - last_notice >= 1.0:
                            print("[CALIBRATION] " + line, flush=True)
                            logfile.write(json.dumps({"Event": "CALIBRATION",
                                                      "TrackingStatus": obs["status"],
                                                      "ValidSamples": len(calibrator.eyes),
                                                      "ElapsedSeconds": round(calibrator.elapsed(t), 1)}) + "\n")
                            last_notice = t
                        if calibrator.profile is not None:
                            profile = calibrator.profile
                            Path(args.profile).write_text(json.dumps(profile, indent=2) + "\n",
                                                          encoding="utf-8")
                            monitor = StudyMonitor(profile, nod_sign=args.nod_down_sign,
                                                   nod_angle=args.nod_angle,
                                                   close_seconds=args.close_seconds,
                                                   assisted_close_seconds=args.assisted_close_seconds,
                                                   no_face_timeout=args.no_face_timeout)
                            print("[CALIBRATION OK] " + json.dumps(profile), flush=True)
                        lines = [line, "OPEN BOTH EYES / KEEP HEAD NEUTRAL"]
                    else:
                        rec = monitor.update(t, obs)
                        emit(t, rec)
                        lines = [
                            f"v{VERSION} {rec['StudentState']} Doze:{rec['DozeTime']:.1f}s "
                            f"Blinks:{rec['BlinkCount']} Valid:{rec['DecisionValid']}",
                            f"EYE {rec['EyeStatus']} L:{rec['EAR_Left']}/{rec['EAR_Threshold_Left']:.3f} "
                            f"R:{rec['EAR_Right']}/{rec['EAR_Threshold_Right']:.3f}",
                            f"HEAD {rec['HeadMovement']} delta:{rec['HeadDownAngle']} "
                            f"nods:{rec['NodCount']}",
                            f"Tracking: {rec['TrackingStatus']}",
                        ]
                    if not args.no_display or args.record_video:
                        overlay(frame, cv2, lines)
                    if args.record_video:
                        if writer is None:
                            path = Path(args.record_video)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     fps, (w, h))
                            if not writer.isOpened():
                                raise RuntimeError(f"Could not create video: {path}")
                            print(f"[VIDEO] Recording {path}", flush=True)
                        writer.write(frame)
                    if not args.no_display:
                        cv2.imshow("Study Monitor v3", frame)
                        key = cv2.waitKey(1) & 0xff
                        if key == ord("q"):
                            break
                        if key == ord("c"):
                            calibrator = Calibrator(args.calibration_seconds, args.ear_close_ratio)
                            monitor = None
                            print("[CALIBRATION RESET] Look straight ahead with eyes open",
                                  flush=True)
    except KeyboardInterrupt:
        print("[STOP] Ctrl+C", flush=True)
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if serial_port is not None:
            serial_port.close()
        if sender is not None:
            sender.close()
        if not args.no_display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass
        print(f"[DONE] Log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
