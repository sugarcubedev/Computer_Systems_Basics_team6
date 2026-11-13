"""Study Monitor (eye-only baseline v4.0) for Raspberry Pi 5.

Based on the user's original working study_monitor.py.
Current scope:
  1. Ten seconds of person-specific LEFT/RIGHT eye-open calibration.
  2. Reliable transition from Dozzy back to Study after eyes reopen.
  3. EAR smoothing (3-frame moving mean) and visible eye landmarks.
  4. Honest UNKNOWN tracking status; PERCLOS is diagnostic only.

MQTT and optional serial interfaces from the original are preserved.
No head-pose estimation, Arduino LCD changes, or auto-start configuration.

Commands:
    python study_monitor.py --source 0
    python study_monitor.py --source 0 --no-display
    python study_monitor.py --simulate
    python study_monitor.py --version
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import time

from mqtt_sender import MQTTSender

VERSION = '4.0-eye-only'
EYE_LEFT = (33, 160, 158, 133, 153, 144)
EYE_RIGHT = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(points, ids, width, height):
    """EAR from pixel distances, correcting non-square frame aspect ratio."""
    p = [(points[i].x * width, points[i].y * height) for i in ids]

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    horizontal = dist(p[0], p[3])
    if horizontal < 5.0:
        return None
    value = (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * horizontal)
    if not math.isfinite(value) or not (0.025 <= value <= 0.75):
        return None
    return value


def read_eyes(points, width, height):
    face_width = (max(pt.x for pt in points) - min(pt.x for pt in points)) * width
    if face_width < 85.0:
        return None, None, 'FACE_TOO_SMALL'
    left = eye_aspect_ratio(points, EYE_LEFT, width, height)
    right = eye_aspect_ratio(points, EYE_RIGHT, width, height)
    if left is None or right is None:
        return None, None, 'EYE_GEOMETRY_UNRELIABLE'
    return left, right, 'OK'


class EyeCalibrator:
    def __init__(self, duration=10.0, close_ratio=0.70, open_ratio=0.80,
                 min_samples=25):
        self.duration = duration
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.min_samples = min_samples
        self.started = None
        self.last_valid = None
        self.left_samples = []
        self.right_samples = []
        self.profile = None

    @property
    def ready(self):
        return self.profile is not None

    def remaining(self, t):
        if self.started is None:
            return self.duration
        return max(0.0, self.duration - (t - self.started))

    def reset(self):
        self.started = None
        self.last_valid = None
        self.left_samples.clear()
        self.right_samples.clear()

    def observe(self, t, left, right):
        if self.ready:
            return False
        if left is None or right is None:
            if self.last_valid is not None and t - self.last_valid > 1.0:
                self.reset()
            return False
        if self.last_valid is not None and t - self.last_valid > 1.0:
            self.reset()
        if self.started is None:
            self.started = t
        self.last_valid = t
        self.left_samples.append(left)
        self.right_samples.append(right)
        if self.remaining(t) > 0 or len(self.left_samples) < self.min_samples:
            return False

        # Upper-half median reduces contamination from natural blinks during setup.
        def open_baseline(values):
            sorted_values = sorted(values)
            return float(median(sorted_values[len(sorted_values) // 2:]))

        left_open = open_baseline(self.left_samples)
        right_open = open_baseline(self.right_samples)
        self.profile = {
            'version': VERSION,
            'baseline_left': round(left_open, 5),
            'baseline_right': round(right_open, 5),
            'threshold_left': round(left_open * self.close_ratio, 5),
            'threshold_right': round(right_open * self.close_ratio, 5),
            'reopen_threshold_left': round(left_open * self.open_ratio, 5),
            'reopen_threshold_right': round(right_open * self.open_ratio, 5),
            'close_ratio': self.close_ratio,
            'open_ratio': self.open_ratio,
            'calibration_seconds': self.duration,
            'valid_samples': len(self.left_samples),
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'note': 'Open-eye only calibration; CLOSED accuracy needs real camera validation.'
        }
        return True


class StudyMonitor:
    """Eye-only state machine. PERCLOS never drives state transitions."""

    def __init__(self, profile, closed_seconds=10.0, recovery_seconds=0.5,
                 mean_window=3, no_face_seconds=4.0):
        self.profile = profile
        self.closed_seconds = closed_seconds
        self.recovery_seconds = recovery_seconds
        self.no_face_seconds = no_face_seconds
        self.close_left = profile['threshold_left']
        self.close_right = profile['threshold_right']
        self.open_left = profile['reopen_threshold_left']
        self.open_right = profile['reopen_threshold_right']
        self.mean_left = deque(maxlen=mean_window)
        self.mean_right = deque(maxlen=mean_window)
        self.state = 'Empty'
        self.last_face_t = None
        self.last_valid_t = None
        self.last_eye_label = 'UNKNOWN'
        self.closed_duration = 0.0
        self.open_duration = 0.0
        self.blinks = 0
        self.history = deque()
        self.history_prev = None
        self.perclos = 0.0
        self.last_smoothed = (None, None)
        self.last_tracking = 'STARTING'
        self.last_reason = 'Awaiting eye measurements'

    def _get_eye_status(self, left, right):
        if left is None or right is None:
            return 'UNKNOWN', None, None
        self.mean_left.append(left)
        self.mean_right.append(right)
        avg_left = sum(self.mean_left) / len(self.mean_left)
        avg_right = sum(self.mean_right) / len(self.mean_right)
        if avg_left <= self.close_left and avg_right <= self.close_right:
            label = 'CLOSED'
        elif avg_left >= self.open_left and avg_right >= self.open_right:
            label = 'OPEN'
        else:
            label = 'UNCERTAIN'
        return label, avg_left, avg_right

    def _update_perclos(self, t, eye):
        if self.history_prev is not None:
            prev_t, prev_eye = self.history_prev
            dt = t - prev_t
            if 0 < dt <= 0.35 and prev_eye in ('OPEN', 'CLOSED') and eye == prev_eye:
                self.history.append((t, dt, prev_eye == 'CLOSED'))
        self.history_prev = (t, eye)
        while self.history and self.history[0][0] < t - 60.0:
            self.history.popleft()
        observed = sum(row[1] for row in self.history)
        self.perclos = sum(row[1] for row in self.history if row[2]) / observed if observed else 0.0

    def update(self, t, face_present, left=None, right=None, tracking='OK'):
        if face_present:
            self.last_face_t = t
        eye, smooth_left, smooth_right = self._get_eye_status(left, right)
        if eye == 'UNKNOWN' and self.last_valid_t is not None and t - self.last_valid_t > 0.5:
            self.mean_left.clear()
            self.mean_right.clear()
        self.last_smoothed = (smooth_left, smooth_right)
        self.last_tracking = tracking
        self._update_perclos(t, eye)

        if eye in ('OPEN', 'CLOSED'):
            # Intermittent UNKNOWN/UNCERTAIN frames <=0.35 s may be ignored,
            # but their full duration is not credited as eye-open/closed time.
            dt = 0.0 if self.last_valid_t is None else t - self.last_valid_t
            same_eyes = eye == self.last_eye_label and 0 < dt <= 0.35
            credit = min(dt, 0.15) if same_eyes else 0.0
            if eye == 'CLOSED':
                self.closed_duration = self.closed_duration + credit if same_eyes else 0.0
                self.open_duration = 0.0
            else:
                if self.last_eye_label == 'CLOSED' and 0.08 <= self.closed_duration <= 0.45:
                    self.blinks += 1
                self.open_duration = self.open_duration + credit if same_eyes else 0.0
                self.closed_duration = 0.0
            self.last_eye_label = eye
            self.last_valid_t = t

            if self.state == 'Empty':
                self.state = 'Study'
            if eye == 'CLOSED' and self.closed_duration >= self.closed_seconds:
                self.state = 'Dozzy'
                self.last_reason = 'Eyes continuously CLOSED for alert threshold'
            elif eye == 'OPEN' and self.state == 'Dozzy' and self.open_duration >= self.recovery_seconds:
                self.state = 'Study'
                self.last_reason = 'Eyes reliably OPEN for recovery threshold'
            elif self.state == 'Study':
                self.last_reason = ('Eye OPEN' if eye == 'OPEN' else 'Eyes CLOSED, alert threshold not reached')
        else:
            if self.last_valid_t is None or t - self.last_valid_t > 0.5:
                self.closed_duration = 0.0
                self.open_duration = 0.0
                self.last_eye_label = 'UNKNOWN'
            self.last_reason = f'No reliable eye decision: {tracking if eye == "UNKNOWN" else eye}'
            if not face_present and (self.last_face_t is None or t - self.last_face_t >= self.no_face_seconds):
                self.state = 'Empty'
                self.closed_duration = 0.0
                self.open_duration = 0.0
                self.last_eye_label = 'UNKNOWN'
                self.last_reason = 'No face for timeout period'

        return self.snapshot(t, eye)

    def snapshot(self, t, eye):
        smooth_left, smooth_right = self.last_smoothed
        # Important: DozeTime is the total continuous eye-closure episode duration
        # once Dozzy, NOT time since Dozzy classification. This maintains the
        # original UNO Q MCU alert rule (Dozzy and DozeTime > 10s) without
        # imposing an unintended second 10-second delay.
        doze_time = self.closed_duration if self.state == 'Dozzy' else 0.0
        valid = eye in ('OPEN', 'CLOSED')
        ratio = ((smooth_left + smooth_right) / 2 if smooth_left is not None and smooth_right is not None else None)
        return {
            'StudentState': self.state,
            'DozeTime': round(doze_time, 2),
            'BlinkCount': self.blinks,
            'EAR': round(ratio, 4) if ratio is not None else None,
            'PERCLOS': round(self.perclos, 3),
            'EAR_Left': round(smooth_left, 4) if smooth_left is not None else None,
            'EAR_Right': round(smooth_right, 4) if smooth_right is not None else None,
            'EAR_Threshold_Left': self.close_left,
            'EAR_Threshold_Right': self.close_right,
            'EAR_Reopen_Left': self.open_left,
            'EAR_Reopen_Right': self.open_right,
            'EyeStatus': eye,
            'ClosedSeconds': round(self.closed_duration, 2),
            'OpenSeconds': round(self.open_duration, 2),
            'TrackingStatus': self.last_tracking,
            'DecisionValid': valid,
            'DecisionReason': self.last_reason,
        }


def overlay_line(cv2, frame, msg, row, color=(255, 255, 255)):
    y = 27 + row * 29
    cv2.putText(frame, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.59, color, 2, cv2.LINE_AA)


def paint_eyes(cv2, frame, points, color):
    h, w = frame.shape[:2]
    for ids in (EYE_LEFT, EYE_RIGHT):
        # Connect the measured contour and show original landmark points.
        line = [ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[0]]
        coords = [(int(points[j].x * w), int(points[j].y * h)) for j in line]
        for a, b in zip(coords, coords[1:]):
            cv2.line(frame, a, b, color, 1, cv2.LINE_AA)
        for j in ids:
            p = points[j]
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, color, -1)


def build_parser():
    p = argparse.ArgumentParser(description='Eye-only Study Monitor, 10 s calibration + 10 s closure alert')
    p.add_argument('--source', default='0', help='camera index or video path')
    p.add_argument('--model', default='face_landmarker.task')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--duration', type=float, default=None, help='stop after this many seconds (optional)')
    p.add_argument('--calibration-seconds', type=float, default=10.0)
    p.add_argument('--ear-close-ratio', type=float, default=0.70)
    p.add_argument('--ear-open-ratio', type=float, default=0.80)
    p.add_argument('--closed-seconds', type=float, default=10.0)
    p.add_argument('--recovery-seconds', type=float, default=0.5)
    p.add_argument('--mean-window', type=int, default=3)
    p.add_argument('--profile', default='ear_profile.json')
    p.add_argument('--log-dir', default='runs')
    p.add_argument('--no-calibration', action='store_true', help='fallback to fixed --threshold; not recommended')
    p.add_argument('--threshold', type=float, default=0.22)
    p.add_argument('--mqtt-host', default=None, help='UNO Q broker (default: MQTT OFF, original sender preserved)')
    p.add_argument('--mqtt-port', type=int, default=1883)
    p.add_argument('--mqtt-topic', default='student/monitor')
    p.add_argument('--serial', default=None)
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--simulate', action='store_true', help='test status machine without camera or MQTT')
    p.add_argument('--version', action='version', version=f'study_monitor.py v{VERSION}')
    return p


def run_simulation(args):
    profile = {
        'threshold_left': 0.14, 'threshold_right': 0.126,
        'reopen_threshold_left': 0.16, 'reopen_threshold_right': 0.144,
    }
    mon = StudyMonitor(profile, args.closed_seconds, args.recovery_seconds, args.mean_window)
    previous = mon.state
    for i in range(200):
        t = i * 0.1
        if t < 1:
            record = mon.update(t, False, tracking='NO_FACE')
        elif t < 4 or t >= 16:
            record = mon.update(t, True, left=0.20, right=0.18)
        else:
            record = mon.update(t, True, left=0.07, right=0.065)
        if record['StudentState'] != previous:
            print(f'[SIM STATE] {previous} -> {record["StudentState"]} at {t:.1f}s, DozeTime={record["DozeTime"]}')
            previous = record['StudentState']
    print('[SIMULATION FINISHED] No camera, network, or Arduino accessed.')


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not (0 < args.ear_close_ratio < args.ear_open_ratio < 1.0):
        parser.error('Require 0 < close ratio < open ratio < 1')
    if args.calibration_seconds <= 0 or args.closed_seconds <= 0 or args.recovery_seconds <= 0:
        parser.error('Time thresholds must be positive')
    if not (1 <= args.mean_window <= 15):
        parser.error('--mean-window must be from 1 to 15')
    if args.simulate:
        run_simulation(args)
        return

    import cv2
    import mediapipe as mp
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing model file: {model_path}. Run from ~/mediapipe_test.')
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera/video source: {args.source}')
    file_input = not isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) if file_input else 0.0
    fps = fps if 1 <= fps <= 240 else 30.0
    frame_index = 0

    calibrator = None if args.no_calibration else EyeCalibrator(
        duration=args.calibration_seconds,
        close_ratio=args.ear_close_ratio,
        open_ratio=args.ear_open_ratio)
    mon = None
    if args.no_calibration:
        profile = {
            'threshold_left': args.threshold, 'threshold_right': args.threshold,
            'reopen_threshold_left': args.threshold * (args.ear_open_ratio / args.ear_close_ratio),
            'reopen_threshold_right': args.threshold * (args.ear_open_ratio / args.ear_close_ratio),
        }
        mon = StudyMonitor(profile, args.closed_seconds, args.recovery_seconds, args.mean_window)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'study_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    profile_path = Path(args.profile)
    sender = None
    port = None
    if args.mqtt_host:
        sender = MQTTSender(host=args.mqtt_host, port=args.mqtt_port, topic=args.mqtt_topic)
    last_log = None
    last_state = None
    last_cal_notice = -1
    start_wall = time.monotonic()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1)
    print(f'[START] Study Monitor v{VERSION}', flush=True)
    print(f'[CALIBRATION] Look straight at camera with eyes naturally OPEN for {args.calibration_seconds:g}s.', flush=True)
    print(f'[LOG] {log_path}', flush=True)
    try:
        if sender:
            sender.connect()
        if args.serial:
            import serial
            port = serial.Serial(args.serial, args.baud, timeout=0.1)
            time.sleep(2)
        with log_path.open('w', encoding='utf-8') as logfile:
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while True:
                    if args.duration is not None and time.monotonic() - start_wall >= args.duration:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        print('[CAMERA] No more frames.', flush=True)
                        break
                    t = frame_index / fps if file_input else time.monotonic()
                    frame_index += 1
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    found = bool(result.face_landmarks)
                    points = result.face_landmarks[0] if found else None
                    left = right = None
                    track = 'NO_FACE'
                    if found:
                        left, right, track = read_eyes(points, w, h)

                    if calibrator is not None and not calibrator.ready:
                        became_ready = calibrator.observe(t, left, right)
                        if became_ready:
                            profile_path.write_text(json.dumps(calibrator.profile, indent=2) + '\n', encoding='utf-8')
                            mon = StudyMonitor(calibrator.profile, args.closed_seconds,
                                               args.recovery_seconds, args.mean_window)
                            print('[CALIBRATION OK] ' + json.dumps(calibrator.profile), flush=True)
                            print('[TEST] OPEN -> CLOSED 3s -> OPEN (Study); then CLOSED 11s -> OPEN (Study).', flush=True)
                        else:
                            remaining = calibrator.remaining(t)
                            notice = int(remaining)
                            if notice != last_cal_notice:
                                print(f'[CALIBRATION] {remaining:.1f}s remaining; samples={len(calibrator.left_samples)}; {track}', flush=True)
                                last_cal_notice = notice
                            if not args.no_display:
                                if points is not None:
                                    paint_eyes(cv2, frame, points, (0, 255, 255))
                                overlay_line(cv2, frame, f'CALIBRATE: EYES OPEN {remaining:.1f}s', 0, (0, 255, 255))
                                overlay_line(cv2, frame, f'Face: {track} / samples: {len(calibrator.left_samples)}', 1)
                                cv2.imshow('Study Monitor - Eye Test', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            continue

                    record = mon.update(t, found, left, right, track)
                    record['LoggedAt'] = datetime.now().astimezone().isoformat(timespec='milliseconds')
                    record['Version'] = VERSION
                    record['CalibrationStatus'] = 'READY' if calibrator is not None else 'SKIPPED'
                    eye_status = record['EyeStatus']
                    state_changed = record['StudentState'] != last_state
                    if state_changed:
                        print(f'[STATE] {last_state} -> {record["StudentState"]}  '
                              f'eye={eye_status}  closed={record["ClosedSeconds"]:.2f}s  '
                              f'open={record["OpenSeconds"]:.2f}s', flush=True)
                        last_state = record['StudentState']
                    # Always persist the transition even if it happens between 0.5s samples.
                    force = state_changed or last_log is None or t - last_log >= 0.5
                    if force:
                        message = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                        logfile.write(message + '\n')
                        logfile.flush()
                        print(message, flush=True)
                        if sender:
                            sender.send(record)
                        if port:
                            port.write((message + '\n').encode('ascii'))
                        last_log = t
                    if not args.no_display:
                        if points is not None:
                            color = ((0, 210, 0) if eye_status == 'OPEN' else
                                     (0, 0, 255) if eye_status == 'CLOSED' else (0, 210, 255))
                            paint_eyes(cv2, frame, points, color)
                        overlay_line(cv2, frame,
                                     f'State: {record["StudentState"]} / Eye: {eye_status}', 0,
                                     (0, 255, 0) if record['StudentState'] == 'Study' else (0, 210, 255))
                        overlay_line(cv2, frame,
                                     f'EAR L {record["EAR_Left"]} / {record["EAR_Threshold_Left"]:.3f}  '
                                     f'R {record["EAR_Right"]} / {record["EAR_Threshold_Right"]:.3f}', 1)
                        overlay_line(cv2, frame,
                                     f'Closed {record["ClosedSeconds"]:.1f}s  Open {record["OpenSeconds"]:.1f}s', 2)
                        overlay_line(cv2, frame, f'Tracking: {track}', 3)
                        cv2.imshow('Study Monitor - Eye Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
    except KeyboardInterrupt:
        print('[STOP] Keyboard interrupt.', flush=True)
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if port:
            port.close()
        if sender:
            sender.close()
        print(f'[LOG SAVED] {log_path}', flush=True)
        if calibrator is not None and calibrator.ready:
            print(f'[PROFILE SAVED] {profile_path}', flush=True)


if __name__ == '__main__':
    main()
