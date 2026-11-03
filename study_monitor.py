"""Raspberry Pi 5 study/drowsiness monitor. MediaPipe Tasks API.
Usage: python study_monitor.py --source video.mp4 [--serial /dev/ttyACM0]
       python study_monitor.py --source 0
       python study_monitor.py --simulate
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time

from mqtt_sender import MQTTSender

EYE_LEFT = (33, 160, 158, 133, 153, 144)
EYE_RIGHT = (362, 385, 387, 263, 373, 380)

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def ear(points, ids):
    p = [points[i] for i in ids]
    width = distance(p[0], p[3])
    return (distance(p[1], p[5]) + distance(p[2], p[4])) / (2 * width) if width > 1e-8 else 0

class StudyMonitor:
    def __init__(self, threshold=0.22):
        self.threshold = threshold
        self.state = 'Empty'
        self.blinks = 0
        self.closed_since = None
        self.open_since = None
        self.doze_since = None
        self.last_face = None
        self.history = deque()
        self.prev_t = None
        self.prev_closed = None
        self.perclos = 0.0

    def update(self, t, face_present, eye_ratio=None):
        if face_present:
            self.last_face = t
        if not face_present:
            if self.last_face is None or t - self.last_face >= 1.0:
                self.state = 'Empty'
                self.closed_since = None
                self.open_since = None
                self.doze_since = None
                self.history.clear()
                self.perclos = 0.0
                self.prev_t = None
                self.prev_closed = None
            return self.snapshot(t, eye_ratio)

        closed = eye_ratio is not None and eye_ratio < self.threshold
        if self.state == 'Empty':
            self.state = 'Study'
            self.history.clear()
            self.prev_t = None
            self.prev_closed = None

        # Each interval is labelled with the previous eye state.
        if self.prev_t is not None and t > self.prev_t:
            dt = min(t - self.prev_t, 0.25)  # avoid treating dropped frames as eye closure
            self.history.append((t, dt, bool(self.prev_closed)))
        self.prev_t, self.prev_closed = t, closed
        while self.history and self.history[0][0] < t - 60.0:
            self.history.popleft()
        total = sum(d for _, d, _ in self.history)
        self.perclos = sum(d for _, d, c in self.history if c) / total if total else 0.0

        if closed:
            if self.closed_since is None:
                self.closed_since = t
            self.open_since = None
        else:
            if self.closed_since is not None:
                duration = t - self.closed_since
                if 0.08 <= duration <= 0.4:
                    self.blinks += 1
                self.closed_since = None
            if self.open_since is None:
                self.open_since = t

        closed_duration = (t - self.closed_since) if self.closed_since is not None else 0.0
        drowsy = closed_duration >= 0.5 or (total >= 10.0 and self.perclos > 0.20)
        if self.state == 'Study' and drowsy:
            self.state = 'Dozzy'
            self.doze_since = t
        elif self.state == 'Dozzy' and self.open_since is not None and t - self.open_since >= 2.0:
            # Resume only after sustained eye opening.
            self.state = 'Study'
            self.doze_since = None
        return self.snapshot(t, eye_ratio)

    def snapshot(self, t, ratio):
        return dict(StudentState=self.state,
                    DozeTime=round(max(0, t - self.doze_since), 1) if self.doze_since is not None else 0.0,
                    BlinkCount=self.blinks,
                    EAR=round(ratio, 3) if ratio is not None else None,
                    PERCLOS=round(self.perclos, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='camera index or video path')
    parser.add_argument('--model', default='face_landmarker.task')
    parser.add_argument('--serial', default=None, help='optional serial device path')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--threshold', type=float, default=0.22)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--mqtt-host', default=None, help='UNO Q MQTT broker IP (omit to disable MQTT)')
    parser.add_argument('--mqtt-port', type=int, default=1883)
    parser.add_argument('--mqtt-topic', default='student/monitor')
    args = parser.parse_args()
    monitor = StudyMonitor(args.threshold)
    sender = None
    if args.mqtt_host:
        sender = MQTTSender(host=args.mqtt_host,
                            port=args.mqtt_port,
                            topic=args.mqtt_topic)
        sender.connect()
    port = None
    if args.serial:
        import serial
        port = serial.Serial(args.serial, args.baud, timeout=0.1)
        time.sleep(2)
    last_sent = -1.0

    def publish(t, record):
        nonlocal last_sent
        if t - last_sent >= 0.5 or last_sent < 0:
            msg = json.dumps(record, separators=(',', ':'))
            print(msg, flush=True)
            if sender:
                sender.send(record)
            if port:
                port.write((msg + '\n').encode('ascii'))
            last_sent = t

    if args.simulate:
        # No camera or model needed: exercises state machine and serial messages.
        for i in range(160):
            t = i * 0.1
            if t < 2: present, ratio = False, None
            elif t < 6: present, ratio = True, 0.30
            elif t < 7.2: present, ratio = True, 0.10
            elif t < 12: present, ratio = True, 0.30
            else: present, ratio = False, None
            record = monitor.update(t, present, ratio)
            publish(t, record)
            if sender:
                time.sleep(0.1)  # pace MQTT test messages
        if sender:
            sender.close()
        if port:
            port.close()
        return

    import cv2
    import mediapipe as mp
    if not Path(args.model).is_file():
        raise FileNotFoundError(f'Missing model: {args.model}')
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {args.source}')
    is_file = not isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
    fps = fps if 1 <= fps <= 240 else 30
    frame_index = 0
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.model),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_index / fps if is_file else time.monotonic()
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            found = bool(result.face_landmarks)
            ratio = None
            if found:
                pts = result.face_landmarks[0]
                ratio = (ear(pts, EYE_LEFT) + ear(pts, EYE_RIGHT)) / 2
                h, w = frame.shape[:2]
                for idx in EYE_LEFT + EYE_RIGHT:
                    pt = pts[idx]
                    cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 2, (0,255,0), -1)
            record = monitor.update(t, found, ratio)
            publish(t, record)
            cv2.putText(frame, f"{record['StudentState']}  Doze {record['DozeTime']:.1f}s  Blinks {record['BlinkCount']}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if not args.no_display:
                cv2.imshow('Study Monitor', frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if port:
        port.close()
    if sender:
        sender.close()

if __name__ == '__main__':
    main()
