"""Raspberry Pi MQTT publisher. Independent of MediaPipe and OpenCV.

Requires: pip install paho-mqtt>=2.1,<3
"""
import json
import os
import threading


class MQTTSender:
    def __init__(self, host, port=1883, topic='student/monitor'):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                'MQTT library missing. Install: python3 -m pip install "paho-mqtt>=2.1,<3"'
            ) from exc

        self._mqtt = mqtt
        self.host = host
        self.port = port
        self.topic = topic
        self._connected = threading.Event()
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f'pi-study-{os.getpid()}',
        )
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._started = False

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f'[MQTT] Broker refused connection: {reason_code}', flush=True)
        else:
            self._connected.set()
            print(f'[MQTT] Connected: {self.host}:{self.port} (topic: {self.topic})', flush=True)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected.clear()
        if reason_code != 0:
            print(f'[MQTT] Disconnected; reconnecting: {reason_code}', flush=True)

    def connect(self, timeout=5):
        try:
            self.client.connect(self.host, self.port, keepalive=30)
        except OSError as exc:
            raise ConnectionError(
                f'MQTT Broker unreachable: {self.host}:{self.port} ({exc})'
            ) from exc
        self.client.loop_start()
        self._started = True
        if not self._connected.wait(timeout):
            self.close()
            raise ConnectionError(
                f'MQTT connection not confirmed within {timeout}s; '
                'check UNO Q IP, broker, and Wi-Fi.'
            )

    def send(self, record):
        """Send one state dict as JSON. Return True if queued for transmission."""
        if not self._connected.is_set():
            return False
        payload = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
        info = self.client.publish(self.topic, payload, qos=1, retain=False)
        if info.rc != self._mqtt.MQTT_ERR_SUCCESS:
            print(f'[MQTT] Publish failed (code={info.rc})', flush=True)
            return False
        return True

    def close(self):
        if self._started:
            self.client.disconnect()
            self.client.loop_stop()
            self._started = False
        self._connected.clear()
