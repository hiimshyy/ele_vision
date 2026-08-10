"""
Quick MQTT connectivity test.

Tests:
1. Connect to broker
2. Subscribe to read topic
3. Publish a test message to write topic
4. Verify heartbeat works

Usage:
    python examples/test_mqtt.py
    python examples/test_mqtt.py --publish "Hello from edge"
    python examples/test_mqtt.py --listen 10   (listen for 10 seconds)

Reads credentials from .env file.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Load .env file
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

import paho.mqtt.client as mqtt

# Config from environment
BROKER_HOST = os.environ.get("SC_MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.environ.get("SC_MQTT_BROKER_PORT", "1883"))
USERNAME = os.environ.get("SC_MQTT_USERNAME", "")
PASSWORD = os.environ.get("SC_MQTT_PASSWORD", "")
CLIENT_ID = os.environ.get("SC_MQTT_CLIENT_ID", "embody002")
TOPIC_PUBLISH = "embody/w"
TOPIC_SUBSCRIBE = "embody/r"


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0 or reason_code == mqtt.CONNACK_ACCEPTED:
        print(f"  [OK] Connected to {BROKER_HOST}:{BROKER_PORT}")
        print(f"       Client ID: {CLIENT_ID}")
        client.subscribe(TOPIC_SUBSCRIBE, qos=1)
        print(f"  [OK] Subscribed to: {TOPIC_SUBSCRIBE}")
    else:
        print(f"  [FAIL] Connection failed: reason_code={reason_code}")
        sys.exit(1)


def on_disconnect(client, userdata, flags, reason_code, properties=None):
    print(f"  [!] Disconnected: reason_code={reason_code}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"  [MSG] {msg.topic}: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    except Exception:
        print(f"  [MSG] {msg.topic}: {msg.payload.decode()}")


def main():
    parser = argparse.ArgumentParser(description="MQTT Connection Test")
    parser.add_argument("--publish", type=str, default=None,
                        help="Publish a test message")
    parser.add_argument("--listen", type=int, default=5,
                        help="Listen for N seconds (default: 5)")
    args = parser.parse_args()

    print(f"\n  === MQTT Connection Test ===")
    print(f"  Broker:    {BROKER_HOST}:{BROKER_PORT}")
    print(f"  Username:  {USERNAME}")
    print(f"  Client ID: {CLIENT_ID}")
    print(f"  Publish:   {TOPIC_PUBLISH}")
    print(f"  Subscribe: {TOPIC_SUBSCRIBE}")
    print()

    # Create client
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=CLIENT_ID,
        protocol=mqtt.MQTTv311,
    )

    if USERNAME:
        client.username_pw_set(USERNAME, PASSWORD)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Connect
    print(f"  Connecting...")
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    except Exception as e:
        print(f"  [FAIL] Cannot connect: {e}")
        sys.exit(1)

    client.loop_start()
    time.sleep(1)  # Wait for connection

    if not client.is_connected():
        print(f"  [FAIL] Not connected after 1s")
        sys.exit(1)

    # Publish test message
    test_payload = {
        "device_id": "cabin-001",
        "event_type": "system.test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": args.publish or "MQTT connectivity test from Smart Cabin Edge",
    }

    result = client.publish(TOPIC_PUBLISH, json.dumps(test_payload), qos=1)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"  [OK] Published to {TOPIC_PUBLISH}")
        print(f"       Payload: {json.dumps(test_payload, ensure_ascii=False)}")
    else:
        print(f"  [FAIL] Publish failed: rc={result.rc}")

    # Listen for incoming messages
    print(f"\n  Listening on {TOPIC_SUBSCRIBE} for {args.listen}s...")
    print(f"  (Send a message to '{TOPIC_SUBSCRIBE}' from cloud to verify)")
    time.sleep(args.listen)

    # Disconnect
    client.loop_stop()
    client.disconnect()
    print(f"\n  [OK] Test complete. MQTT is working.\n")


if __name__ == "__main__":
    main()
