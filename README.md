# VisionVoice AI

VisionVoice AI is a real-time mobility-awareness prototype for visually impaired people. A YOLO26 detector tracks objects; scene, risk, and alert agents turn that stream into prioritized spoken updates and a shared dashboard.

> **Safety boundary:** This is a prototype, not a certified navigation or collision-avoidance device. It cannot establish that a crossing is safe and must never replace a cane, guide dog, traffic signals, local orientation, or user judgment. Test in a controlled environment with a sighted assistant before any public-space use.

## What it does

- Uses a webcam via WebRTC, with image-upload mode for quick validation and a **Remote viewer** mode for sharing the processed feed from one broadcaster to other devices connected to the same app server.
- Loads `yolo26n.pt` by default (or a local/custom Ultralytics checkpoint) and uses ByteTrack-backed object IDs from `model.track()`.
- Tracks every object class returned by the loaded YOLO model at the same time. The default COCO-trained YOLO26 model covers 80 classes; a custom checkpoint can provide different classes.
- Reports each object's relative location (`far left`, `left`, `ahead`, `right`, or `far right`) and an approximate distance.
- Applies separate scene-understanding, risk-assessment, and alert-planning agents.
- Prioritizes collision hazards (bus, truck, car, motorcycle, bicycle) and potentially harmful tools (knife, scissors, baseball bat) before normal objects.
- Offers either priority-hazard speech or a combined report of every simultaneously detected object. Hazards are always ordered first. Changed scenes are announced immediately; unchanged critical hazards repeat at a configurable interval.
- When no immediate hazard is detected, announces: “No immediate hazards detected. Continue with caution.”
- Uses the browser's speech synthesis, so each connected phone or computer speaks through its own audio output. The dashboard shows objects, position, range, hazard status, and recent alerts.

## Run locally

Python 3.10–3.12 is recommended. The first model load downloads the official YOLO26 weights if they are not already cached, so it needs an internet connection.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit, grant camera access, and choose **Live broadcaster**. On Apple Silicon, select `mps` in the sidebar when the installed PyTorch build supports it; otherwise leave the device on **Auto**.

## Share the live feed on your Wi-Fi network

Start the app on the computer connected to the camera, listening on your local network:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Find that computer's local IP address (on macOS, try `ipconfig getifaddr en0`), then open `http://YOUR_LOCAL_IP:8501` on another phone, tablet, or computer connected to the same Wi-Fi. On the camera device choose **Live broadcaster**; on the other devices choose **Remote viewer** and enable **Speak alerts on this device**.

This built-in sharing is for a trusted local network and a single Streamlit server process. Do not expose it directly to the public internet: add authentication, HTTPS, a TURN server for WebRTC, and a privacy/security review first.

## Create a temporary global link

For a demonstration to trusted people outside your Wi-Fi network, install Cloudflare Tunnel and use the included launcher:

```bash
brew install cloudflared
python3 share.py
```

The launcher starts the Streamlit app and prints both a **Local link** and a temporary **Global link** (ending in `trycloudflare.com`). It also writes the global link into the app's **Share links** panel. Send the global link to a phone or computer anywhere with internet access, then choose **Remote viewer** and enable **Speak alerts on this device**.

The global URL stops working when `share.py` stops. It has no sign-in protection, so it is suitable only for a short, trusted demonstration. Before real-world use, deploy behind authentication and HTTPS, use a named tunnel with access controls, and protect user video and location data.

## Agent flow

```text
Camera → YOLO26 detection + tracking → Scene agent → Risk agent → Hazard-first alert planner → shared dashboard + browser voice
```

Distance is estimated from the camera's horizontal field of view, the bounding-box size, and a typical real-world dimension for the detected class. Configure the field of view in the sidebar for the camera you are using. It is a rough monocular estimate—not a measurement of distance, speed, or time-to-collision—and will be less reliable for unusual object sizes, partial objects, and occlusions. Unknown custom classes are still detected and located, but show **range unavailable** until a size prior is added to `OBJECT_SIZE_PRIORS` in `visionvoice/core.py`.

The risk score is an intentionally transparent heuristic based on object type, screen position, estimated range (or box scale when range is unavailable), confidence, and—only when a stable track grows across several frames—apparent approach.

## Verification

```bash
pytest -q
```

The unit tests cover risk prioritization, conservative approach detection, and alert de-duplication. For field validation, add measured-distance sensors, traffic-signal state recognition, route localization, a privacy review, accessibility testing with visually impaired participants, and staged outdoor safety testing.
