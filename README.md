# Hand Gesture Assistive Tool

This app detects hand gestures (raised fingers) via webcam using MediaPipe Hands, speaks a Tamil message via gTTS/pygame, and can optionally send an SMS via Twilio.

## Features
- CPU-friendly MediaPipe-based hand pose detection
- Stabilized gesture detection with adjustable buffer (3–15)
- Handedness-aware thumb detection
- Voice playback (toggle `v`), SMS sending (toggle `s`)
- Rate limiting to avoid spamming (30s per gesture)
- On-screen overlays and keyboard controls

## Controls
- q: Quit
- s: Toggle SMS on/off
- v: Toggle Voice on/off
- r: Reset last-sent memory
- + / -: Increase / Decrease gesture buffer size
- h: Print help

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If pip is not on PATH, use your Python interpreter explicitly.

3. Configure Twilio (optional) by creating a `.env` file in this folder:

```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890
DEST_PHONE_NUMBER=+911234567890
```

## Run

```powershell
python main.py
```

Make sure a webcam is available. Press `q` to quit.

## Notes
- If audio fails on Windows, it may be due to device access; try running as admin or checking sound settings.
- On first run, `gTTS` needs internet access to synthesize audio.
