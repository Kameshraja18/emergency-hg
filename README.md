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

## GitHub hosting

This project is ready to live as a GitHub source repository. A GitHub Actions workflow will install the Python dependencies and syntax-check the main scripts on every push and pull request.

Because this app depends on a local webcam, it is not a GitHub Pages app. The intended deployment model is:

1. Host the source code on GitHub.
2. Run the app on a local machine, laptop, or edge device with a camera.
3. Optionally use GitHub Actions for validation and release automation.

To publish your local changes to the existing GitHub remote:

```powershell
git add README.md .github\workflows\ci.yml main.py evaluate_labeled_set.py requirements.txt PROJECT_REPORT.md
git commit -m "Prepare GitHub hosting and CI"
git push -u origin HEAD
```

## Notes
- If audio fails on Windows, it may be due to device access; try running as admin or checking sound settings.
- On first run, `gTTS` needs internet access to synthesize audio.

## Labeled evaluation

To run the evaluation script, provide a CSV with at least these columns:

- `true_label`: the ground-truth gesture label
- `predicted_label`: the model or system output label
- `latency_ms`: optional end-to-end alert latency in milliseconds

Example:

```powershell
python evaluate_labeled_set.py labeled_results.csv
```
