# Hand Gesture Assistive Tool

This app detects hand gestures (raised fingers) via webcam using MediaPipe Hands, speaks a Tamil message via gTTS, and can optionally send an SMS via Twilio.

## Features
- CPU-friendly MediaPipe-based hand pose detection
- Stabilized gesture detection with adjustable buffer (3–15)
- Handedness-aware thumb detection
- Voice playback locally when available; Streamlit demo mode can play browser audio
- SMS sending (toggle `s` in desktop mode)
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

## Streamlit deployment

Streamlit Cloud cannot use the local desktop webcam loop directly. The app now supports a deployable demo mode:

1. Open the repository on Streamlit Cloud.
2. Set the main file to `main.py`.
3. Use the camera snapshot or image upload UI.
4. Keep `runtime.txt` in the repo so Streamlit Cloud uses Python 3.11 instead of 3.14.

If you want the original full live-webcam experience, run the desktop mode locally with `python main.py`.

If you run the local desktop mode and want the OpenCV window behavior, install `opencv-python` in your local environment. The repository defaults to `opencv-python-headless` so Streamlit Cloud can import `cv2` reliably.

## GitHub hosting

This project is ready to live as a GitHub source repository. A GitHub Actions workflow will install the Python dependencies and syntax-check the main scripts on every push and pull request.

Because the desktop mode depends on a local webcam, it is not a GitHub Pages app. The deployable Streamlit mode is snapshot-based for cloud use, while the full live-webcam experience remains local.

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
