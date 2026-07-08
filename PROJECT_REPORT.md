

# Emergency Hand-Gesture Assistive System — Project Report (Detailed)

## ABSTRACT

This project presents a real-time, camera-based hand-gesture assistive system designed to help non-verbal or mobility-impaired users communicate needs using simple numeric hand gestures. The system leverages MediaPipe Hands for robust landmark detection, OpenCV for camera capture and overlay, and gTTS with pygame for Tamil text-to-speech output. Gestures are stabilized using a configurable circular buffer, majority voting and median value checks, and triggers require dwell windows and cooldowns to prevent accidental activations. Optional Twilio SMS integration allows remote notifications to caregivers or emergency contacts. The design emphasizes low-cost deployment, local processing wherever possible, and configurability to adjust sensitivity and timing for diverse user needs.

This abstract provides a concise overview of the project's goals, primary techniques, and intended beneficiaries. The core idea is to translate simple numeric gestures (0–10 fingers) into meaningful actions (spoken messages, logs, and notifications) with a focus on reliability, minimal training/calibration, and practical deployment in low-resource environments.

The rest of this report details the working environment, system analysis, design, implementation, testing, and future directions to inform both technical reviewers and non-technical stakeholders about the system's capabilities and limitations.

## ACKNOWLEDGEMENT

This project relies heavily on open-source software and community-maintained libraries. We acknowledge and thank the developers and contributors of MediaPipe, OpenCV, pygame, and gTTS for their work, which enables the rapid prototyping of computer vision and audio systems. Thanks are also due to testers and early users who helped refine gesture-to-message mappings and provided practical feedback on usability and accessibility.

Special thanks to documentation authors and forum contributors whose troubleshooting and examples were used to integrate MediaPipe and pygame smoothly. Finally, we recognize that community feedback—especially from users who rely on assistive technologies—has been invaluable in guiding practical design choices such as dwell windows, cooldown periods, and message phrasing.

## LIST OF TABLES

Table 1: Hardware and Software Requirements

- Hardware: Webcam, CPU (dual-core+), optional microphone and speakers for audio playback.
- Software: Python 3.8+, MediaPipe, OpenCV, pygame, gTTS, Twilio SDK (optional), and optional environment tooling like `python-dotenv`.

Additional tables (to add if needed):
- Table 2: Configuration parameters and recommended ranges (buffer_size, cooldown_seconds, stable_ratio).
- Table 3: Gesture-to-message mapping (0–10) with example Tamil phrases and English translations.
- Table 4: Project evaluation metrics for emergency gesture recognition.

### TABLE 4: PROJECT EVALUATION METRICS

| Metric | Recommended Reporting Format | Notes |
| --- | --- | --- |
| Accuracy | `__%` validation/test accuracy | Report on the held-out test split, not training data. |
| Gesture classes | `__` classes | Example: help, SOS, stop, wave, distress, neutral. |
| Precision | `__%` | Important when minimizing false emergency alerts. |
| Recall | `__%` | Critical for emergency detection; higher is better. |
| F1 score | `__%` | Useful when class balance is uneven. |
| False positive rate | `__%` | Measures how often non-emergency gestures trigger alerts. |
| Inference / alert latency | `__ ms` or `__ s` | Time from gesture recognition to notification trigger. |
| Dataset size | `__ images` or `__ video samples` | Mention number of users if recorded custom data. |
| Deployment detail | `Webcam only`, `Webcam + Raspberry Pi`, or `Webcam + mobile` | State where the system was actually tested. |

Current project status from the repository:

- Logged alert events: 49 entries in `event_log.csv`.
- Gesture classes observed in the log: 9 classes (`1, 2, 3, 4, 5, 7, 8, 9, 10`).
- Deployment detail: webcam only, using the default camera index in `main.py`.
- Accuracy, precision, recall, F1, false positive rate, latency, and labeled dataset size: not measured in the repository yet.

## LIST OF FIGURES

Figure 1: High-level system architecture showing camera capture, MediaPipe processing, gesture logic, and action outputs (TTS, SMS, log).

Figure 2: Example runtime overlay showing detected finger counts, stability status, per-hand counts, and message preview.

Figure 3: Timing diagram illustrating buffer windows, dwell requirement, cooldown, and global minimum interval between actions.

## TITLE

Emergency Hand-Gesture Assistive System

## INTRODUCTION

The ability to communicate basic needs quickly and reliably is vital for people with limited speech or mobility. Modern computer vision and lightweight ML tools allow us to recognize simple, intuitive gestures using commodity hardware (webcams) without heavy model training or large datasets. This project implements an assistive system that maps counts of raised fingers to prewritten Tamil messages. Its aim is not to provide a full sign-language recognition system, but rather a compact signaling tool that is easy to use, easy to configure, and robust enough for everyday practical use.

The system prioritizes the following qualities:

- Reliability: reduce false positives with buffering, median checks, and dwell windows.
- Accessibility: Tamil-language messages and clear on-screen overlays.
- Local-first operation: prefer local TTS playback; use network services only when explicitly enabled (gTTS and Twilio).
- Configurability: runtime adjustments via `config.json` and `messages.json` for buffer sizes, stable ratios, dwell windows, and message content.

Context and motivation: caretakers or medical staff can receive immediate signals from a patient who cannot call out or reach a phone. A user can raise one to ten fingers to signify different needs (food, water, help, emergency), and the system will announce the request and optionally notify a remote contact.

### 1.1 PROJECT INTRODUCTION

The Emergency Hand-Gesture Assistive System uses MediaPipe's hand landmark detection to locate 21 hand points per detected hand in each video frame. From these landmarks, deterministic geometric checks (thumb tip vs IP for thumb, tip vs PIP for other fingers) infer whether each finger is raised. Per-hand counts are bounded [0,5], summed across both hands and capped at 10. To mitigate jitter, a fixed-size FIFO buffer holds recent per-frame totals; a candidate value is accepted only when a configured proportion of the buffer agrees and the median equals the candidate. A dwell counter ensures consecutive stable windows before triggering any action. Triggers then invoke audio playback (gTTS + pygame), optional beep generation, event logging to CSV, and optional SMS alerts via Twilio.

The following sections describe environment needs, system analysis, design choices, implementation details, testing plans, and how to extend the system for production or research use.

## WORKING ENVIRONMENT

This section explains what hardware and software are required to run the system reliably, and typical environmental considerations for best performance.

### 2.1 HARDWARE REQUIREMENT

Minimum hardware requirements are intentionally modest to allow deployment on low-cost systems:

- A webcam (USB or built-in) that captures at least 640×480 should suffice; higher-resolution cameras improve landmark precision but increase CPU usage.
- A computer with a dual-core CPU (e.g., modern Intel/AMD dual-core or efficient mobile CPUs) is recommended; real-time on 30 fps is feasible on such hardware with the MediaPipe CPU pipeline.
- Stereo or mono speakers (or headphones) for audio playback; a microphone is not required unless voice input is added later.
- Optionally, an external monitor or large-screen display improves visibility for caretakers.

Environmental recommendations:
- Stable lighting (avoid strong backlight and deep shadows) improves landmark detection reliability.
- Simple backgrounds reduce false landmark detections; cluttered scenes may cause occasional misses.

### 2.2 SOFTWARE REQUIREMENT

- Python 3.8 or later is supported. It is recommended to use a virtual environment (venv or conda) to manage dependencies.
- Key Python packages: `opencv-python`, `mediapipe`, `pygame`, `gTTS`, `twilio` (optional), `python-dotenv` (optional). Exact versions should be pinned in `requirements.txt` for reproducible deployments.
- On Windows, installing the MSVC build tools or using prebuilt wheels is recommended for binary packages.

### 2.3 SYSTEM SOFTWARE

The core software stack includes:
- MediaPipe Hands (hand landmark detection module)
- OpenCV (video capture, image processing, and overlay drawing)
- pygame (audio playback for TTS and beep generation)
- gTTS (Google Text-to-Speech) — note: gTTS makes network calls to Google services; offline alternatives can be used if needed.
- Twilio Python SDK (optional) for sending SMS when environment variables with credentials are provided.

System configuration:
- `config.json` controls runtime tunables like `buffer_size`, `cooldown_seconds`, `stable_ratio`, `global_min_interval`, `beep_enabled`, `logging_enabled`, and `dwell_windows`.
- `messages.json` provides human-editable mappings from numeric gestures `0–10` to Tamil phrases (UTF-8 encoded). Missing keys fall back to default messages embedded in the code.

## SYSTEM ANALYSIS

In this section we analyze feasibility, examine alternatives, and present the reasoning behind the chosen approach.

### 3.1 FEASIBILITY STUDY

Functional feasibility:
- Counting fingers using deterministic rules from MediaPipe landmarks is straightforward and reliable in many real-world scenarios.
- Real-time performance is feasible on commodity hardware using MediaPipe's optimized CPU implementation; hardware acceleration (GPU) is optional for higher frame rates.

Operational feasibility:
- End-users require minimal training: a short calibration or help screen can explain the mapping of counts to messages.
- System use cases are limited and discrete, making verification and validation easier than full sign-language recognition.

Economic feasibility:
- The solution uses a webcam and free/open-source libraries, requiring minimal capital expenditure. Optional Twilio SMS costs depend on usage and account setup.

### 3.2 EXISTING SYSTEM

We researched common gesture recognition systems and products. Many rely on deep-learning models trained for broad gesture vocabularies, custom datasets, or cloud services. While capable, these systems introduce complexity and cost and often require significant user-specific training for high accuracy. For the specific target use-case—simple numeric signaling—heavy ML models provide marginal benefit over deterministic landmark-based rules and add deployment friction.

### 3.3 DRAWBACKS OF EXISTING SYSTEMS

- Complexity: large models increase memory and compute requirements and complicate deployment in constrained environments.
- Data requirements: collection of annotated datasets and personalization for different skin tones, hand shapes, or camera positions is time-consuming.
- Privacy and cost: cloud-based recognition or TTS can introduce privacy concerns and ongoing costs.

### 3.4 PROPOSED SYSTEM

We propose a local-first, deterministic approach:
- Use MediaPipe for landmark extraction (robust and optimized), then apply geometric rules for finger detection.
- Stabilize results via a sliding buffer and statistical checks (majority and median agreement).
- Require consecutive stable windows for a dwell threshold before triggering actions.
- Apply per-gesture cooldowns and a global minimum interval to prevent repeated or accidental notifications.

This approach minimizes false positives while still being responsive. It also preserves user privacy by default and keeps the runtime self-contained unless the operator opts into network features (gTTS/Twilio).

### 3.5 BENEFITS OF PROPOSED SYSTEM

- Lightweight: no dataset or heavy training required; deterministic logic is transparent and explainable.
- Configurable: `config.json` allows easy tuning for different users and environments.
- Accessible: built-in Tamil messages and clear overlays support end-users and caretakers.
- Extensible: the modular logic allows adding more complex gestures or integrating offline TTS engines later.

### 3.6 SCOPE OF THE PROJECT

In-scope:
- Real-time detection and interpretation of numeric hand gestures (0–10 fingers).
- Triggering spoken messages, event logging, optional SMS, and simple visual overlays.
- Configurable sensitivity and timing via JSON config.

Out-of-scope (for this version):
- Full sign-language recognition or continuous gesture segmentation.
- Robust multi-user tracking and long-term user profiles.
- End-to-end remote monitoring dashboards (these may be added as extensions).

## SYSTEM DESIGN

This section documents the architecture and main components, including suggested diagrams and class responsibilities for a refactor.

### 4.1 USE CASE DIAGRAM

Primary actors and use cases:
- Actor: End user (patient) — performs hand gestures.
- Actor: Caretaker — observes overlays and receives SMS alerts if configured.
- System: Gesture recognition app — captures frames, detects hands, stabilizes counts, triggers actions.

Main use cases:
- Perform gesture: user performs fingers; system displays count and message preview.
- Trigger message: when stable, system speaks message and logs event.
- Notify remote: optionally send SMS to configured contacts.

### 4.2 DATA FLOW DIAGRAM

Detailed data flow steps:
1. Camera captures raw frames in BGR.
2. Frames are flipped for mirror view and converted to RGB for MediaPipe.
3. MediaPipe returns hand landmarks and optional handedness.
4. Gesture logic computes per-hand finger counts and sums them.
5. Counts are appended to a circular buffer; statistical checks decide candidate stability.
6. If stable across dwell windows, action manager composes message, plays TTS, generates beep, logs event, and optionally sends SMS.
7. UI overlay updates with count, stability, and message preview.

### 4.3 CLASS DESIGN

Although implemented in a single script, it is helpful to outline a modular class-based decomposition for maintainability:

- CameraManager: Initialize camera, handle frame grabbing and conversion, expose frames to the pipeline.
- HandDetector: Wrap MediaPipe Hands initialization and call, return landmarks and handedness.
- GestureLogic: Functions to count fingers per hand, maintain buffer, compute majority/median, check stability and dwell windows.
- ActionManager: TTS generation and playback, beep generation, Twilio SMS sending, and event logging.
- ConfigManager: Load and validate `config.json` and `messages.json` and expose runtime parameters.

Each class should have a small, testable public API. For example, GestureLogic could expose: `append_count(count)`, `get_candidate()`, `is_stable(candidate)`, and `reset()`.

## PROJECT DESCRIPTION

This section covers objectives, module descriptions, implementation specifics, and maintenance guidelines in more depth.

### 5.1 OBJECTIVE

The objective is to deliver an assistive application that: 1) detects numeric hand gestures robustly using a webcam; 2) maps gestures to human-friendly Tamil messages; 3) provides immediate audible feedback; and 4) optionally notifies caregivers via SMS. The tool should be usable by non-technical caretakers and adaptable to different environments through configuration.

### 5.2 MODULE DESCRIPTION

Detailed modules and responsibilities:
- Input module (CameraManager): acquires frames, handles flip and color conversion.
- Detection module (HandDetector): performs MediaPipe hand detection and returns landmarks.
- Processing module (GestureLogic): counts fingers, runs stabilization and dwell logic, handles buffer resizing.
- Output module (ActionManager): plays TTS, generates beep, logs to CSV, and interfaces with Twilio when enabled.
- UX module: overlays text on frames (status, per-hand counts, messages) and handles keyboard controls.

Configuration files:
- `messages.json`: mapping integers to strings in Tamil (UTF-8); fallback to defaults if missing.
- `config.json`: runtime parameters (buffer size, cooldown, stable_ratio, global_min_interval, dwell_windows).

### 5.3 IMPLEMENTATION

Implementation highlights and rationale:
- MediaPipe Hands: chosen for speed and reliable landmark extraction without requiring model training.
- Deterministic finger counting: thumb handled by horizontal/vertical checks depending on handedness; other fingers use tip vs pip vertical comparison.
- Stabilization: deque buffer (configurable length), majority vote, and median checks reduce noise in real-time streams.
- Dwell windows: require consecutive stable windows before firing actions to minimize accidental activations.
- Rate limiting: per-gesture cooldown and a global minimum interval prevent spammy repeated actions.
- Audio pipeline: gTTS synthesizes to MP3 which pygame plays; the file is removed after playback when possible.
- Logging: append-only CSV with timestamps, gesture, message, and flags for SMS and voice playback.

Implementation pitfalls and mitigations:
- File locking on Windows for TTS-generated MP3s: code attempts to remove files safely; consider using unique filenames or in-memory playback for robustness.
- gTTS network failure: when gTTS fails, the system logs the error and continues; an offline TTS option is recommended for production.

### 5.4 MAINTENANCE

Maintenance checklist:
- Pin and regularly update `requirements.txt` and periodically test on target hardware.
- Add unit tests for GestureLogic (counting and buffer-based stability logic) and for basic ActionManager flows (mocking gTTS and Twilio).
- Document environment setup and common troubleshooting steps (camera selection, lighting adjustments).
- Consider packaging into a simple installer or a portable Python executable for non-technical deployments.

## SYSTEM TESTING

Thorough testing is essential to ensure reliability. This section presents test plans and examples.

### 6.1 TESTING DEFINITION

Types of tests:
- Unit tests: pure functions and logic (finger counting, majority vote, median, stability checks).
- Integration tests: pipeline from camera frame -> MediaPipe -> gesture logic -> action manager. These can be partly automated by feeding recorded frames.
- Manual acceptance tests: run the application in real settings to verify overlay clarity, TTS quality, and SMS delivery.

Sample unit test cases to implement:
- Thumb detection correctness for left and right hands using mocked landmarks.
- Tip-vs-pip logic for index/middle/ring/pinky across multiple angles.
- Buffer stability: sequences that should and should not trigger stability at various `STABLE_RATIO` and `buffer_size` settings.

### 6.2 TESTING OBJECTIVE

Objectives for testing:
- Confirm accuracy of finger counting under normal lighting and typical camera angles.
- Validate that the buffer and dwell logic prevent false activations while still being responsive.
- Ensure that TTS playback completes and that SMS calls are attempted only when credentials are present.
- Verify logging entries for correctness and completeness.

Test execution suggestions:
- Record short video clips (multiple users, various lighting) and run the system against these clips to produce repeatable test reports.
- Use mocking frameworks (unittest.mock) to simulate Twilio and gTTS network failures and verify graceful degradation.

## CONCLUSION

### 7.1 SUMMARY

The Emergency Hand-Gesture Assistive System is a lightweight, configurable tool that transforms simple hand gestures into actionable alerts, with a strong emphasis on accessibility and practical deployment. By combining MediaPipe's robust landmark detection with deterministic logic and stabilizing heuristics, the system achieves reliable operation without heavy training data or cloud dependencies (unless explicitly enabled for TTS or SMS).

The design choices favor transparency, low cost, and ease of use—important factors for real-world assistive tools deployed in care homes, clinics, or private residences.

### 7.2 FUTURE ENHANCEMENTS

Planned extensions and research directions:
- Replace gTTS with an offline TTS engine (pyttsx3 or a dedicated Tamil voice model) to improve reliability and privacy.
- Add a calibration wizard that guides users to the optimal camera position and lighting and optionally adapts thumb logic to user hand posture.
- Refactor into modular packages with unit tests and add CI to ensure changes don't regress core detection logic.
- Add localization (multi-language support) and a simple web dashboard showing recent events and logs (with authentication).

## APPENDIX

### 8.1 SCREEN SHOTS
Include screenshots demonstrating:
- Live overlay with status line and message preview.
- Example of the flashed emergency border during an emergency gesture.
- Sample `config.json` and `messages.json` files used for testing.

### 8.2 CODING

Key code pointers:
- `main.py` contains the main loop, MediaPipe integration, gesture logic, and action triggers.
- `messages.json` (optional) — mapping numeric gestures to Tamil messages (UTF-8).
- `config.json` (optional) — runtime tuning parameters.

Addendum: recommended unit tests to add in `tests/`:
- `test_gesture_logic.py`: tests for `count_fingers`, `majority_vote`, `median_value`, and `is_stable`.
- `test_config_loading.py`: tests for `load_config` and boundary conditions for configuration values.

---

This detailed version expands each major section to provide context, design rationale, implementation detail, and a practical path to production-quality improvements. If you'd like, I can now:
- Split the report into separate markdown files (e.g., `design.md`, `testing.md`) for a documentation site.
- Create `config.json` and `messages.json` templates and add them to the repository.
- Implement unit tests for the gesture logic and run them.
