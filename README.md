# ChatARangAI v2

## What this is
AI plank-fatigue coach with real-time **TENS cueing**. Tracks hip angle from video (Continuity Camera or seed MP4), detects fatigue drift, and **tells a human operator** when to fire **TENS** bursts on **quads (UP)** or **lumbar (DOWN)**. No hardware control—**we output cues + logs** for analysis.

## Why
Test the hypothesis: **interrupt TENS** (brief, targeted bursts triggered by posture error) **reduces fatigue slope** and **extends time-to-failure (TTF)** vs **control** and **static** stimulation.

## What it does
- Pose → hip angle → EMA + median → **drift = angle − baseline**
- Auto-calibration; auto-fail; AR arrows; TTS cues
- Conditions: **control / static / interrupt**
- Logs every frame + event; writes per-trial plots + aggregate reports

## Run
./run.sh  
./.venv/bin/python agent.py --device 1 --subject S01 --condition interrupt --calib_time 2  
# if device 1 isn’t the iPhone, use --device 2  
# seed video:  
# ./.venv/bin/python agent.py --seed seeds/seed.mp4 --subject SEED01 --condition control  
./.venv/bin/python aggregate.py

## Controls
c calibrate/start • e end (good job) • q quit • space confirm (latency)

## TENS mapping (operator only)
- **Channel 1 = quads → UP (green)** → TTS: “channel one up”
- **Channel 2 = lumbar → DOWN (red)** → TTS: “channel two down”
- Fixed envelope for cues: **PW 200 µs**, **~1.5 s hold**, **period ≥ 3 s**; intensity fixed per subject (log mA externally)

## Defaults (flags override)
calib_time 3 s • ema_alpha 0.10 • need_drift 6° • need_hold 1.2 s • hyst_reset 2° • period 3 s • static_period 12 s • safety_drift 12° • safety_hold 2.0 s • pose_miss 12 s

## Conditions
- **control:** no cues  
- **static:** periodic alternating cues (quads/lumbar)  
- **interrupt:** cues only when drift holds beyond thresholds

## Outputs
- Per trial CSV: `logs/session_<subject>_<condition>_<id>.csv`
- Per trial plots: `logs/plots/session_*_angle.png`, `session_*_drift.png`
- Summaries: `logs/reports/trial_summaries.csv`
- Aggregates: `logs/reports/aggregate_summary.csv`,  
  `logs/plots/aggregate_slope_scatter.png`, `aggregate_ttf_hist.png`

## CSV columns (core)
t_abs, t_rel, t_vid, event, hip_angle, avg_angle, slope_5s, drift, pose_valid, channel, cmd_id, t_cmd_vid, t_confirm_vid, latency_s, expected_dir, ttf_s

## Auto-fail (trial end)
|drift| ≥ 12° for ≥ 2 s **or** pose lost ≥ 12 s → stop + “good job”

## Troubleshooting
- Camera permission: `tccutil reset Camera` then allow the venv Python in System Settings → Privacy & Security → Camera
- Find device index:  
  `python3 - <<'PY'\nimport cv2\nfor i in range(6):\n cap=cv2.VideoCapture(i); print(i, cap.isOpened()); cap.release()\nPY`
- No plots: ensure CSV has `calibrated` & `start` (press **c** to force)

## Safety
Operator only. Stop on discomfort, dizziness, HR issues, skin problems. Contraindications: pacemaker/ICD, epilepsy, pregnancy, broken skin.
