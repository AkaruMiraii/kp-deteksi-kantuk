# Landmark diagnostics

Run from the project root, inside the laptop's interactive PowerShell:

```powershell
python -m scripts.landmark_diagnostics --record
```

Close all other webcam applications first. Without `--record` or `--record-video`, no logs or images are saved. `--record` saves logs/snapshots only. `--record-video` explicitly opts into local silent raw video and also enables logs. No audio or network upload is performed.

## Raw video pilot

```powershell
python -m scripts.landmark_diagnostics --record-video --eye-threshold 0.12 --eye-duration 1.5 --eye-max-gap 0.5 --mouth-threshold 0.5 --mouth-duration 2.0 --mouth-max-gap 0.5
```

Start with your own 60-second pilot; do not include another participant without their consent. This saves `raw.avi` (FFV1 lossless BGR, no overlays) and `video_frames.csv` (frame_id, original timestamp_ms, decoded-pixel hash) in the session directory. Storage can be substantial; this pilot has no automatic duration/size cap, so end with Q after about one minute. No camera frames that the program did not read are claimed as recorded.

On exit the video is decoded in full and every pixel hash is checked before `raw_video.complete` becomes true. This takes additional time; wait for `Saved:` before closing the terminal. Errors propagate and are recorded in session metadata; they are not silently accepted. Forced process termination can leave incomplete evidence.

The AVI container has nominal 30 FPS, but actual capture intervals vary. For event timing and playback use the mapping timestamps, NOT the AVI player's duration. Frame 0 in the video corresponds to frame 0 in the log. Decode, hash, frame count and timestamps must all match.

To verify a completed session (replace SESSION_FOLDER with its actual path):

```powershell
python -m scripts.replay_session "SESSION_FOLDER" --verify-only
```

To watch the raw video without predictions, omit `--verify-only`; Q exits. To rerun inference headlessly with original capture times, use `--analyze`. Analysis creates a new directory under `outputs/diagnostics/replays/`, preserves source files, uses source thresholds and current decision policy, and records the source session. It does not promise equal landmark estimates across new detector runs or model changes. A model hash mismatch refuses analysis. GUI playback is sequential raw viewing, not yet an annotation editor.

Raw source is suitable for a later independent annotation workflow. Older snapshots/logs cannot be upgraded into real videos of unseen frames.

## Pilot interval annotation

```powershell
python -m scripts.annotate_session ".\outputs\diagnostics\20261007_132051_4c5c587e"
```

This reads and verifies the existing raw video first; no webcam is opened. Source video, prediction log and mapping are never rewritten. UI shows raw pixels plus a separate control panel, without predictions. Slider seeks to a frame; A/D step backward/forward one frame; J/L move 30 frames; Space toggles play/pause using source timestamps. Lowercase key input is expected (do not hold Shift).

For a single eyes-closed interval: find the first frame where both eyelids are visibly closed, B marks start. Find the last visibly closed frame before reopening, E marks end. Press 1 to commit `eyes_closed_both`, then S to save, Q to exit. Bounds are inclusive. Do not mark the first reopened frame as part of the closure. X cancels pending marks; U undoes the last draft interval. Same-channel overlaps are refused; eye and mouth labels may overlap in time. Unclear/hidden eyes use 9/7, not the closed label.

Labels: 1 eyes_closed_both; 2 eyes_open; 3 mouth_relaxed; 4 mouth_deliberate_opening; 5 mouth_speaking; 6 mouth_yawn_like; 7 eyes_not_visible; 8 mouth_not_visible; 9 eyes_uncertain; 0 mouth_uncertain. These are manual visual observations, not computed drowsiness targets. Consult the evaluation rubric for ambiguous mouth behavior.

S creates a new CSV/JSON revision in the session's `annotations/` folder. Q/window close saves changed committed intervals. Pending start/end selections are not labels and are not committed by Q. Files include source hashes and annotator identity. Each launch starts a new empty draft; reopening does not resume earlier labels yet, but old revisions are preserved. For this first exercise mark only one interval in one launch. Unmarked frames remain unknown, never automatic negatives; `annotation_complete` is false until a later completeness/review workflow. This is not a finished labeled dataset or a metrics result.

## Opt-in eye duration smoke test

```powershell
python -m scripts.landmark_diagnostics --record --eye-threshold 0.12 --eye-duration 1.5 --eye-max-gap 0.5
```

All three parameters are required together and must be finite and positive. Without them, the display says `Eye duration: OFF`; eye state and duration are unavailable in logs. These example values are development parameters, not calibrated defaults.

A single persistent `EyeClosureTracker` is updated each frame. Both EAR values must be below the threshold. Candidate duration uses frame timestamps in seconds; after the configured duration, the state is `prolonged_closure`. A missing/invalid observation or excessive interframe gap resets continuity. `no_bilateral_closure` does not certify that both eyes are open or that the person is not drowsy.

CSV/JSON now include `eye_state` and `eye_duration_seconds`; session metadata records the rule and `calibrated: false`. The predicted `Eye` status is separate from the manually selected `Activity`. This is an eye channel only, not a combined drowsiness classifier.

Test eyes open, brief blinks, closure for 2-3 seconds, reopening, face out and re-entry. For the reset test, close eyes briefly, move out of view, then return: the old closure time must not resume. Exit with Q.

## Opt-in mouth duration smoke test

```powershell
python -m scripts.landmark_diagnostics --record --eye-threshold 0.12 --eye-duration 1.5 --eye-max-gap 0.5 --mouth-threshold 0.5 --mouth-duration 2.0 --mouth-max-gap 0.5
```

Mouth parameters are independently opt-in and required together. The example MAR threshold 0.5 and duration 2.0 seconds are development probes, not calibrated values. They apply only to the documented four-point MAR formula.

`MouthOpeningTracker` persists across frames independently from the eye timer. MAR strictly above threshold starts `opening_candidate`; sustained opening reaches `prolonged_opening`. MAR at/below threshold yields `no_large_opening`. Invalid observations or excessive gaps reset the mouth timer and yield unavailable data as applicable. Eye and mouth outputs share the existing conservative geometry gate; either channel can remain non-triggered while the other triggers.

CSV/JSON include `mouth_state`, `mouth_duration_seconds`, `final_status`, and `drowsiness_trigger`.

## 3-State Drowsiness Decision (Decision Fusion)

Policy `v2_eye_evidence_mouth_observation` replaces the historical eye-OR-mouth rule. It is a provisional milestone, not completion of the full EAR/MAR project:
1. `tidak_dapat_dinilai`: face absent, invalid geometry, an active channel not assessable, or the eye decision channel disabled. No channels enabled also yields this state, never a negative finding. Disabled-channel decisions do not invent or reset an active valid mouth timer.
2. `indikasi_kantuk`: currently triggered only by `prolonged_closure`, with trigger `eye`. This is a visual indicator based on development parameters, not a medical inference.
3. `tidak_terdeteksi_indikasi_kantuk`: valid eye observation without prolonged closure under this provisional rule. It does not mean confirmed alertness or a completed yawn assessment.

`prolonged_opening` remains visible and logged, but cannot independently trigger drowsiness before yawn evidence is validated. CSV/JSON include `decision_reason` and `decision_policy`. Metadata flags mouth evidence and full acceptance as false. Unknown state strings raise errors. Historical logs keep their original policy and must not be relabeled in place.

See `docs/detection-spec.md` and `docs/evaluation-protocol.md` for the acceptance gap, annotation rubric and calibration gates.

Test relaxed mouth, a brief opening, opening for about 3 seconds, closing again, normal speaking, and leaving/re-entering view while a timer is active. Repeat the long-opening test twice if comfortable. Activity keys remain manual context; `speaking` does not override computed geometry.

## Controls

Focus the OpenCV window. Activity labels take effect on the next frame:

- `1`: neutral, eyes open and mouth relaxed.
- `2`: eyes closed.
- `3`: mouth open. This is not automatically a yawn label.
- `4`: speaking.
- `5`: face outside the frame; press before moving out.
- `S`: request an extra snapshot on the next frame.
- `Q` or close the window: exit and finalize the summary.

Hold each activity for about 5 seconds. For eyes closed, select `2`, close the eyes, then select `1` after reopening. Transition frames may disagree with the label; labels are user-supplied context, not independent ground truth.

## Output

Each run creates a unique directory under `outputs/diagnostics/`:

- `metrics.csv`: one row per processed frame, timestamps, activity, EAR, MAR, geometry status and processing time.
- `frames.jsonl`: the same metrics plus float pixel coordinates of the 16 selected eye/mouth landmarks and snapshot paths.
- `session.json`: package versions, model SHA256, index mappings, MAR formula, counts and exit reason.
- `snapshots/`: paired original and annotated PNG files. Default: one pair every 2 seconds, maximum 120 pairs total, including manual snapshots. Logging continues after the snapshot limit.

Missing faces or invalid selected geometry have unavailable metrics, not zero values. Checks cover finite coordinates, points within image bounds, sufficient indices and nonzero widths only. These checks do not certify landmark accuracy or drowsiness classification.

MAR is distance(13,14) / distance(78,308). EAR reuses `scripts/eye_features.py`. Distances use float pixel coordinates; rounding is for drawing only.

Processing duration uses `perf_counter_ns()`; the summary records its resolution. Webcam timestamps still use `monotonic_ns()` and remain strictly increasing. Older sessions measured with Windows Python 3.11 `monotonic_ns()` may contain 0/15/16 ms quantization; do not reinterpret those historical values as precise timings.

Processing time includes RGB conversion, inference, feature extraction and drawing. It excludes camera read, file writes and GUI display. Snapshot capture can slow the overall loop, so this mode is not the final FPS benchmark.

Images and logs contain private face data. They stay in the existing ignored `outputs/` directory; only transfer for the explicitly requested review. Use `Q` for normal shutdown. Forced process termination can leave no summary; the per-frame logs are flushed regularly.

## Automated verification

```powershell
python -m unittest discover -s tests -v
```

The test images created by unit tests are synthetic fixtures, not observations of a face. `--input-image PATH` provides a headless single-image integration check without opening the webcam. Do not mix its output with a real webcam session.
