# Gate Detection Baseline

This is the current CV-based gate detector built for the Bebop gate task.

The point here was not to make something that looks nice only on a few screenshots but something that gives usable gate estimates on the actual MAVLab/Cyberzoo-style images and can later be connected to guidance. It is still a classical vision baseline so not the final answer, but it is a much more serious starting point than the first simple thresholding script.

## What it does

The detector is mainly built around the fact that in this dataset the gate is recognized most reliably through the two blue side posts, while the top part is often checkered / high-contrast rather than fully blue.

So the pipeline is roughly:

1. preprocess the frame
2. segment blue regions
3. merge likely vertical post components
4. score left-right post pairs as gate candidates
5. keep a weak tracker estimate between frames if needed
6. return control-relevant quantities, not just a box

The output is not only a yes/no detection. It also gives:

- gate center in pixels
- opening box
- normalized lateral error
- normalized vertical error
- yaw proxy
- optional range estimate
- optional pose hooks if camera calibration is available

## Main design choices

A few choices here are intentional:

- **Blue-first detection**  
  The blue side posts are the most stable and consistent cue in this dataset.

- **Ring mode disabled by default**  
  In practice, on these images, full-ring reasoning was less reliable than post-pair reasoning. The gate behaves more like a blue arch with a strong opening than a clean blue ring.

- **Tracker does not count as a valid detection**  
  The tracker is only there as a fallback estimate for continuity. It should not artificially improve offline metrics.

- **Edge-based top support**  
  The top of the gate is often not blue, so using edge density there works better than forcing everything through blue segmentation.

## Files

### `enhanced_gate_detector.py`
Main detector implementation.

Contains:
- `GateDetectorConfig`
- `GateDetection`
- `AlphaBetaGateTracker`
- `EnhancedGateDetector`

### `evaluate_gate_detector.py`
Offline evaluator for a folder of images.

It:
- runs the detector on a dataset
- saves overlays
- writes a CSV summary
- builds a montage for quick inspection

## How to run

Activate the environment first:

Then run the evaluator on the raw dataset. ```bash
python3 evaluate_gate_detector.py data/raw --output-dir gate_eval_output --min-confidence 0.30 --max-save 60

Run the evaluator on the raw dataset:

python3 evaluate_gate_detector.py data/raw --output-dir gate_eval_output --min-confidence 0.30 --max-save 60

This saves:

gate_eval_output/gate_detection_summary.csv
gate_eval_output/gate_detection_montage.png
gate_eval_output/overlays/...
What the detector is actually looking for

The detector is not just doing “find blue and draw a box”.

It checks for:

blue segmentation consistency
two plausible vertical post components
enough vertical overlap between them
a reasonable opening width and height
a mostly hollow opening region
some top support from image edges near the gate top
left/right symmetry that is at least believable

So the confidence score is still a geometric score, not a learned probability.

Current status

This version is a much better baseline than the original simple thresholding script, but it is still a CV baseline and it still has failure cases.

Typical failures are:

strong blur
clutter near the posts
gate partially off-frame
wrong blue structures treated as posts
very close or very skewed views


