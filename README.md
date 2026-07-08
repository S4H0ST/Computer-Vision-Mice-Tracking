
# Computer Vision Mice Tracking System
### Bachelor's Thesis (TFG) - Universidad Rey Juan Carlos (URJC)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8_Pose-magenta.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

**Automated analysis of rodent behavior for pharmacological studies using Deep Learning (Pose Estimation and Recurrent Neural Networks).**

---

## Project Overview

This project automates the observation of the **Open Field Test**, a standard protocol in pharmacology used to assess anxiety and locomotion in mice (specifically white mice in a box with holes). 

By replacing manual observation with Computer Vision and Deep Learning, this tool allows researchers to:
* Eliminate human error and observational bias.
* Extract objective metrics based on the animal's biomechanics.
* Classify complex postures that require temporal analysis.

    ![data_demo](media_original/poses.png)

---

## Hybrid Architecture (Key Features)

The system no longer relies on simple bounding boxes. Instead, it uses a two-phase hybrid AI architecture:

1. **The Eyes (YOLOv8 Pose):** Identifies skeletal *Keypoints* (Snout, Spine, Tail Base) frame by frame to capture exact biomechanics.
2. **The Brain (RNN - LSTM):** Analyzes the temporal sequence of these *Keypoints* to understand continuous movement and classify the action.
3. **The Instinct (Spatial Logic):** Maps the physical environment (walls, holes) to provide spatial context (e.g., *Head Dipping*).

### Detected Behaviors

The system trains YOLO on **5 base labels** but outputs **7 final behaviors** to the researcher. See the [Posture Hierarchy](#posture-hierarchy-yolo-labels-vs-final-output-behaviors) section below for the full breakdown.

| Final Behavior | Origin |
|---|---|
| Climbing | Direct YOLO label |
| Grooming | Direct YOLO label |
| Head Dipping | YOLO label + spatial zone confirmation |
| Rearing | Direct YOLO label |
| Walking | Derived from `rat_horizontal` via RNN / displacement |
| Immobile | Derived from `rat_horizontal` via RNN / displacement |
| Sniffing | Derived from `rat_horizontal` via spatial logic (nose near wall) |

---

## Posture Hierarchy: YOLO Labels vs. Final Output Behaviors

### The 5 YOLO Training Classes and 7 Final Behaviors

YOLO is trained on **5 classes** — each representing a visually distinct body posture that a CNN can learn from a single frame. These are then processed by a second logic layer to produce **7 final output behaviors**:

| # | YOLO Training Label | Final Behavior | Detection Layer |
|---|---|---|---|
| 1 | `rat_climbing` | **Climbing** | YOLO (direct) |
| 2 | `rat_grooming` | **Grooming** | YOLO (direct) |
| 3 | `rat_head_dipping` | **Head Dipping** | YOLO + spatial zone (hole map) |
| 4 | `rat_rearing` | **Rearing** | YOLO (direct) |
| 5 | `rat_horizontal` | **Walking** | YOLO + RNN / centroid displacement |
| 5 | `rat_horizontal` | **Immobile** | YOLO + RNN / centroid displacement |
| 5 | `rat_horizontal` | **Sniffing** | YOLO + spatial logic (nose keypoint near wall) |

### Why Does `rat_horizontal` Become 3 Different Behaviors?

This is a core architectural decision of the project. YOLO analyzes each frame **independently** and can only capture **static posture** — the shape of the animal's silhouette at a single instant. When a rat is walking, standing still, or sniffing the wall edge, its body maintains the same horizontal profile in every individual frame. There is no pixel-level difference that a CNN can exploit.

Differentiating these three states requires **information that a single frame cannot provide**:

- **Walking vs. Immobile** → requires **temporal context**: the RNN observes the sequence of keypoint positions across multiple frames and detects whether the centroid and spine keypoints are displacing over time.
- **Sniffing** → requires **spatial context**: the nose keypoint (snout) is checked against the arena wall boundaries. If the rat is in a horizontal posture *and* the snout keypoint is within a threshold distance of the wall, the behavior is reclassified as sniffing.

Training YOLO on fewer, visually clean classes (5 instead of 7+) is deliberate: it reduces label ambiguity during training, avoids the model learning indistinguishable visual patterns, and delegates temporal and spatial disambiguation to the appropriate downstream layers. This is the same reason `head_dipping` also relies on spatial zone mapping to confirm the snout is above a labeled hole, rather than trusting posture alone.

> **TFG note:** This multi-layer derivation is a fundamental argument in the system's design justification — it demonstrates why a pure CNN approach (Phase 2) was insufficient and why the hybrid architecture (YOLO Pose + RNN + Spatial Logic) was necessary.

---

## Project Structure

```text
Computer-Vision-Mice-Tracking/
├── `data.yaml`
├── `main_model.py`
├── `extraccion_frames/`
│   ├── `dataset_builder.py`
│   ├── `frame_extractor.py`
│   └── `main_dataFrames_config.py`
├── `helpers/`
│   ├── `base.py`
│   ├── `configuracion.py`
│   └── `interfaces.py`
├── `modules/`
│   ├── `brain_rnn/`
│   │   ├── `dataset.py`
│   │   ├── `inference.py`
│   │   ├── `model.py`
│   │   └── `trainer.py`
│   ├── `core_yolo/`
│   │   ├── `calibrator.py`
│   │   ├── `detector.py`
│   │   └── `trainer.py`
│   └── `detector_agujeros/`
│       └── `agujeros.py`
└── `README.md`
```

---

## Installation and Requirements

### Prerequisites

* Python 3.9 or higher.
* CUDA-compatible GPU (Highly recommended for training and real-time inference).

### Installation Steps

1. **Clone the repository:**
```bash
git clone [https://github.com/S4H0ST/Computer-Vision-Mice-Tracking.git](https://github.com/S4H0ST/Computer-Vision-Mice-Tracking.git)
cd Computer-Vision-Mice-Tracking
```


2. **Create a virtual environment and install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install ultralytics opencv-python pandas pyyaml
```

---

## Workflow (Usage)

The system is designed to be highly modular. Run the main script to open the interactive menu:

```bash
cd scripts
python main_model.py
```

### Main Menu Options:

1. **[SETUP] Calibrate Zones:** Opens the first frame of the video. Click to define the outer wall, inner wall, and the center of the 4 holes (Creates `coords.json`).
2. **[EYES] Train YOLO Model:** Trains the visual model using the dataset labeled with articular points (Pose).
3. **[DATA] Generate CSV (YOLO Only):** Analyzes the video using only the visual model and saves the articular coordinates frame by frame into a `.csv` file.
4. **[BRAIN] Train Temporal Network (RNN):** Uses the previously generated `.csv` to teach the recurrent network how to interpret temporal movement patterns.
5. **[FINAL] Generate HYBRID FINAL VIDEO:** Runs the system in production by merging **YOLO Pose + Spatial Logic + RNN** to export the fully analyzed and corrected video.

---

## Project Evolution and Architecture Decision Record (ADR)

This project has gone through multiple research and development phases, iterating over different Computer Vision approaches to overcome physical limitations in detecting complex animal behavior.

### Phase 1: Base Tracking and Overfitting Control

* **Repository Status:** [17050d38](https://github.com/S4H0ST/Computer-Vision-Mice-Tracking/commit/17050d38f900f684169ed35e07098b993205c43e)
* **Objective:** Achieve 100% rat detection in the controlled environment.
* **Development:** Training began with a massive dataset. However, due to the high similarity between frames, the neural network suffered from severe *overfitting*.
* **Solution:** The dataset size was drastically reduced, and rigorous *Data Augmentation* was applied. Being a hyper-controlled environment, lighting modifications were discarded, applying exclusively geometric transformations (rotations, scaling, and cropping) to force the model to generalize the rodent's shape.

### Phase 2: Behavior Labeling and CNN Limitations

* **Repository Status:** [a3c817d7](https://github.com/S4H0ST/Computer-Vision-Mice-Tracking/commit/a3c817d7a1edca2046258149519fc6171b413144)
* **Objective:** Classify static and dynamic postures using **MakeSense** for bounding box labeling.
* **Physical Problem:** Convolutional Neural Networks (CNNs) like YOLO analyze frame by frame. For a CNN without temporal context, a rat *Walking* looks visually identical to an *Immobile* rat, since the outer Bounding Box enclosing them is exactly the same.

![data_demo](media_original/DemoGit_rat.gif)

### Phase 3: Mathematical Heuristics and Spatial Logic (Brute Force)

* **Repository Status:** [60defaae](https://github.com/S4H0ST/Computer-Vision-Mice-Tracking/commit/60defaae6c993f1306cf4d460bde992cbd8418ae)
* **Objective:** Differentiate movement from inactivity by measuring spatial pixel displacement.
* **Development:** Algorithmic logic was implemented by extracting the centroid $(cx, cy)$ of the Bounding Box in each frame. Displacement speed was calculated using the Euclidean distance between consecutive frames:

$$v = \frac{\sqrt{(cx_t - cx_{t-1})^2 + (cy_t - cy_{t-1})^2}}{\Delta t}$$

* **Result:** This allowed estimating the *Walking* state using speed thresholds and mapping the spatial location relative to walls/holes. Even so, it remained a fragile system against subtle posture changes.

### Phase 4: Temporal Integration (RNN) and Dual Tracking

* **Repository Status:** [b9b8b741](https://github.com/S4H0ST/Computer-Vision-Mice-Tracking/commit/b9b8b741c3bdede00d62269d13e417b5c398bcdd)
* **Objective:** Provide the system with "memory" to understand continuous actions over time.
* **Development:** A Recurrent Neural Network (RNN) was introduced to analyze YOLO's historical data, and a specific *Box Tracking* was added for the rat's head.
* **Phase Status:**
* **[SUCCESS]** Perfect detection of *Head Dipping* thanks to head tracking and spatial zones.
* **[WARNING]** *Data Issue:* *Climbing* failed due to a shortage of images in atypical vertical positions.
* **[LIMITATION]** *Architectural Limit:* The RNN still confused *Rearing*, *Grooming*, and *Walking* because the outer bounding box is "blind" to the articular micro-movements of the limbs.


### Phase 5 (Current): Architectural Leap to YOLO Pose (Pose Estimation)

* **Repository Status:** In Active Development (`main` branch)
* **Objective:** Overcome geometric ambiguity by moving from an "area" approach to an "articular biomechanics" approach.
* **Development:** Bounding boxes are replaced by **Skeletal Keypoints** (Snout, Spine Center, Tail Base).
* **Technological Base:** The **YOLOv8-Pose** model is used, which shares the real-time inference *backbone* but adapts its output *head* to predict matrices of articular point coordinates.
* **Key Advantage:** It allows the RNN to differentiate complex states by measuring the variation in height ($Y$) between the snout and the tail (solving the Walking vs Rearing conflict) or by detecting exclusive local vibrations in the snout (Grooming).

*[INSERT IMAGE: A screenshot of you labeling articular points in Roboflow/CVAT, or the final skeleton drawn during inference]*

### Phase 5b (Current): Dataset Quality Analysis — Motion Blur vs. Sharp Frames

* **Objective:** Quantify the performance gap caused by motion blur and provide hard evidence for the TFG memory.
* **Problem:** In Open Field Test recordings, the rat frequently moves fast between frames, producing **motion blur**. Blurry frames significantly reduce keypoint localization accuracy, since the snout and tail base keypoints become ill-defined. A single aggregated mAP50 score on the full validation set masks this effect.
* **Solution — Sharpness Classification:** Each validation frame is evaluated using the **Laplacian variance** metric:

$$\sigma^2_{\nabla} = \text{Var}\!\left(\nabla^2 I\right)$$

The Laplacian operator amplifies high-frequency edges; its variance collapses to near-zero in blurry images. A threshold of **100** (tunable) separates *sharp* frames from *blurry* frames. This is implemented in `scripts/dataset_tools/sharpness_splitter.py`.

* **Solution — Split Evaluation:** The validation set is split into two independent subsets (`valid_sharp/` and `valid_blurry/`), each with its own temporary `data.yaml`. The trained YOLO-Pose model is evaluated separately on both, producing two independent mAP50 scores (`scripts/evaluation/evaluate_sharpness_gap.py`):

```
=== EVALUACIÓN: SOLO IMÁGENES NÍTIDAS ===
mAP50 (nítidas): 0.8921

=== EVALUACIÓN: SOLO IMÁGENES BORROSAS ===
mAP50 (borrosas): 0.7134

[GAP] Diferencia de rendimiento: 0.1787
[i] El gap es notable — el motion blur está afectando la precisión.
```

* **Why this appears in training debug statistics:** When the gap exceeds **0.05 mAP50**, the system flags it as significant. This means the training set is underrepresenting blurry-but-valid frames, and the model has not learned to handle them. The corrective action is a targeted labeling round where blurry frames are deliberately included in the training split, forcing the model to generalize across image quality levels.
* **TFG justification:** Instead of the qualitative claim *"motion blur degrades performance"*, the gap metric provides the quantitative argument: *"the model loses X mAP50 points on blurry frames, which represents Y% of all captured frames in the test video."* This is the kind of objective evidence expected in an engineering thesis.

### Phase 5c (Current): Offline Data Augmentation Strategy

* **Objective:** Artificially expand a small labeled dataset (248 unique images) to reach a training size sufficient for robust pose estimation, while correcting the class imbalance between majority and minority behaviors.

#### Why Data Augmentation Is Necessary

A dataset of 248 unique frames is insufficient to train a 5-class pose estimator to production quality. Without augmentation, the model would memorize the specific texture, lighting, and framing of those exact frames — a textbook case of **overfitting**: the training loss converges but the validation mAP50 plateaus or degrades. Augmentation forces the model to learn the invariant features of each posture (the rat's silhouette, limb angles, keypoint geometry) rather than superficial image properties.

#### Why ×3 (and Not ×2 or ×5)

The multiplier represents a trade-off between two failure modes:

| Multiplier | Risk |
|---|---|
| ×2 | Training set too small (~496 images). Insufficient diversity, model still prone to overfitting. |
| **×3 (chosen)** | **~744 images. Diversity gain is real; augmented variants still differ meaningfully from originals.** |
| ×5+ | Most training examples are derivatives of the same 248 images. Model overfits to augmented artifacts (e.g., overly bright frames that do not exist in real inference). |

This sweet spot is consistent with empirical findings in the YOLOv8 documentation for small-dataset fine-tuning scenarios.

#### Asymmetric ×4 for Minority Classes

The raw class distribution contains a 2:1 imbalance between the most frequent class (`head_dipping`, 69 images) and the least frequent (`rearing`, 34 images). Left uncorrected, this causes the model to systematically under-predict minority behaviors. One additional augmentation round is applied exclusively to `grooming` and `rearing`, reducing the effective ratio to approximately 1.5:1 — within the acceptable range for detection tasks.

#### Augmentation Transforms and Their Biological Justification

Only transforms that reflect real variation in the experimental setup are applied. Physically implausible augmentations (e.g., extreme zoom, heavy blur on already-blurry footage) are deliberately excluded.

| Transform | Applied to | Justification |
|---|---|---|
| **Horizontal flip** | All classes | Rats traverse the arena in both directions. The behavior is identical regardless of which way the animal faces. Keypoint coordinates are adjusted: `x_new = 1.0 − x_old`. |
| **Brightness +30 %** | All classes | Ambient lighting conditions vary between experimental sessions and recording environments. The model must be robust to moderate overexposure. |
| **Brightness −25 %** | `grooming`, `rearing` only | Minority behaviors are also observed in underlit frames (e.g., rat near a corner or partially occluded). The extra dark variant increases difficulty specifically for these classes, forcing the model to learn their keypoint structure rather than relying on global brightness cues. |

#### Implementation

Augmentation is applied offline **after** the Roboflow export is placed in `datasets/train/` and **before** training. This keeps the augmented data on disk, making the training process transparent and reproducible.

```bash
cd scripts
python dataset_tools/augment_dataset.py
```

The script is **idempotent**: re-running it detects the `_aug1/_aug2/_aug3` suffixes and does not create duplicates.

#### Target Metrics and Justification

The augmentation is designed to push the model from its current baseline to the following targets:

| Metric | Baseline (313 imgs, no aug) | Target (248 labeled + aug) | Rationale |
|---|---|---|---|
| **mAP50-pose** | 0.51 | **≥ 0.70** | Minimum defensible threshold for an engineering thesis. Published rodent trackers (SLEAP, DeepLabCut) report 0.85–0.92 on high-resolution infrared footage; 0.70 is realistic with a consumer camera at this resolution. |
| **mAP50-cls** | 0.70 | **≥ 0.80** | Classification of the 5 base labels. Already at 0.70; augmentation should push it above 0.80 by reducing overfitting on majority classes. |
| **Per-class mAP50** | Varies | **≥ 0.60 all classes** | No single behavior should be systematically missed. Values below 0.60 on a class indicate the model has effectively not learned it. |
| **mAP50-95** | 0.44 | **≥ 0.45** | Strict IoU metric (50 %–95 %). Values of 0.45–0.60 represent success in pose estimation tasks. |

> **Overfitting check during training:** If `val/pose_loss` starts rising while `train/pose_loss` continues to fall (typically after epoch 35–40 with this dataset size), training should be stopped early. The 50-epoch default in `configuracion.py` includes a 10-epoch safety margin for this monitoring.
