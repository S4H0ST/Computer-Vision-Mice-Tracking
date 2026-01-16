# 🐭 Computer Vision Mice Tracking System
### Bachelor's Thesis (TFG) - Universidad Rey Juan Carlos (URJC)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-magenta.svg)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

**Automated analysis of rodent behavior for pharmacological studies using Deep Learning.**

---

## 📖 Project Overview
This project is developed as a **Bachelor's Thesis (Trabajo de Fin de Grado)** in Computer Engineering at **Universidad Rey Juan Carlos**.

The system automates the observation of **Open Field Tests**, a standard protocol in pharmacology to assess anxiety and locomotion in mice. By replacing manual observation with Computer Vision, this tool aims to:
* Reduce human error and bias.
* Provide objective metrics (speed, time in zones, specific behaviors).
* Accelerate the testing process for new drugs (e.g., chemotherapy side effects).

---

## ⚙️ Key Features
Based on the current codebase, the system includes:

* **Hybrid Detection Engine:** Combines **YOLOv8** object detection with **Heuristic Rule-Based Logic** (`behavior_rules.py`) to distinguish complex actions.
* **Behaviors Detected:**
    * `rat_walking` / `rat_immobility`
    * `rat_rearing` (standing up)
    * `rat_climbing` (climbing walls)
    * `rat_head_dipping` (exploring holes)
* **Interactive Zone Calibration:** A GUI tool to define the arena boundaries (Walls) and interest points (Holes) before analysis.
* **Data Toolkit:** Tools to extract frames from raw videos and build datasets automatically for training.
* **Automated Reporting:** Generates a CSV with frame-by-frame behavioral data and speed metrics.

---

## 🎥 Demo
![DemoRatgif](rat_detector_project\output\Gif\DemoGit_rat.gif)

---

## 🚀 Installation

### Prerequisites
* Python 3.9+
* CUDA-compatible GPU (Recommended for training)

### Setup
```bash
# Clone the repository
git clone [https://github.com/YourUsername/Computer-Vision-Mice-Tracking.git](https://github.com/YourUsername/Computer-Vision-Mice-Tracking.git)
cd Computer-Vision-Mice-Tracking

# Create virtual environment (Optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python numpy pydantic
