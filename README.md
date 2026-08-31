# Dual-UAV Oil Pipeline Inspection & Defect Detection System

An automated, dual-UAV inspection framework developed in collaboration with **Quanser** for pipeline surveillance, real-time hazard identification, and coordinated re-inspection of defective duct segments using DJI Tello micro-drones.

---

## Overview

Manual inspection of industrial pipeline infrastructure is high-risk, expensive, and time-consuming. This project introduces an autonomous, multi-drone inspection system designed for scale-model oil duct networks. 

Using low-cost DJI Tello micro-drones and edge perception algorithms, the custom framework automates pipeline line-tracking, real-time hazard detection (fire, structural anomalies, foreign obstructions), and dual-UAV task coordination. When defects are logged by the primary drone, telemetry data and target inspection outcomes (`BUENO`/`MALO`) are written to a shared manifest (`traj_segments.txt`). Upon mission completion, an automated network script switches host Wi-Fi interfaces to dispatch a secondary drone directly to flagged hazard coordinates for targeted re-inspection—eliminating the need for manual flight intervention.

---

## Key Features

* **Vision-Guided Tracking & PID Control:** Combines HSV color segmentation, geometric line fitting (`cv2.fitLine`), and multi-axis closed-loop PID control ($x, y, \text{yaw}$) to maintain precise positioning above pipeline markers. Features fallback memory navigation and multi-stage elevation search routines for lost-line recovery.
* **Real-Time Edge Hazard Detection:** Runs an edge-optimized YOLOv8 model (`best.pt`) to classify targets in real time:
  * `0: FIRE` – Evaluated during hover inspection; flags point as `MALO` and triggers Arduino alarm payload (`b'1'`).
  * `1: ANOMALY` – Structural defect detected in active ROI; triggers a 4.0-second hovering inspection state.
  * `2: SHOE` – Foreign obstacle hazard; immediately pauses flight trajectory (`b'2'`) until clear (`b'0'`).
* **Sequential Dual-UAV Coordination & Wi-Fi Handoff:** Uses `pywifi` to handle automated SSID network handoffs between Drone 1 (`TELLO-FE193A`) and Drone 2 (`TELLO-9A57E0`). Drone 2 parses logged defect locations, computes target relative vector headings ($\Delta x, \Delta y, \theta, d$), and navigates directly to fire targets.
* **Multi-Threaded Architecture & Hardware Integration:** Integrates a dedicated FFMPEG UDP video streaming thread alongside real-time serial protocol communication (`pyserial`, `COM14` @ 9600 baud) for physical hardware status signaling via Arduino.
* **Custom Hardware Design:** Features a custom optical camera mount and enclosure designed in **SolidWorks** for optimized field-of-view during downward pipeline tracking.

## Repository Structure

```text
uav-vision-pipeline-inspection/
├── archive/                   # Archived development scripts and legacy versions
│   ├── legacy-versions/       # Earlier flight logic iterations
│   └── others/                # Experimental scripts
├── docs/                      # Technical reports, presentations, and project papers
├── models/                    # Trained neural network weights
│   └── best.pt                # YOLO object detection weights
├── src/                       # Production source code
│   └── reto_final.py          # Core multi-threaded control loop & vision pipeline
├── .gitignore                 # Version control exclusions
├── LICENSE                    # Project license
├── README.md                  # Project documentation
└── requirements.txt           # Python dependency manifest
