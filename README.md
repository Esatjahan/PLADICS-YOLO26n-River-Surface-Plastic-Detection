# PLADICS

## PLADICS: A Lightweight YOLO26n-Based Framework for River-Surface Plastic Waste Detection and Collection


## Overview

PLADICS (Plastic Detection and Intelligent Collection System) is an integrated deep learning, robotics, and IoT-based framework for river-surface plastic waste detection and collection.

The framework combines lightweight object detection, target tracking, autonomous navigation, physical collection control, and IoT-based monitoring into a unified detection-to-collection workflow.


## Research Objective

The primary objective of PLADICS is to develop a lightweight and deployment-oriented plastic detection framework for autonomous collection systems.

The study focuses on single-class plastic detection, where all collectable plastic objects are treated as one target class to support navigation and collection decisions.


## System Overview

The complete PLADICS pipeline consists of:

```
River Surface Images
          |
          v
Dataset Auditing and Refinement
          |
          v
YOLO-based Detection Model Evaluation
          |
          v
YOLO26n Model Selection
          |
          v
ByteTrack Target Tracking
          |
          v
Rover Navigation Decision
          |
          v
Motor Control and Conveyor Collection
          |
          v
IoT Monitoring and Mission Logging
```


## Main Contributions

- Construction of refined single-class and multiclass river plastic datasets
- Comparative evaluation of lightweight YOLO, CNN-based, and transformer-based detectors
- Selection of YOLO26n as a lightweight deployment candidate
- Robustness evaluation under reduced training data conditions
- Confidence calibration analysis
- Detection-specific explainability analysis using SHAP
- Integration of detection, tracking, navigation, collection, and IoT monitoring


## Detection Models Evaluated

The framework evaluates multiple object detection architectures:

| Model | Category |
|---|---|
| YOLOv8n | Lightweight YOLO baseline |
| YOLOv10n | Lightweight YOLO baseline |
| YOLO11n | Lightweight YOLO baseline |
| YOLO26n | Lightweight deployment model |
| YOLO26s | Larger YOLO variant |
| NanoDet-Plus-M | Compact CNN detector |
| RT-DETR-L | Transformer-based detector |


## Final Selected Model

The selected deployment model is:

**YOLO26n-100**

Performance on the cleaned single-class test set:

| Metric | Value |
|---|---:|
| Precision | 0.919 |
| Recall | 0.844 |
| mAP@0.5 | 0.915 |
| mAP@0.5:0.95 | 0.608 |


## Dataset

Datasets used:

- River Surface Image Dataset (RiSID)
- Riverine Plastic Litter Dataset


Dataset refinement included:

- Annotation auditing
- Invalid bounding box removal
- Class correction
- Non-plastic contamination removal


The large-scale datasets, trained weights, and experimental artifacts are available separately:

Hugging Face Artifact Repository:

https://huggingface.co/datasets/Esatjahan/PLADICS-Full-Artifacts


## Repository Structure

```
PLADICS/

├── deployment/
│   Raspberry Pi deployment and hardware integration

├── scripts/
│   Training, preprocessing, and evaluation scripts

├── confidence_analysis/
│   Confidence calibration analysis

├── explainability/
│   Model explainability analysis

├── xai_yolo26/
│   YOLO26 explainability experiments

├── YOLO-26-CAM/
│   CAM visualization experiments

├── robustness_tests/
│   Reduced-data robustness experiments

├── qualitative_results/
│   Detection visualization examples

├── results/
│   Experimental results and evaluation outputs

├── figures/
│   Generated research figures

├── paper/
│   Manuscript and research documents

└── docs/
    Additional documentation
```


## Deployment Framework

The deployment module includes:

- YOLO26n inference pipeline
- ByteTrack-based target tracking
- Target selection logic
- Rover navigation control
- L298N motor controller interface
- Conveyor-based collection mechanism
- MQTT telemetry communication


## Hardware Platform

Target deployment platform:

- Raspberry Pi 4
- Raspberry Pi Camera Module V2
- L298N Motor Driver
- Conveyor-based collection system


## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

For Raspberry Pi deployment:

```bash
pip install -r requirements_raspberry_pi.txt
```


## Running the System

Example:

```bash
python deployment/pladics_deployment.py
```


## Artifact Availability

Due to repository size limitations:

- Source code and documentation are hosted on GitHub
- Large datasets, model weights, and experiment archives are hosted on Hugging Face


GitHub Repository:

https://github.com/Esatjahan/PLADICS-YOLO26n-River-Surface-Plastic-Detection


Hugging Face Repository:

https://huggingface.co/datasets/Esatjahan/PLADICS-Full-Artifacts


## Citation

If you use this repository, please cite:

```
PLADICS: A Lightweight YOLO26n-Based Framework for River-Surface Plastic Waste Detection and Collection
```
