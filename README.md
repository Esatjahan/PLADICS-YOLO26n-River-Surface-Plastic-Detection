# PLADICS

## PLADICS: A Lightweight YOLO26n-Based Framework for River-Surface Plastic Waste Detection and Collection

## Overview

PLADICS is an integrated deep learning and IoT-based framework designed for river-surface plastic waste detection and intelligent collection.

The framework combines lightweight object detection, target tracking, autonomous navigation, and IoT monitoring for deployment-oriented plastic waste collection.

## Research Objective

This work focuses on single-class plastic detection for autonomous collection.

The objective is to detect and localize collectable plastic objects in real-world river environments rather than performing fine-grained plastic type classification.

## System Pipeline

The complete pipeline consists of:

1. Dataset refinement and annotation auditing
2. YOLO-based plastic detection
3. Model evaluation and comparison
4. Explainability analysis
5. Target tracking using ByteTrack
6. Autonomous rover navigation
7. Plastic collection mechanism
8. IoT-based monitoring

## Main Contributions

- Cleaned single-class plastic detection dataset preparation
- Lightweight YOLO-based detection framework
- Comparative evaluation of multiple object detection models
- Explainability and confidence analysis
- Real-time deployment framework for Raspberry Pi-based rover

## Repository Structure

```
PLADICS/

├── deployment/
│   Deployment and hardware integration code

├── scripts/
│   Training, preprocessing and evaluation scripts

├── confidence_analysis/
│   Confidence calibration analysis

├── explainability/
│   Model explainability analysis

├── xai_yolo26/
│   YOLO26 explainability experiments

├── YOLO-26-CAM/
│   CAM visualization results

├── results/
│   Experimental results and metrics

├── figures/
│   Generated research figures

├── qualitative_results/
│   Detection visualization examples

├── robustness_tests/
│   Robustness evaluation

├── paper/
│   Research manuscript and documents

└── docs/
    Additional documentation
```

## Dataset

Datasets used:

- River Surface Image Dataset (RiSID)
- Riverine Plastic Litter Dataset

The datasets were refined through:

- Annotation auditing
- Invalid bounding box removal
- Non-plastic contamination removal

The complete large-scale datasets, trained models, and experimental artifacts are available at:

Hugging Face:

https://huggingface.co/datasets/Esatjahan/PLADICS-Full-Artifacts

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

For Raspberry Pi deployment:

```bash
pip install -r requirements_raspberry_pi.txt
```

## Deployment

The deployment module contains:

- YOLO26n inference
- ByteTrack-based target tracking
- Navigation decision logic
- Motor controller interface
- Conveyor collection control
- MQTT telemetry communication

## Hardware Platform

Target deployment hardware:

- Raspberry Pi 4
- Raspberry Pi Camera Module V2
- L298N Motor Driver
- Conveyor-based collection mechanism

## Running the System

Example:

```bash
python deployment/pladics_deployment.py
```

## Artifact Availability

Large datasets, model weights, and experiment files are hosted separately due to repository size limitations.

Hugging Face Artifact Repository:

https://huggingface.co/datasets/Esatjahan/PLADICS-Full-Artifacts

## Citation

If you use this repository, please cite:

**PLADICS: A Lightweight YOLO26n-Based Framework for River-Surface Plastic Waste Detection and Collection**
