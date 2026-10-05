# PLADICS Deployment Module

This module contains the deployment pipeline of PLADICS.

## Components

- YOLO26n based plastic detection
- ByteTrack target tracking
- Rover navigation logic
- Conveyor collection control
- MQTT telemetry

## Hardware Target

- Raspberry Pi 4
- Raspberry Pi Camera Module V2
- L298N Motor Driver
- Conveyor based collection mechanism

## Current Status

Software implementation completed.

Hardware validation will be performed after Raspberry Pi integration.

## Structure

main.py
Main deployment runner

configs/
Deployment configuration

trackers/
ByteTrack configuration

hardware/
Motor controller interface
