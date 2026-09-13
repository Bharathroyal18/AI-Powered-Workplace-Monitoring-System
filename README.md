# AI-Powered Workplace Monitoring System

## Project Overview

The AI-Powered Workplace Monitoring System is a computer vision based application designed to automate employee face recognition and time-based attendance monitoring.

The system uses Haar Cascade for face detection and LBPH (Local Binary Patterns Histograms) for face recognition. It also includes model evaluation, robustness testing, fairness evaluation, and a web-based dashboard.

## Features

- Employee registration and face capture
- Haar Cascade face detection
- LBPH face recognition
- Real-time webcam recognition
- Time-based attendance tracking
- Configurable attendance percentage
- Model accuracy evaluation
- Brightness robustness testing
- Blur robustness testing
- Occlusion robustness testing
- Camera-motion transformation testing
- Precision, Recall and F1-score evaluation
- Per-employee performance evaluation
- Fairness and consistency evaluation
- Blur-based data augmentation
- Web dashboard
- Evaluation results dashboard

## Technologies Used

- Python 3
- OpenCV
- OpenCV-Contrib
- NumPy
- Flask
- HTML
- CSS
- Haar Cascade
- LBPH Face Recognition

## Project Structure

```text
AI-Powered-Workplace-Monitoring-System/
│
├── backend/
│   ├── app.py
│   ├── dashboard.py
│   ├── capture_faces.py
│   ├── capture_test.py
│   ├── train_model.py
│   ├── recognize.py
│   ├── attendance.py
│   ├── evaluation.py
│   ├── evaluation_metrics.py
│   ├── robustness_test.py
│   ├── blur_test.py
│   ├── occlusion_test.py
│   ├── camera_motion_test.py
│   ├── fairness.py
│   ├── fairness_improvement.py
│   ├── fairness_improved_evaluation.py
│   ├── blur_augmentation.py
│   ├── train_blur_model.py
│   ├── test_blur_model.py
│   ├── test_normal_blur_model.py
│   ├── response_time_test.py
│   ├── employees.json
│   ├── requirements.txt
│   ├── dataset/
│   ├── test_dataset/
│   ├── dataset_blur/
│   ├── dataset_augmented/
│   ├── trainer/
│   └── haarcascade/
│
├── frontend/
│   ├── index.html
│   ├── results.html
│   └── style.css
│
├── README.md
└── venv/