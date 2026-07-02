# Complete Self-Driving Car

An end-to-end autonomous-driving pipeline that combines classical computer vision and deep learning to perceive the road and drive a car inside the **Udacity Self-Driving Car Simulator**.

The project covers three areas of self-driving car software:

- **Lane Detection** — classical OpenCV pipeline (Canny edge detection + Hough transform)
- **Traffic Sign Classification** — CNN classifier for 43 German traffic-sign classes
- **Behavioral Cloning** — an NVIDIA-style CNN that learns to steer directly from camera images, deployed live via `drive.py`

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Training Your Own Model](#training-your-own-model)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview

A self-driving car senses its environment, interprets that data, and generates driving commands without human input. This project reproduces that pipeline at small scale using a driving simulator instead of a physical vehicle:

| Stage | What happens |
|---|---|
| **Perceive** | The simulator streams a live front-facing camera image |
| **Decide** | A trained CNN predicts the correct steering angle |
| **Act** | Steering and throttle commands are sent back to the simulator in real time |

Of the three modules, **Behavioral Cloning** is the one that actually drives the car in Autonomous Mode. Lane Detection and Traffic Sign Classification are standalone demonstrations of the perception techniques involved in autonomous driving.

---

## Repository Structure

```
Complete-Self-Driving-Car/
├── drive.py                              # Flask/Socket.IO server that drives the car live
├── model/
│   └── model.h5                          # Trained NVIDIA CNN weights used by drive.py
├── requirements.txt                      # Python dependencies
├── 1.Finding_Lanes/
│   ├── lane.py                           # Classical OpenCV lane-detection pipeline
│   └── artifacts/                        # Sample video and test image
├── Notebooks/
│   ├── Behavioral Cloning.ipynb          # Trains the NVIDIA CNN steering model
│   └── Traffic Signs Classification.ipynb# Trains the traffic-sign CNN classifier
├── Data/
│   └── driving_log.csv                   # Logged camera paths + steering/throttle/speed
├── Self_Driving_Car.ipynb                # Top-level / exploratory notebook
├── simulator-windows-64/                 # Windows build of the driving simulator
└── LICENSE
```

---

## How It Works

### 1. Lane Detection (`1.Finding_Lanes/lane.py`)

A purely classical computer-vision pipeline — no machine learning involved:

1. Convert frame to grayscale and apply Gaussian blur
2. Run Canny edge detection
3. Mask everything outside a triangular region of interest
4. Detect line segments with the Probabilistic Hough Transform
5. Average and extrapolate left/right segments into two lane lines
6. Overlay the lane lines on the original frame

### 2. Traffic Sign Classification (`Notebooks/Traffic Signs Classification.ipynb`)

A LeNet-inspired CNN trained on 32×32 grayscale, histogram-equalized traffic-sign images across 43 classes, using categorical cross-entropy loss and the Adam optimizer.

### 3. Behavioral Cloning (`Notebooks/Behavioral Cloning.ipynb` → `model/model.h5`)

An NVIDIA-style CNN that regresses a single steering angle directly from a preprocessed camera frame:

```
Input (66×200×3, YUV)
→ Conv 24@5×5 s2 → Conv 36@5×5 s2 → Conv 48@5×5 s2 → Conv 64@5×5
→ Flatten
→ Dense 100 → Dense 50 → Dense 10 → Dense 1 (steering angle)
```

Training data is balanced by capping oversampled steering bins, extended using the left/right camera images (±0.15 steering correction), and augmented on the fly with random pan, zoom, brightness shift, and horizontal flip.

### 4. Real-Time Inference (`drive.py`)

A Flask + Socket.IO server bridges the trained model and the simulator:

1. Simulator connects to `drive.py` on port `4567`
2. Each frame arrives with the car's current speed
3. `img_preprocess()` reshapes the frame; `model.predict()` returns a steering angle
4. Throttle is computed as `1 − speed / speed_limit`
5. Steering angle and throttle are sent back to the simulator

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Numerical / Data | NumPy, pandas, scikit-learn |
| Computer Vision | OpenCV (`opencv-contrib-python`), Pillow, imgaug, matplotlib |
| Serving / Networking | Flask, python-socketio, eventlet |
| Simulation | Udacity Self-Driving Car Simulator |
| Language | Python 3.7 |

---

## Installation

**1. Clone the repository**

```bash
git clone <repository-url>
cd Complete-Self-Driving-Car
```

**2. Create and activate an environment**

```bash
conda create -n sdcar python=3.7 -y
conda activate sdcar
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Get the simulator**

Download the Udacity Self-Driving Car Simulator (a pre-built Windows binary is also bundled under `simulator-windows-64/`).

---

## Usage

1. Launch the simulator and select a track
2. Switch to **Autonomous Mode**
3. Start the inference server:

```bash
python drive.py
```

The car will begin driving using predictions from `model/model.h5`.

---

## Training Your Own Model

1. In the simulator, use **Training Mode** to manually drive and record a new `driving_log.csv` and `IMG/` folder
2. Point `Notebooks/Behavioral Cloning.ipynb` at your new data directory and run it end to end
3. The notebook saves a new `model.h5` — copy it into `model/` to use it with `drive.py`

---

## Results

- Keeps the car centered in-lane around the simulator's test track using only camera input
- Steering behavior is learned entirely from human demonstration — no hand-coded lane-following logic at inference time
- Throttle smoothly eases off as speed approaches the software-defined limit
- Training/validation MSE loss is plotted across epochs in the notebook to confirm convergence

---

## Limitations & Future Work

**Current limitations**
- Trained and tested only in simulation, not on real roads
- Traffic-sign classifier runs standalone and isn't fused into `drive.py`'s control loop
- Single-track training data limits generalization to new environments

**Future directions**
- Fuse sign classification and lane detection into the live control loop
- Train across multiple tracks and lighting/weather conditions
- Add object detection for obstacles, pedestrians, and other vehicles
- Introduce temporal models (RNN/LSTM) to reason over sequences of frames

---

## License

This project is released under the [MIT License](LICENSE).
