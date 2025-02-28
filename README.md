# Transgender Voice Training Tool

Transgender Voice Training Tool being built in Python.

**Latest Download:** [https://github.com/Kavex/VocalTuner/releases/](https://github.com/Kavex/VocalTuner/releases/)

This Python-based tool provides real-time analysis of vocal pitch and resonance (through formant frequencies) for transgender voice training. The tool offers live visual feedback and summary metrics to help users monitor and adjust their voice toward their desired pitch and resonance targets.

## Features

- **Real-Time Audio Analysis:**  
  Captures audio from your selected microphone in real time using PyAudio.

- **Pitch Detection & Note Conversion:**  
  Uses an autocorrelation algorithm to estimate the fundamental frequency (pitch) and converts it to the nearest musical note (e.g. A3, D4).

- **Formant Analysis:**  
  Computes the first two formant frequencies (F1 and F2) using a basic Linear Predictive Coding (LPC) approach. These formants are key for evaluating vocal resonance and are critical in transgender voice training.

- **Quality Metrics:**  
  Calculates basic voice quality metrics such as jitter, shimmer, and Harmonic-to-Noise Ratio (HNR).

- **Live Charts:**  
  Displays a live, moving-window chart (last 100 samples) of pitch and formant data on the left side of the window.

- **Metrics Summary:**  
  Provides a fixed metrics summary box on the right side of the window that displays:
  - **Pitch Summary:** Minimum, maximum, and average pitch.
  - **Formant Summary:** Minimum, maximum, and average values for F1 and F2.
  - **Quality Metrics:** Jitter, shimmer, and HNR.
  
  The summary metrics are updated automatically at a configurable reset interval (default is 3 seconds) without affecting the live chart.

- **User Controls:**  
  - **Microphone Selection:** Choose from available audio devices.
  - **Sensitivity Slider:** Adjust the RMS threshold for voice detection.
  - **Reset Interval Field:** Set the interval (in seconds) at which summary metrics are recalculated and reset.
  - **Start/Stop Buttons:** Control the real-time audio processing.

- **Guidance Information:**  
  The tool displays pitch range guidelines for transgender voice training:
  - **Gender Neutral:** 150–185 Hz
  - **Vocal Feminization:** Aim ~180 Hz
  - **Vocal Masculinization:** Aim ~150 Hz
  - **Average Female Pitch:** ~225 Hz
  - **Average Male Pitch:** ~125 Hz
  
  *Note: These are general guidelines. Please consult a speech therapist for personalized advice.*

## Dependencies

- Python 3.x
- [NumPy](https://numpy.org/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [Matplotlib](https://matplotlib.org/)
- Tkinter (usually bundled with Python)

## Installation

To run from source code, open your terminal or command prompt in the project directory and install the required packages:

```bash
pip install numpy pyaudio matplotlib

![image](https://github.com/user-attachments/assets/32ed39c1-68c8-4527-bd0e-b8cd5a3709eb)
