# Temperature Prediction of Phase Change Materials (PCMs) using Deep Learning and Experimental Photos

[![Paper](https://img.shields.io/badge/Paper-Energy%20Science%20%26%20Engineering-blue)](https://doi.org/10.1002/ese3.70110)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](https://github.com/mhranjbar/CLRE/PCM)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code and dataset for the paper **"End-To-End Deep Learning Temperature Prediction Algorithms of a Phase Change Materials From Experimental Photos"**, published in the journal *Energy Science & Engineering*.

This research introduces a novel, cost-effective method for predicting the temperature profile of a Phase Change Material (PCM) during its melting process, using only photos from a conventional digital camera.

<p align="center">
  <img src="https://raw.githubusercontent.com/mhranjbar/CLRE/refs/heads/main/PCM/images/video_2025-07-12_18-28-32.gif" alt="Animation of the melting process and temperature prediction">
  <br>
  <em>A graphical representation of the PCM melting process over time.</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mhranjbar/CLRE/refs/heads/main/PCM/images/ese370110-fig-0001-m.jpg" alt=" melting process and temperature prediction">
  <br>
  <em>A graphical representation of the PCM melting process, comparing the actual measured temperatures (dots) with the model's predictions (lines) over time.</em>
</p>

## 📝 Overview

Measuring the temperature at different locations within a Phase-change material (PCM) during its melting process is a significant challenge due to its irregular shape and nonlinear temperature changes. Traditional methods can be complex, inaccurate, or expensive.

In this study, we developed four end-to-end deep neural networks that take RGB photos of the PCM as input and predict the temperatures at four key locations as output. Our best-performing model, **WIV-DNN**, which is based on the MobileNets architecture, successfully predicts the temperature with an **average error of less than 0.9%** and an inference time of just **0.03 seconds** per image.

---

## ✨ Key Features

- **💡 Innovative:** This is the first study to predict the internal temperature of a PCM solely from visual images using deep learning, to the best of the authors' knowledge.
- **💰 Cost-Effective:** The method uses a standard digital camera, eliminating the need for expensive thermographic cameras.
- **⚡️ Fast & Accurate:** The WIV-DNN model achieves a Mean Squared Error (MSE) of 0.05 and an average error of 0.9%, with a rapid prediction time of 0.03s per image.
- **🛰️ Non-Contact:** After training, the system can monitor temperature profiles without requiring any physical probes inside the material, making it ideal for probe-free applications.
- **🌐 Generalizable:** The methodology has significant potential for other scientific and industrial applications, such as analyzing temperature levels in polar regions from satellite imagery or monitoring snow distribution on mountains.

---

## 📦 Dataset

The complete dataset includes **825 photos** of the PCM melting process, with corresponding temperature data for each photo. For each image, the temperatures at four locations, recorded by K-type thermocouples, are provided.

- **Number of Images:** 825
- **Image Resolution:** `600x600` pixels (cropped from `1920x1080`) 
- **Output Data:** 4 temperature values per image 
You can download the full dataset from the link below:

[**⬇️ Download the Dataset from Google Drive**](https://drive.google.com/file/d/1xxXln1PPOgPzk92ylIBm3hpi0Wk3XnxO/view?usp=sharing)

---

## 🤖 Pre-trained Model

We provide the best-performing model from our study, **WIV-DNN**, for direct use. This model is based on the efficient MobileNets architecture and consists of 3,261,796 parameters.

You can download the pre-trained model in `.h5` (TensorFlow/Keras) format from the link below:

[**⬇️ Download the WIV-DNN Model from Google Drive**](https://drive.google.com/file/d/1_wxrKXV9GO6zASgb4sceZsB7cvXctlma/view?usp=sharing)

---

## 🚀 How to Use

Install the Prerequisites and run the code.

### Prerequisites
- Python 3.8+
- TensorFlow
- Keras
- NumPy
- OpenCV-Python


## 📜 Citation
If you use this code, dataset, or methodology in your research, please cite our paper. Your support helps us to continue our work.
```bibtex
@article{Ranjbar2025,
  author = {Ranjbar, Mohammad Hassan and Gharali, Kobra and Ng, Artie},
  title = {End-To-End Deep Learning Temperature Prediction Algorithms of a Phase Change Materials From Experimental Photos},
  journal = {Energy Science & Engineering},
  volume = {13},
  pages = {3410--3421},
  year = {2025},
  doi = {10.1002/ese3.70110},
  keywords = {convolutional neural networks, Deep learning, deep neural networks, experimental pictures, phase change material, temperature measurements},
  publisher = {Society of Chemical Industry and John Wiley & Sons Ltd.}
}

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
