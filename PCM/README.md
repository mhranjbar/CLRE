# Temperature Prediction of Phase Change Materials (PCMs) using Deep Learning and Experimental Photos

[![Paper](https://img.shields.io/badge/Paper-Energy%20Science%20%26%20Engineering-blue)](https://doi.org/10.1002/ese3.70110)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](https://github.com/mhranjbar/CLRE/PCM)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code and dataset for the paper **"End-To-End Deep Learning Temperature Prediction Algorithms of a Phase Change Materials From Experimental Photos"**, published in the journal *Energy Science & Engineering*.

This research introduces a novel, cost-effective method for predicting the temperature profile of a Phase Change Material (PCM) during its melting process, using only photos from a conventional digital camera.

<p align="center">
  <img src="https://raw.githubusercontent.com/mhranjbar/CLRE/main/PCM/result.gif" alt="Animation of the melting process and temperature prediction">
  <br>
  <em>A graphical representation of the PCM melting process, comparing the actual measured temperatures (dots) with the model's predictions (lines) over time.</em>
</p>

<p align="center">
  <img src="https://scijournals.onlinelibrary.wiley.com/cms/asset/25b0f617-cd9f-4f51-b678-7e4602ca280e/ese370110-fig-0001-m.jpg" alt="(A) Schematic of the experimental setup. (B) Locations of four thermometers inside the container of the PCM.n">
  <br>
  <em>A graphical representation of the PCM melting process, comparing the actual measured temperatures (dots) with the model's predictions (lines) over time.</em>
</p>

## 📝 Overview

[cite_start]Measuring the temperature at different locations within a Phase-change material (PCM) during its melting process is a significant challenge due to its irregular shape and nonlinear temperature changes[cite: 13]. [cite_start]Traditional methods can be complex, inaccurate, or expensive[cite: 15].

[cite_start]In this study, we developed four end-to-end deep neural networks that take RGB photos of the PCM as input and predict the temperatures at four key locations as output[cite: 18, 171]. [cite_start]Our best-performing model, **WIV-DNN**, which is based on the MobileNets architecture, successfully predicts the temperature with an **average error of less than 0.9%** and an inference time of just **0.03 seconds** per image[cite: 20, 524, 533].

---

## ✨ Key Features

- [cite_start]**💡 Innovative:** This is the first study to predict the internal temperature of a PCM solely from visual images using deep learning, to the best of the authors' knowledge[cite: 70].
- [cite_start]**💰 Cost-Effective:** The method uses a standard digital camera, eliminating the need for expensive thermographic cameras[cite: 21, 536, 538].
- [cite_start]**⚡️ Fast & Accurate:** The WIV-DNN model achieves a Mean Squared Error (MSE) of 0.05 and an average error of 0.9%, with a rapid prediction time of 0.03s per image[cite: 524, 533].
- [cite_start]**🛰️ Non-Contact:** After training, the system can monitor temperature profiles without requiring any physical probes inside the material, making it ideal for probe-free applications[cite: 80, 539].
- [cite_start]**🌐 Generalizable:** The methodology has significant potential for other scientific and industrial applications, such as analyzing temperature levels in polar regions from satellite imagery or monitoring snow distribution on mountains[cite: 440, 441].

---

## 📦 Dataset

[cite_start]The complete dataset includes **825 photos** of the PCM melting process, with corresponding temperature data for each photo[cite: 130]. [cite_start]For each image, the temperatures at four locations, recorded by K-type thermocouples, are provided[cite: 93, 130].

- [cite_start]**Number of Images:** 825 [cite: 130]
- [cite_start]**Image Resolution:** `600x600` pixels (cropped from `1920x1080`) [cite: 124, 125]
- [cite_start]**Output Data:** 4 temperature values per image [cite: 130]

You can download the full dataset from the link below:

[**⬇️ Download the Dataset from Google Drive**](YOUR-GDRIVE-LINK-FOR-DATASET)

---

## 🤖 Pre-trained Model

[cite_start]We provide the best-performing model from our study, **WIV-DNN**, for direct use[cite: 396]. [cite_start]This model is based on the efficient MobileNets architecture and consists of 3,261,796 parameters[cite: 261, 264].

You can download the pre-trained model in `.h5` (TensorFlow/Keras) format from the link below:

[**⬇️ Download the WIV-DNN Model from Google Drive**](YOUR-GDRIVE-LINK-FOR-MODEL)

---

## 🚀 How to Use

Follow the steps below to run the code and test the model on your own images.

### 1. Prerequisites
- Python 3.8+
- [cite_start]TensorFlow [cite: 273]
- NumPy
- OpenCV-Python
