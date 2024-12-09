# Generating RGB-Depth Images Using Stereo Vision

This repository contains Python code to generate RGB-depth images using stereo vision techniques. The core algorithm used in this project is **StereoSGBM (Semi-Global Block Matching)**, which generates depth maps from stereo image pairs. The project demonstrates how adjusting key parameters can affect depth map accuracy and computational efficiency.

## Files Overview

### 1. **disparity_map_generator.py**

This script implements the **StereoSGBM** algorithm for generating disparity maps from stereo image pairs. It reads two input images (left and right) in grayscale, calculates the disparity map using various parameters, and visualizes the results.

#### Features:
- Reads and preprocesses stereo images (left and right).
- Configurable parameters for block size, number of disparities, etc.
- Generates a disparity map and visualizes the result with a color map.
- Saves the generated disparity map as an image file.

#### How to use:
1. Find the folder which contains all folders of all datasets you want to analyze.
2. Folder[datasets...] and each dataset has a folder inside which contains the same name folder and also the format of middlebury 2014. (might need like 5 min to set up, I am too lazy to explain with precision)
2. Run the script:
   ```bash
   python dataSetRunner.py
3. The output will be csv of all datasets.

2. experiment_results_analyzer.py
This script runs a series of experiments to evaluate the performance of the StereoSGBM algorithm, focusing on the trade-offs between accuracy (MAE) and execution time. It tests different configurations (block size and disparity range) and generates visualizations such as scatter plots, heatmaps, and bar charts to analyze the results.

Features:
Reads ground truth disparity maps and computes the MAE (Mean Absolute Error).
Measures the execution time for each experiment configuration.
Visualizes the relationship between MAE and execution time.
Provides insights into the best configurations for optimal accuracy and performance.
How to use:
Place your stereo image pairs and corresponding ground truth disparity maps in the appropriate directories.

Run the script:

python dataSetRunner.py

The script will display charts and save the analysis results.

Requirements
Python 3.x
Required Python packages:
numpy
opencv-python
matplotlib
seaborn
imageio
scikit-learn

Dataset
This project uses the Middlebury Stereo 2014 dataset for stereo image pairs and ground-truth disparity maps. The dataset can be downloaded from Middlebury Stereo 2014.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
StereoSGBM: OpenCV's implementation of the Semi-Global Block Matching algorithm.
Middlebury Stereo Dataset: A benchmark dataset for evaluating stereo vision algorithms.