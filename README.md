# Stereo depth with OpenCV StereoSGBM

Compute disparity maps from stereo image pairs with OpenCV StereoSGBM. Measure how block size and disparity range change the error and the run time.

University course project (2024).

## What it does

- `dataSetRunner.py` runs StereoSGBM on every scene in a Middlebury 2014 folder.
- It tries 4 block sizes (3, 5, 7, 11) and 3 disparity ranges (128, 256, 512).
- It saves one disparity image per setting, and writes MAE and run time per scene to `experiment_results.csv`.
- `visualise-results.py` reads that CSV and makes charts: MAE against run time, average MAE per setting, and a heatmap.

## Quick start

Install the requirements:

```bash
pip install -r requirements.txt
```

Download scenes from the [Middlebury Stereo 2014 dataset](https://vision.middlebury.edu/stereo/data/scenes2014/). Put them in one folder. Each scene folder must contain another folder with the same name:

```
datasets/
  Adirondack-imperfect/
    Adirondack-imperfect/
      calib.txt
      im0.png
      im1.png
      disp0.pfm
```

Run the experiment, then the charts:

```bash
python dataSetRunner.py datasets
python visualise-results.py
```

The scripts write their output files (PNG images and CSV files) to the current folder.

## How it works

1. Read the left image (`im0.png`) and right image (`im1.png`) in grayscale.
2. For each setting, run `cv2.StereoSGBM_create` with the chosen `blockSize` and `numDisparities`. The other parameters stay fixed.
3. Divide the SGBM output by 16. OpenCV returns disparity as fixed-point numbers scaled by 16.
4. Save a normalized 0-255 copy for viewing. This copy is not used for the error.

## Evaluation

The ground truth is `disp0.pfm`, in pixels. MAE is the mean absolute difference between the SGBM disparity (in pixels) and the ground truth. It uses only valid pixels:

- The ground truth is finite and greater than 0.
- SGBM found a match (disparity greater than 0).

Run time covers the SGBM call, the image write and the MAE step. This repository has no published results. Run the scripts on your own copy of the dataset to get numbers.

## License

MIT. See [LICENSE](LICENSE).

The dataset belongs to its authors. Cite the Middlebury 2014 paper if you use it: Scharstein et al., "High-resolution stereo datasets with subpixel-accurate ground truth", GCPR 2014.
