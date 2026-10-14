import numpy as np
import cv2
import os
import time
import argparse
import pandas as pd

def read_pfm(file):
    """
    Read a PFM ground truth file as a float32 array (values in pixels).
    """
    with open(file, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        if header not in ('Pf', 'PF'):
            raise ValueError(f"{file} is not a PFM file")
        channels = 3 if header == 'PF' else 1
        width, height = map(int, f.readline().decode('ascii').split())
        scale = float(f.readline().decode('ascii').strip())
        dtype = '<f4' if scale < 0 else '>f4'
        data = np.fromfile(f, dtype=dtype, count=width * height * channels)
    shape = (height, width, channels) if channels == 3 else (height, width)
    # PFM stores rows from bottom to top, so flip to the usual top-to-bottom order.
    return np.flipud(data.reshape(shape)).astype(np.float32)

def compute_mae(disparity_map, ground_truth):
    """
    Compute Mean Absolute Error between disparity map and ground truth.
    """
    # Both inputs must be float32 disparities in pixels. Only compare pixels
    # where both the ground truth and the SGBM result are valid.
    valid = np.isfinite(ground_truth) & (ground_truth > 0) & (disparity_map > 0)
    if not valid.any():
        return float('nan')
    return float(np.mean(np.abs(disparity_map[valid] - ground_truth[valid])))

def read_calib(calib_path):
    """
    Read and parse the calibration data from the text file for each dataset folder.
    """
    calib_data = {}
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('cam0'):
                # Extract the matrix data from 'cam0' line
                cam0_matrix_str = line.split('=')[1].strip().strip('[]')  # Remove the leading 'cam0=' and brackets
                cam0_matrix = np.array([list(map(float, row.split())) for row in cam0_matrix_str.split(';') if row.strip()])
                calib_data['cam0'] = cam0_matrix
            elif line.startswith('cam1'):
                # Extract the matrix data from 'cam1' line
                cam1_matrix_str = line.split('=')[1].strip().strip('[]')  # Remove the leading 'cam1=' and brackets
                cam1_matrix = np.array([list(map(float, row.split())) for row in cam1_matrix_str.split(';') if row.strip()])
                calib_data['cam1'] = cam1_matrix
            elif line.startswith('doffs'):
                calib_data['doffs'] = float(line.split('=')[1].strip())
            elif line.startswith('baseline'):
                calib_data['baseline'] = float(line.split('=')[1].strip())
            elif line.startswith('ndisp'):
                calib_data['ndisp'] = int(line.split('=')[1].strip())
            elif line.startswith('width'):
                calib_data['width'] = int(line.split('=')[1].strip())
            elif line.startswith('height'):
                calib_data['height'] = int(line.split('=')[1].strip())
    
    return calib_data

def dataSetRunner(dataset_path):
    """
    Main function to process all datasets, test performance and accuracy.
    Outputs results in a CSV format for analysis.
    """
    # Parameters for experiments
    block_sizes = [3, 5, 7, 11]  # Different block sizes for StereoSGBM
    num_disparities_values = [128, 256, 512]  # Different disparity values for StereoSGBM
    all_results = []  # List to store results for all datasets
    
    # Iterate through each folder
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder, folder)
        print(f"\nProcessing dataset folder: {folder}")
        
        # Read the calibration file from the specific folder
        calib_path = os.path.join(folder_path, 'calib.txt')
        calib_data = read_calib(calib_path)
        
        # Read images and ground truths
        imgL = cv2.imread(os.path.join(folder_path, "im0.png"), 0)
        imgR = cv2.imread(os.path.join(folder_path, "im1.png"), 0)  # Use im1.png for right image
        ground_truth = read_pfm(os.path.join(folder_path, "disp0.pfm"))
        
        # Start processing for this dataset folder
        folder_results = {'folder': folder}
        
        # Loop through different block sizes and number of disparities
        for block_size in block_sizes:
            for num_disparities in num_disparities_values:
                print(f"Testing with Block Size = {block_size}, Num Disparities = {num_disparities}")

                # Set parameters for StereoSGBM
                stereo = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=num_disparities,
                    blockSize=block_size,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=10,
                    speckleRange=8
                )

                # Start timer for processing
                start_time = time.time()

                # Compute disparity map using StereoSGBM
                disp = stereo.compute(imgL, imgR).astype(np.float32)

                # StereoSGBM returns fixed-point values scaled by 16.
                # Divide by 16 to get disparity in pixels, the same unit as the ground truth.
                disp_px = disp / 16.0
                # Pixels SGBM could not match are negative (-1 in fixed point). Mark them invalid.
                disp_px[disp < 0] = 0

                # Normalize a copy for visualization only. Never use it for MAE.
                disp_vis = cv2.normalize(disp_px, None, 0, 255, cv2.NORM_MINMAX)
                disp_vis = np.uint8(disp_vis)

                # Save the disparity map
                disparity_map_path = f"disparity_map_{folder}_block{block_size}_disparities{num_disparities}.png"
                cv2.imwrite(disparity_map_path, disp_vis)

                # Compute MAE between the disparity in pixels and the ground truth
                mae = compute_mae(disp_px, ground_truth)
                print(f"MAE: {mae:.4f}")

                # Measure execution time
                execution_time = time.time() - start_time
                print(f"Execution Time: {execution_time:.2f} seconds")

                # Store the results for this configuration
                folder_results[f"block_{block_size}_disparities_{num_disparities}_mae"] = mae
                folder_results[f"block_{block_size}_disparities_{num_disparities}_time"] = execution_time

        # Append the results of the current folder to the overall results
        all_results.append(folder_results)
    
    # Create a pandas DataFrame to structure and analyze results
    results_df = pd.DataFrame(all_results)
    
    # Output results to CSV
    results_df.to_csv('experiment_results.csv', index=False)
    print("\nExperiment results saved to 'experiment_results.csv'")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StereoSGBM on Middlebury 2014 scenes and write experiment_results.csv.")
    parser.add_argument("dataset_dir", help="folder with one sub-folder per scene (scene/scene/im0.png, ...)")
    args = parser.parse_args()
    results = dataSetRunner(args.dataset_dir)
    print(results)
