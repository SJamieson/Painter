import argparse
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from itertools import combinations

parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--input_dir', default=None, type=str, help='Input directory')
parser.add_argument('--ref_dir', type=str, help='Reference directory')
parser.add_argument('--ref_csv', type=str, help='Reference CSV file')
parser.add_argument('--output_dir', type=str, help='Output directory')

args = parser.parse_args()

if args.input_dir is not None:
    input_files = sorted(os.listdir(args.input_dir), key=lambda file_name: int(file_name.split('_')[-1].split('.')[0]))
    ref_files = sorted(os.listdir(args.ref_dir))

    if len(input_files) != len(ref_files):
        raise ValueError("Number of files in input directory and reference directory are not equal")

    for input_file, ref_file in zip(input_files, ref_files):
        shutil.copy(os.path.join(args.input_dir, input_file), os.path.join(args.output_dir, ref_file))

output_files = sorted(os.listdir(args.output_dir))
intensities = []  # hold intensities
timestamps = []  # hold timestamps
for output_file in tqdm(output_files):
    img = Image.open(os.path.join(args.output_dir, output_file)).convert('L')
    img_array = np.array(img)
    total_intensity = np.sum(img_array)
    intensities.append(total_intensity / (img.width * img.height))  # normalize intensity

    timestamp_str = os.path.splitext(output_file)[0]  # Get the filename without extension
    secs, nsecs = map(int, timestamp_str.split('-'))  # Split the seconds and nanoseconds part
    timestamp = secs + nsecs * 1e-9
    timestamps.append(timestamp)

timestamps, intensities = np.array(timestamps), np.array(intensities)

# plot normalized intensities

# Read CSV file for User Input values
df = pd.read_csv(args.ref_csv)

user_inputs = df['User Input'].values
# user_inputs = (user_inputs - np.min(user_inputs)) / (np.max(user_inputs) - np.min(user_inputs))  # normalize user inputs

# get timestamps from Image Filename column in CSV
user_timestamps = []
for filename in df['Image Filename'].values:
    timestamp_str = os.path.splitext(filename)[0]  # Get the filename without extension
    secs, nsecs = map(int, timestamp_str.split('-'))  # Split the seconds and nanoseconds part
    timestamp = secs + nsecs * 1e-9
    user_timestamps.append(timestamp)
user_timestamps, user_inputs = np.array(user_timestamps), np.array(user_inputs)

user_timestamps -= timestamps.min()
timestamps -= timestamps.min()

intensity_interpolator = interp1d(timestamps, intensities, kind='linear', fill_value='extrapolate')
user_intensities_interpolated = intensity_interpolator(user_timestamps)

# Define correlation function
def compute_correlation(thresholds):
    labels = np.zeros_like(user_intensities_interpolated)
    labels[user_intensities_interpolated > thresholds[0]] = 1
    labels[user_intensities_interpolated > thresholds[1]] = 2
    return -pearsonr(labels, user_inputs)[0]

# Generate combinations of potential thresholds
potential_thresholds = list(np.linspace(0, intensities.max(), 10))
threshold_combinations = list(combinations(potential_thresholds, 2))

# Find the thresholds that maximize correlation
best_thresholds = min(threshold_combinations, key=compute_correlation)
print(f'Best thresholds: {best_thresholds}')

# Apply the thresholds to the intensities to get the labels
labels = np.zeros_like(intensities)
labels[intensities > best_thresholds[0]] = 1
labels[intensities > best_thresholds[1]] = 2

# plot user inputs
plt.plot(timestamps, labels + 1, label='Intensities')
# plt.axhline(y=best_thresholds[0], color='r', linestyle='--', label='low threshold')
# plt.axhline(y=best_thresholds[1], color='g', linestyle='--', label='high threshold')
# plt.title('Normalized Intensities')
plt.plot(user_timestamps, user_inputs, label='User Inputs')
plt.legend()
plt.show()

# plot user inputs
plt.plot(timestamps, intensities / intensities.max(), label='Intensities')
plt.title('Normalized Intensities with Thresholds')
plt.plot(user_timestamps, (user_inputs - user_inputs.min()) / (user_inputs.max() - user_inputs.min()), label='User Inputs')
plt.legend()
plt.show()