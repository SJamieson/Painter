import argparse
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--input_dir', type=str, help='Input directory')
parser.add_argument('--ref_dir', type=str, help='Reference directory')
parser.add_argument('--ref_csv', type=str, help='Reference CSV file')
parser.add_argument('--output_dir', type=str, help='Output directory')

args = parser.parse_args()


def sort_key_func(file_name):
    return int(file_name.split('_')[-1].split('.')[0])


input_files = sorted(os.listdir(args.input_dir), key=sort_key_func)
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
intensities /= intensities.max()

# plot normalized intensities
plt.plot(timestamps, intensities, label='Intensities')
plt.title('Normalized Intensities')

# Read CSV file for User Input values
df = pd.read_csv(args.ref_csv)

user_inputs = df['User Input'].values
user_inputs = (user_inputs - np.min(user_inputs)) / (np.max(user_inputs) - np.min(user_inputs))  # normalize user inputs

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

# plot user inputs
plt.plot(user_timestamps, user_inputs, label='User Inputs')
plt.legend()
plt.show()