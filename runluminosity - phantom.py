import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.signal import find_peaks
from datetime import datetime, timedelta


def processHF(hf_file):
    #get the timestamp from the name of the file
    timestamp = datetime.strptime(os.path.basename(hf_file).split('_')[0][:26], '%Y-%m-%d-%H-%M-%S-%f')
    add_pre_trigger_HF = timedelta(milliseconds=1)

    # Adjust the timestamp by adding 1 ms (name of the file + pre trigger time = ts of trigger event)
    timestamp += add_pre_trigger_HF

    return timestamp

def convert_mp4_to_jpg(mp4_path, output_folder, reference_ts):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(mp4_path)
    frame_count = 0

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {mp4_path}")

    time_per_frame = 9.349202045605408e-1   # The value of time per frame in milliseconds
    trigger_frame = 533                     # The frame where the trigger event occurs

    # pre_trigger  = 0.5002 seconds
    # post_trigger = 0.9994 seconds
    # quantity_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # time_per_frame = 1e3 * (pre_trigger + post_trigger) / quantity_of_frames # em ms

    reference_ts -= timedelta(milliseconds=(time_per_frame * trigger_frame))  

    while True:

        ts_now = reference_ts + timedelta(milliseconds=(time_per_frame * frame_count))

        # transform timestamp to string
        ts_now = ts_now.strftime('%Y-%m-%d-%H-%M-%S-%f')

        ret, frame = cap.read()
        if not ret:
            break
        # Corte o frame para a metade superior
        height = frame.shape[0]
        half_frame = frame[:height // 2, :]
        frame_path = os.path.join(output_folder, f'{frame_count:04d}-{ts_now}.jpg')
        cv2.imwrite(frame_path, half_frame)
        frame_count += 1

    cap.release()
    print(f"Converted {frame_count} frames from '{mp4_path}' to '{output_folder}'.")

def compute_luminosities(image_files, background_frame_count=200):

    total_files = len(image_files)

    if total_files < background_frame_count:
        raise ValueError(f"Need at least {background_frame_count} images to compute background.")

    print(f"Calculating background from first {background_frame_count} images...")
    background_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in image_files[50:background_frame_count]]
    background = np.mean(background_images, axis=0)

    luminosities = []

    print("Processing images and calculating luminosity:")
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        subtracted = cv2.subtract(img, background)
        subtracted[subtracted < 0] = 0
        total_luminosity = np.sum(subtracted)
        luminosities.append(total_luminosity)

        print(f"[{idx + 1:3d}/{total_files}] {os.path.basename(img_path)} -> Luminosity: {total_luminosity:.2f}")

    return np.array(luminosities)

def detect_peaks(luminosities, distance=20, prominence=17e4):
    peaks, properties = find_peaks(luminosities, distance=distance, prominence=prominence)
    return peaks, properties

def plotpeaks(folder_input, luminosities, peaks, start_index=0):


    plt.figure(figsize=(10, 5))
    x_axis = np.arange(start_index, start_index + len(luminosities))
    plt.plot(x_axis, luminosities, marker='o', label='Luminosity')
    plt.plot(x_axis[peaks], luminosities[peaks], 'rx', label='Detected Peaks')
    for peak in peaks:
        plt.annotate(f'{peak}', 
                     (x_axis[peak], luminosities[peak]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', fontsize=8, color='red')
    plt.title('Total Luminosity Over Time with Peaks')
    plt.xlabel('Frame Index')
    plt.ylabel('Total Luminosity')
    plt.xlim(min(x_axis), max(x_axis))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('.', folder_input, f'luminosity_peaks-{folder_input}.png'))
    plt.show()

if __name__ == "__main__":
    # The folder containing the video and the trigger file
    folders_inputs = [
        'v9.1_FNN_Y202501 1H001644.882607000 (20250626_~210552_UTC)',
        

    ]
        
    for folder_input in folders_inputs:

        # folder to store the selected peaks
        peak_folder = os.path.join('.', folder_input, f'selected-peaks-{folder_input}')
        os.makedirs(peak_folder, exist_ok=True)

        image_files = sorted(glob(os.path.join(folder_input, '*.jpg')))

        luminosities = compute_luminosities(image_files, background_frame_count=75)
        np.save(os.path.join('.', folder_input, f'luminosity-vector-{folder_input}.npy'), luminosities)

        peaks, properties = detect_peaks(luminosities)

        plotpeaks(folder_input, luminosities, peaks)

        print(f"\nDetected {len(peaks)} lightning events:")
        print("Frame indices of peaks:", peaks)

        for idx, peak_idx in enumerate(peaks):

            dst_folder = os.path.join(peak_folder, f'{peak_idx}')
            os.makedirs(dst_folder, exist_ok=True)

            list_of_peaks = np.linspace(peak_idx-4, peak_idx+5, 10, dtype=int)

            for frame_idx in list_of_peaks:
                # print(f"Processing frame index: {frame_idx}")
                if frame_idx < 0:
                    frame_idx = 0
                if frame_idx >= len(image_files):
                    frame_idx = len(image_files) - 1
                
                src_path = image_files[frame_idx]
                dst_path = os.path.join(dst_folder, os.path.basename(src_path))

                cv2.imwrite(dst_path, cv2.imread(src_path))
        print(f"Saved {len(peaks)} peak frames to '{peak_folder}' folder.")
