import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from datetime import datetime

def record_speaker_data(channels=16, repeat=8, sleep_duration=3):

    folder_name = input("Enter a name for this recording session (folder name): ").strip()
    os.makedirs(folder_name, exist_ok=True)
    print(f"Saving files in folder: {folder_name}/")

    #excitation_path = "/home/apurv/data_recording/excitation.wav"

    excitation_path = input("Etner the path to excitation wav file")
    
    if not os.path.isfile(excitation_path):
        print("Error: The provided excitation file does not exist.")
        return
    
    base_filename = input("Enter a base name for the output recorded files (without extension): ").strip()

    excitation, fs = sf.read(excitation_path)

    if excitation.ndim > 1:
        excitation = excitation[:, 0] 

    excitation = excitation / np.max(np.abs(excitation))

    duration = len(excitation) / fs

    for i in range(repeat):

        print(f"\n--- Recording {i + 1}/{repeat} ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        recorded_filename = f"{timestamp}_{base_filename}_{i + 1}.wav"

        print(f"Playing excitation ({duration:.2f}s) and recording {channels}-channel response...")
        recorded = sd.playrec(excitation, samplerate=fs, channels=channels)
        
        sd.wait()
        print("Recording complete.")

        recorded_path = os.path.join(folder_name, recorded_filename)
        sf.write(recorded_path, recorded, fs)
        print(f"Saved {channels}-channel recording to: {recorded_path}")

        print("Computing impulse response for microphone 1 (channel 0)...")
        inv_filter = excitation[::-1]
        rir = fftconvolve(recorded[:, 0], inv_filter)

        start_sample = 21300
        end_sample = 22000
        rir_cropped = rir[start_sample:end_sample]

        start_sample = 21300
        end_sample = 22000
        rir_cropped = rir[start_sample:end_sample]

        plot_ir(start_sample, end_sample, rir_cropped, i)

        if i < repeat - 1:
            print(f"Waiting {sleep_duration} seconds before next recording...")
            time.sleep(sleep_duration)

    return excitation
    
def plot_ir(start_sample, end_sample, rir_cropped, i):
    
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(start_sample, end_sample), rir_cropped)
    plt.title(f"IR Segment (Mic 1) - Samples {start_sample} to {end_sample} - Recording {i + 1}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    record_speaker_data()
