import librosa
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def calculate_wav_info(y, sr, bit_depth, file_name):
    total_time_seconds = len(y) / sr
    total_information_bits = total_time_seconds * sr * bit_depth
    total_information_bytes = int(total_information_bits / 8)
    file_size_bytes = os.path.getsize(file_name)

    print(f"Total time: {total_time_seconds:.2f} seconds")
    print(f"Sampling rate: {sr} Hz")
    print(f"Total information: {total_information_bytes} bytes")
    print(f"File size: {file_size_bytes} bytes")

    if total_information_bytes > file_size_bytes:
        print("The calculated information is larger than the file size. This might be an error in calculation or assumptions about bit depth.")
    elif total_information_bytes < file_size_bytes:
        print("The file size is larger due to the file header in WAV format. Adding the file header size of 44 bytes in WAV format results in the same value.")
    else:
        print("The calculated information and file size are equal.")

def down_sampling(y, sr, resampling_rate):
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=resampling_rate)
    file_name = f'/home/yeona/speech_interface/output/week1/output_downsample_{resampling_rate}.wav'
    sf.write(file=file_name, data=y_resampled, samplerate=resampling_rate)

def save_spectrum_plot(y, sr, frame_length_ms, frame_number, title, filename_prefix):
    frame_size = int(sr * frame_length_ms / 1000)
    hop_length = frame_size // 2  # 50% 오버랩을 사용
    hann_window = np.hanning(frame_size)
    
    
    magnitude_db_frames = []
    phase_frames = []

    for i in range(0, len(y), hop_length):
        end_idx = i + frame_size
        if end_idx < len(y):
            y_frame = y[i:end_idx]
        else:
            y_frame *= hann_window
        
        Y = np.fft.fft(y_frame)
        freqs = np.fft.fftfreq(frame_size, 1/sr)

        magnitude = np.abs(Y)
        phase = np.angle(Y)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        magnitude_db_frames.append(magnitude_db)
        phase_frames.append(phase)
        
    magnitude_db_frames = np.array(magnitude_db_frames).T
    phase_frames = np.array(phase_frames).T
    
    
    if len(magnitude_db_frames[0]) > frame_number and len(phase_frames[0]) > frame_number:
        plt.figure(figsize=(10, 4))
        plt.plot(freqs[:frame_size//2], magnitude_db_frames[:frame_size//2, frame_number])
        plt.title(f'/home/yeona/speech_interface/output/week1/{title} - Magnitude Spectrum (Frame {frame_number + 1})')
        plt.ylabel('Magnitude (dB)')
        plt.xlabel('Frequency (Hz)')
        plt.savefig(f'/home/yeona/speech_interface/output/week1/{filename_prefix}_frame{frame_number + 1}_magnitude_spectrum.png')
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(freqs[:frame_size//2], phase_frames[:frame_size//2, frame_number])
        plt.title(f'{title} - Phase Spectrum (Frame {frame_number + 1})')
        plt.ylabel('Phase (Radians)')
        plt.xlabel('Frequency (Hz)')
        plt.savefig(f'/home/yeona/speech_interface/output/week1/{filename_prefix}_frame{frame_number + 1}_phase_spectrum.png')
        plt.close()
    else:
        print("Requested frame number exceeds the total number of frames.")



def main():
    file_name = '/home/yeona/speech_interface/source/recording_48000_mono.wav'  # Replace this with the path to your actual WAV file
    bit_depth = 16  # Bit depth per sample
    sampling_rate = 48000  # Sampling frequency in Hz
    y, sr = librosa.load(file_name, sr=sampling_rate, mono=True)
    frame_length = 20
    frame_number = 1
    
    calculate_wav_info(y, sr, bit_depth, file_name)
    save_spectrum_plot(y, sr, frame_length, frame_number, str(sr), filename_prefix = str(sr))
    

    resampling_rate = 16000
    down_sampling(y, sr, resampling_rate)
    file_name = f'/home/yeona/speech_interface/output/week1/output_downsample_{resampling_rate}.wav'

    y, sr = librosa.load(file_name, sr = resampling_rate, mono=True)
    calculate_wav_info(y, sr, bit_depth, file_name)
    save_spectrum_plot(y, sr, frame_length, frame_number,  str(sr), filename_prefix = str(sr))

    resampling_rate = 8000
    down_sampling(y, sr, resampling_rate)
    file_name = f'/home/yeona/speech_interface/output/week1/output_downsample_{resampling_rate}.wav'

    y, sr = librosa.load(file_name, sr = resampling_rate, mono=True)
    calculate_wav_info(y, sr, bit_depth, file_name)
    save_spectrum_plot(y, sr, frame_length, frame_number,  str(sr), filename_prefix = str(sr))

if __name__ == "__main__":
    main()