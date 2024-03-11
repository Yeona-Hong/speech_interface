import librosa
import os



def calculate_wav_info(y, sr, bit_depth, file_name):
    # Calculate the total time in seconds
    total_time_seconds = len(y) / sr

    # Calculate total information in bits (since it's mono, we don't multiply by 2)
    total_information_bits = total_time_seconds * sr * bit_depth

    # Calculate the total information in bytes
    total_information_bytes = total_information_bits / 8

    # Get the file size in bytes
    file_size_bytes = os.path.getsize(file_name)

    # Output the results
    print(f"Total time: {total_time_seconds:.2f} seconds")
    print(f"Sampling rate: {sr} Hz")
    print(f"Total information: {total_information_bytes} bytes")
    print(f"File size: {file_size_bytes} bytes")

    # Compare the total information with the file size
    if total_information_bytes > file_size_bytes:
        print("The calculated information is larger than the file size. This might be an error in calculation or assumptions about bit depth.")
    elif total_information_bytes < file_size_bytes:
        print("The file size is larger, possibly due to metadata, encoding overhead, or because the actual bit depth/sample format is different.")
    else:
        print("The calculated information and file size are equal.")
    
def main():
    # Define the path to your WAV file
    file_name = 'recording_48000_mono.wav'  # Replace this with the path to your actual WAV file
    bit_depth = 16  # Bit depth per sample
    sampling_rate = 48000  # Sampling frequency in Hz
    
    y, sr = librosa.load(file_name, sr=sampling_rate, mono=True)
    
    calculate_wav_info(y, sr, bit_depth, file_name)
    

if __name__ == "__main__":
    main()