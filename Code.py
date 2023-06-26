import numpy as np
import scipy.signal as sps
import soundfile as sf
import matplotlib.pyplot as plt


def audio_compression(input_signal):
    # Compute the Fourier transform of the input signal
    global input_spectrum
    input_spectrum = np.fft.fft(input_signal)

    # Apply audio compression by reducing the amplitudes of the high-frequency components
    compression_factor = float(input('Enter the Compression Factor: '))
    compressed_spectrum = input_spectrum * compression_factor

    # Compute the inverse Fourier transform to obtain the compressed audio signal
    compressed_signal = np.fft.ifft(compressed_spectrum)

    return compressed_signal.real

def noise_cancellation(input_signal):
    # Compute the Fourier transform of the input signal
    global input_spectrum
    input_spectrum = np.fft.fft(input_signal)

    # Create a Chebyshev filter to remove noise from the audio signal
    filter_order = 1
    filter_cutoff = 100 / (sample_rate/2)  # Adjust this value based on the desired frequency cutoff
    b, a = sps.cheby1(filter_order, 0.5, filter_cutoff, btype='low', analog=False, output='ba')

    # Apply the filter to the input spectrum
    filtered_spectrum = sps.lfilter(b, a, input_spectrum)

    # Compute the inverse Fourier transform to obtain the filtered audio signal
    filtered_signal = np.fft.ifft(filtered_spectrum)

    return filtered_signal.real

# Load the input audio signal
input_file = input('Enter File Name (.wav only): ')
input_signal, sample_rate = sf.read(input_file)

# Choose the feature to use
print("Select the feature to use:")
print("1. Audio Compression")
print("2. Noise Cancellation")
feature_choice = int(input("Enter your choice (1 or 2): "))

# Apply the chosen feature to the input signal
if feature_choice == 1:
    output_signal = audio_compression(input_signal)
    feature_name = "Audio Compression"
else:
    output_signal = noise_cancellation(input_signal)
    feature_name = "Noise Cancellation"

# Write the output signal to a WAV file
output_file = 'output_audio.wav'
sf.write(output_file, output_signal, sample_rate)

# Plot the input and output signals for comparison
time = np.arange(len(input_signal)) / sample_rate
frequencies = np.fft.fftfreq(len(input_signal), d=1/sample_rate)
fft = np.fft.fft(input_signal)
amp = np.abs(fft)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, input_signal)
plt.title('Input Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3,1,2)
plt.plot(frequencies, amp)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(time, output_signal)
plt.title('Output Audio Signal ({})'.format(feature_name))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()