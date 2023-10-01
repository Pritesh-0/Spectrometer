import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('t3.png', cv2.IMREAD_GRAYSCALE)

pixel_positions = np.arange(0, image.shape[1])
calibration_factor=2
wavelengths = pixel_positions * calibration_factor

# Extract the spectrum
spectrum = np.sum(image, axis=0)  # Sum along columns
#print(spectrum)

known=spectrum

fft_known=np.fft.fft(known)
fft_spect=np.fft.fft(spectrum)

ccr=np.fft.ifft(fft_known*np.conj(fft_spect))


# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, spectrum, color='b', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Wavelength Spectrum')
plt.grid(True)
plt.show()


#plt.figure(figsize=(10, 5))
#plt.plot(np.abs(ccr), color='b', linewidth=2)
#plt.xlabel('Lag')
#plt.ylabel('Cross-correlation')
#plt.title('Cross-correlation of Spectra')
#plt.grid(True)
#plt.show()

# Calculate the maximum cross-correlation value
#max_correlation = np.max(np.abs(ccr))

# You can set a threshold to determine if the spectra match
#threshold = 0.9  # Adjust this threshold as needed
#if max_correlation >= threshold:
#    print("Spectra match.")
#else:
#    print("Spectra do not match.")

