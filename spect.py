import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def check(elements,peaks,thresh):
    for element, ref_peaks in elements.items():
        diff = all(abs(peak-ref_peak) <=thresh for peak, ref_peak in zip(peaks,ref_peaks))
        print(element,' : ',diff)
        if diff:
            print(element)


elements={
        'Orthorhombic sulphur':[265,285],
        'Polymeric sulphur':[360],
        'Sulphur Dioxide':[280],
        'Iron-sulphur clusters':[450,620],
        'Phosphate':[370,436,690,827],
        'Nitrate':[203,302],
        'Manganese':[336,358,401,436,530],
        'Zinc':[302,455],
        'Citric Acid':[200],
        'Butyric Acid':[206],
        'Iron Oxide Nanoparticles':[295],
        'Fe203 NP':[320,420],
        'Al203':[491,562,650],
        'Silica NP':[206]
        }


image = cv2.imread('t2.png', cv2.IMREAD_GRAYSCALE)

pixel_positions = np.arange(0, image.shape[1])
#print(pixel_positions)

calibration_factor=2
wavelengths = pixel_positions * calibration_factor

# Extract the spectrum
spectrum = np.sum(image, axis=0)  # Sum along columns
#print(len(spectrum),len(pixel_positions))

peaks=find_peaks(spectrum,height=10,threshold=5,distance=5)
print(wavelengths[peaks[0]])


#known=spectrum

#fft_known=np.fft.fft(known)
#fft_spect=np.fft.fft(spectrum)
#print(fft_spect)
#ccr=np.fft.ifft(fft_known*np.conj(fft_spect))
#print(ccr)
#plt.plot(ccr)

# Plot the spectrum

plt.style.use('Solarize_Light2')
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, spectrum, color='b', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Wavelength Spectrum')
plt.grid(True)
plt.show()

check(elements,wavelengths[peaks[0]],10)

fig,ax=plt.subplots(4,4)
r,c=0,0
for i in elements:
    #ax[r,c].figure(figsize=(10, 5))
    mini,maxi=elements[i][0]-50,elements[i][-1]+50
    ax[r,c].plot(wavelengths, spectrum, color='b', linewidth=2)
    ax[r,c].set_xlim([mini,maxi])
    for j in elements[i]:
        ax[r,c].axvline(x=j,color='red',linestyle='dashed')
    #ax[r,c].set(xlabel='Wavelength (nm)')
    #ax[r,c].set(ylabel='Intensity')
    ax[r,c].set_title(i)
    #ax[r,c].grid(True)
    r+=1
    if r>3:
        r=0
        c+=1
    if c>3:
        break


plt.subplots_adjust(top=0.94,
bottom=0.07,
left=0.125,
right=0.9,
hspace=0.315,
wspace=0.2)
#plt.subplot_tool()
#print(plt.style.available)
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

