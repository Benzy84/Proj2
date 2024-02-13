import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.restoration import unwrap_phase

# Cropping function
def crop_image_interactively(image_array, title='Select Crop Area'):
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap='gray')
    ax.set_title(title)
    crop_coords = []

    def on_select(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        crop_coords.extend([x1, y1, x2, y2])
        plt.close(fig)

    rect_selector = RectangleSelector(ax, on_select,
                                      useblit=True,
                                      button=[1], minspanx=5, minspany=5,
                                      spancoords='pixels', interactive=True)
    plt.show()

    if crop_coords:
        cropped_image = image_array[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
        return cropped_image, crop_coords
    else:
        return None, None

# Load and crop the original image
image_path = r'C:\Users\Owner\Desktop\1.tif'  # Update path as needed
original_image = Image.open(image_path)
original_image_array = np.array(original_image)

cropped_image, crop_coords = crop_image_interactively(original_image_array, 'Select Original Area')

# Perform FFT on the cropped image
fft_result = np.fft.fft2(cropped_image)
fft_shifted = np.fft.fftshift(fft_result)
fft_magnitude = np.log(np.abs(fft_shifted) + 1)

# Crop the FFT magnitude image interactively
cropped_fft_magnitude, fft_crop_coords = crop_image_interactively(fft_magnitude, 'Select FFT Area for Reconstruction')

if cropped_fft_magnitude is not None:
    cropped_fft = fft_shifted[fft_crop_coords[1]:fft_crop_coords[3], fft_crop_coords[0]:fft_crop_coords[2]]
    reconstructed_field = np.fft.ifft2(np.fft.ifftshift(cropped_fft))

    # Calculate amplitude and phase of the reconstructed field
    reconstructed_amplitude = np.abs(reconstructed_field)
    reconstructed_phase = np.angle(reconstructed_field)

    # Apply 2D phase unwrapping
    unwrapped_phase = unwrap_phase(reconstructed_phase)

    # Visualization of original cropped, reconstructed amplitude, and unwrapped phase in one figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cropped_image, cmap='gray')
    axs[0].set_title("Original Cropped Image")
    axs[0].axis('off')

    axs[1].imshow(reconstructed_amplitude, cmap='gray')
    axs[1].set_title("Reconstructed Amplitude")
    axs[1].axis('off')

    axs[2].imshow(unwrapped_phase, cmap='gray')
    axs[2].set_title("Unwrapped Phase")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
