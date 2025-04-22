import cv2
import numpy as np
import torch
import math
import time
from skimage.metrics import structural_similarity as ssim
from propagation_ASM import *  # Custom Angular Spectrum Method implementation
from model import AdvancedPhaseEstimationNet  # Complex-valued UNet model

# ----------------------------- Parameters ----------------------------- #
img_num = 805  # Image index to process
z = 200  # Propagation distance (mm)
pitch = 0.0036  # Pixel pitch (mm)
wavelength = 0.000532  # Wavelength (mm) — Green laser
n = 1072  # Image height
m = 1920  # Image width

# ----------------------------- Frequency Grid Setup ----------------------------- #
x = torch.linspace(-n//2, n//2 - 1, n)
y = torch.linspace(-m//2, m//2 - 1, m)
v = 1 / (n * pitch)
u = 1 / (m * pitch)
fx = x * v
fy = y * u
fX, fY = torch.meshgrid(fx, fy, indexing='ij')

# ----------------------------- Transfer Functions ----------------------------- #
H = (-2 * np.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX)**2 - (wavelength * fY)**2)
H2 = (2 * np.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX)**2 - (wavelength * fY)**2)

# Apply band-limiting filter
xlimit = 1 / (torch.sqrt((2 / m / pitch * z)**2 + 1)) / wavelength
ylimit = 1 / (torch.sqrt((2 / n / pitch * z)**2 + 1)) / wavelength
band_limit_mask = ((torch.abs(fX) < xlimit) & (torch.abs(fY) < ylimit)).int()

# Create complex transfer functions with mask
H = torch.complex(torch.cos(H), torch.sin(H)) * band_limit_mask
H2 = torch.complex(torch.cos(H2), torch.sin(H2)) * band_limit_mask
H, H2 = H.cuda(), H2.cuda()

# ----------------------------- PSNR Function ----------------------------- #
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

# ----------------------------- Model Setup ----------------------------- #
model = AdvancedPhaseEstimationNet()
model_name = '2024-02-21_weight_E100_200_0.000532.pth' # model path
model.load_state_dict(torch.load(f'./model_output/{model_name}'), strict=False)

# Move model to GPU
if torch.cuda.is_available():
    model = model.cuda()

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Parameters:", total_params)

# ----------------------------- Load Image ----------------------------- #
path = f'/DIV2K/DIV2K_valid_HR/'
img_id = f'0{img_num}'
img = cv2.imread(path + img_id + '.png')
img_resized = cv2.resize(img, (1920, 1072))

# Use red channel for reconstruction
gray = cv2.split(img_resized)[2]
cv2.imwrite(f'result/{img_id}.png', gray)  # Save grayscale input
gray = np.reshape(gray, (1, 1, 1072, 1920))

# Convert to tensor and normalize
target_amp = torch.from_numpy(gray).float().cuda() / 255.0
target_amp_complex = torch.complex(target_amp, torch.zeros_like(target_amp))

# ----------------------------- Inference ----------------------------- #
with torch.no_grad():
    # Warm-up & measure generation time
    start_time = time.time()
    for _ in range(100):
        output = model(target_amp_complex, H2)
    elapsed = (time.time() - start_time) / 100
    print('Generation time:', elapsed)

# ----------------------------- Phase to Hologram ----------------------------- #
# Convert phase to displayable range
holo = output / (2 * np.pi) + 0.5
holo_image = np.uint8(holo.cpu().numpy() * 255)
cv2.imwrite(f'result/{img_id}_holo_{z}_{pitch}.png', holo_image)

# ----------------------------- Reconstruction ----------------------------- #
# Convert predicted phase to complex representation
real = torch.cos(output)
imag = torch.sin(output)
complex_field = torch.complex(real, imag)

# Propagate forward using ASM
recon_freq = torch.fft.fftn(complex_field)
recon_field = torch.fft.ifftn(recon_freq * H)
recon_amp = torch.abs(recon_field)
recon_amp /= torch.max(recon_amp)  # Normalize

# Save reconstruction image
recon_image = np.uint8(recon_amp.cpu().numpy() * 255)
cv2.imwrite(f'result/{img_id}_recon.png', recon_image)

# ----------------------------- Evaluation ----------------------------- #
# PSNR and SSIM calculation
psnrr = psnr(gray.squeeze(), recon_image)
data_range = np.max([np.max(gray), np.max(recon_image)]) - np.min([np.min(gray), np.min(recon_image)])
ssim_val, _ = ssim(gray.squeeze(), recon_image, full=True, data_range=data_range)

print('PSNR:', psnrr)
print('SSIM:', ssim_val)

# Save evaluation metrics
with open(f'./result/{img_id}.txt', 'w') as f:
    f.write(f'IMG No#{img_id}\n')
    f.write(f'Generation Time: {elapsed}\n')
    f.write(f'PSNR: {psnrr}\n')
    f.write(f'SSIM: {ssim_val}\n')

# Show image (optional)
cv2.imshow('Reconstruction', recon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
