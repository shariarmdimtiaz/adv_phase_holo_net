import io
from torch import nn
import cv2
import numpy
import torch.fft
from tqdm import trange
import time
import csv
import scipy
from propagation_ASM import *  # ASM-based propagation method
from datetime import date
from model import AdvancedPhaseEstimationNet  # Custom complex-valued network

# ---------------------------- Constants & Parameters ---------------------------- #
z = 200  # Propagation distance in mm
num = 100  # Number of training epochs
rangege = 720  # Number of training images
rangegege = 180  # Number of validation images
pitch = 0.0036  # Pixel pitch (mm)
wavelength = 0.000532  # Wavelength (mm) - Green laser
n = 1072  # Height
m = 1920  # Width
pad = True  # Padding flag (unused here)

# ---------------------------- Frequency Domain Mesh Grid ---------------------------- #
x = numpy.linspace(-n//2, n//2 - 1, n)
y = numpy.linspace(-m//2, m//2 - 1, m)
x = torch.from_numpy(x)
y = torch.from_numpy(y)

# Frequency steps
v = 1 / (n * pitch)
u = 1 / (m * pitch)
fx = x * v
fy = y * u

# Create 2D meshgrid for spatial frequencies
fX, fY = torch.meshgrid(fx, fy, indexing='ij')

# ---------------------------- Transfer Functions (H, H2) ---------------------------- #
# Angular spectrum method (ASM) transfer function for forward and backward propagation
H = (-1) * (2 * numpy.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX) ** 2 - (wavelength * fY) ** 2)
H2 = (2 * numpy.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX) ** 2 - (wavelength * fY) ** 2)

# Convert phase to real + imaginary components
Hreal = torch.cos(H)
Himage = torch.sin(H)
H2real = torch.cos(H2)
H2image = torch.sin(H2)

# Band-limiting based on sampling condition
xlimit = 1 / torch.sqrt((2 * 1 / m / pitch * z) ** 2 + 1) / wavelength
ylimit = 1 / torch.sqrt((2 * 1 / n / pitch * z) ** 2 + 1) / wavelength
a = (abs(fX) < xlimit) & (abs(fY) < ylimit)
a = a.numpy().astype(int)

# Apply band-limit mask to transfer functions
Hreal *= a
Himage *= a
H = torch.complex(Hreal, Himage)

H2real *= a
H2image *= a
H2 = torch.complex(H2real, H2image)

H = H.cuda()
H2 = H2.cuda()

# ---------------------------- Training Setup ---------------------------- #
lr = 0.0001
epoch = num
today = date.today().strftime('%Y-%m-%d')
model_name = f'{today}_weight_E{epoch}_{z}_{wavelength}'

model = AdvancedPhaseEstimationNet()
criterion = nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

optvars = [{'params': model.parameters()}]
optimizer = torch.optim.Adam(optvars, lr=lr)

# Dataset paths
trainpath = f'./DIV2K/DIV2K_train_HR/'
validpath = f'./DIV2K/DIV2K_valid_HR/'

# ---------------------------- Training & Validation Loop ---------------------------- #
train_losses = []
valid_losses = []

for k in trange(num):  # Epoch loop
    current_train_loss = 0
    current_valid_loss = 0

    # ---------------- Training ---------------- #
    for kk in range(rangege):
        c = 1 + kk
        b = str(c).zfill(4)  # Zero-pad index
        imgpath = trainpath + b + '.png'

        img = cv2.imread(imgpath)
        img_resized = cv2.resize(img, (1920, 1072))
        gray = cv2.split(img_resized)[2]  # Use red channel
        gray = numpy.reshape(gray, (1, 1, 1072, 1920))

        # Normalize and convert to tensor
        target_amp = torch.from_numpy(gray).cuda() / 255.0
        target_amp_complex = torch.complex(target_amp, torch.zeros_like(target_amp)).cuda()
        target_amp = target_amp.double().squeeze()

        # Predict phase
        output = model(target_amp_complex, H2)

        # Reconstruct complex field
        grayreal = torch.cos(output)
        grayimage = torch.sin(output)
        gray_complex = torch.complex(grayreal, grayimage)

        # Propagation (forward) and loss calculation
        propagated = torch.fft.fftn(gray_complex) * H
        final = torch.fft.ifftn(propagated)
        final_amp = torch.abs(final)
        loss = criterion(final_amp, target_amp)

        # Backpropagation
        current_train_loss += loss.cpu().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(current_train_loss / rangege)
    print('\nTrain Loss:', train_losses[-1])

    # Save intermediate reconstruction
    final_normalized = final_amp / torch.max(final_amp)
    recon_image = numpy.uint8(final_normalized.cpu().numpy() * 255)
    cv2.imwrite(f'./model_output/recon/1{k}.png', recon_image)

    # ---------------- Validation ---------------- #
    with torch.no_grad():
        for kk in range(rangegege):
            c = 721 + kk
            b = '0' + str(c)
            imgpath = validpath + b + '.png'
            img = cv2.imread(imgpath)
            img_resized = cv2.resize(img, (1920, 1072))
            gray = cv2.split(img_resized)[2]
            gray = numpy.reshape(gray, (1, 1, 1072, 1920))

            target_amp = torch.from_numpy(gray).cuda() / 255.0
            target_amp_complex = torch.complex(target_amp, torch.zeros_like(target_amp)).cuda()
            target_amp = target_amp.double().squeeze()

            # Predict and reconstruct
            output = model(target_amp_complex, H2)
            grayreal = torch.cos(output)
            grayimage = torch.sin(output)
            gray_complex = torch.complex(grayreal, grayimage)
            propagated = torch.fft.fftn(gray_complex) * H
            final = torch.fft.ifftn(propagated)
            final_amp = torch.abs(final)
            loss = criterion(final_amp, target_amp)
            current_valid_loss += loss.cpu().item()

            # Save selected validation result
            if kk == 38:
                final_normalized = final_amp / torch.max(final_amp)
                recon_image = numpy.uint8(final_normalized.cpu().numpy() * 255)
                cv2.imwrite(f'./model_output/recon/2{k}.png', recon_image)

        valid_losses.append(current_valid_loss / rangegege)
        print('Validation Loss:', valid_losses[-1])
    time.sleep(1)

# ---------------------------- Save Results ---------------------------- #
# Save training and validation loss history
with open(f'./model_output/{model_name}_loss.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
    for epoch, train_loss, val_loss in zip(range(num), train_losses, valid_losses):
        writer.writerow([epoch, train_loss, val_loss])

# Save model weights
torch.save(model.state_dict(), f'./model_output/{model_name}.pth')

# Optional: save loss curves in MATLAB format
# scipy.io.savemat(f'{model_name}_avgloss.mat', {'avgloss': numpy.mat(valid_losses)})
# scipy.io.savemat(f'{model_name}_avgtloss.mat', {'avgtloss': numpy.mat(train_losses)})
