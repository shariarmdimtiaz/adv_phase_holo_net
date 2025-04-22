import torch
from torch import nn
import torch.fft
import torch.nn.functional as func

# Importing complex-valued operations and layers
from complexPyTorch.complexLayers import ComplexConvTranspose2d, ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu, complex_elu, complex_sigmoid
# from complexPyTorch.complexLayers import ComplexBatchNorm2d, complex_max_pool2d
# from complexPyTorch.complexFunctions import complex_upsample2, complex_upsample


class ComplexDownBlock(nn.Module):
    """
    Downsampling block using complex-valued convolution with stride=2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return complex_relu(self.down_conv(x))


class ComplexUpBlockRelu(nn.Module):
    """
    Upsampling block using complex-valued transposed convolution followed by ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            ComplexConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return complex_relu(self.up_conv(x))


class ComplexUpBlockElu(nn.Module):
    """
    Final upsampling block using complex-valued transposed convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            ComplexConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return complex_elu(self.up_conv(x))


class PhaseInitNetwork(nn.Module):
    """
    First sub-network to initialize the complex field using phase estimation
    """
    def __init__(self):
        super().__init__()
        self.encoder1 = ComplexDownBlock(1, 8)
        self.encoder2 = ComplexDownBlock(8, 16)
        self.encoder3 = ComplexDownBlock(16, 32)
        self.encoder4 = ComplexDownBlock(32, 64)

        self.decoder3 = ComplexUpBlockRelu(64, 32)
        self.decoder2 = ComplexUpBlockRelu(32, 16)
        self.decoder1 = ComplexUpBlockRelu(16, 8)
        self.final_up = ComplexUpBlockElu(8, 1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        # x4 = self.encoder4(x3)  # Unused in current flow

        # Decoding path (note: decoder3 is skipped)
        x6 = self.decoder2(x3)     # Was decoder3(x4)
        x7 = self.decoder1(x2)
        x_out = self.final_up(x7 + x1)

        # Phase-based reconstruction of the initial complex field
        phase = torch.atan2(x_out.real, x_out.imag)
        init_complex = torch.complex(x.real * torch.cos(phase), x.real * torch.sin(phase))

        return init_complex


class PhaseRefinementNetwork(nn.Module):
    """
    Second sub-network to refine phase of complex field (U-Net like structure)
    """
    def __init__(self):
        super().__init__()
        self.encoder1 = ComplexDownBlock(1, 8)
        self.encoder2 = ComplexDownBlock(8, 16)
        self.encoder3 = ComplexDownBlock(16, 32)
        self.encoder4 = ComplexDownBlock(32, 64)

        self.decoder4 = ComplexUpBlockRelu(64, 32)
        self.decoder3 = ComplexUpBlockRelu(32, 16)
        self.decoder2 = ComplexUpBlockRelu(16, 8)
        self.final_up = ComplexUpBlockElu(8, 1)

    def forward(self, x):
        # Encoding path
        d1 = self.encoder1(x)
        d2 = self.encoder2(d1)
        d3 = self.encoder3(d2)
        d4 = self.encoder4(d3)

        # Decoding with skip connections
        u4 = self.decoder4(d4)
        u3 = self.decoder3(u4 + d3)
        u2 = self.decoder2(u3 + d2)
        u1 = self.final_up(u2 + d1)

        # Final phase extraction
        output = torch.squeeze(u1)
        refined_phase = torch.atan2(output.real, output.imag)

        return refined_phase


class AdvancedPhaseEstimationNet(nn.Module):
    """
    Proposed network combining both the initialization and refinement networks
    with Fourier domain modulation
    """
    def __init__(self):
        super().__init__()
        self.initializer = PhaseInitNetwork()
        self.refiner = PhaseRefinementNetwork()

    def forward(self, input_field, H2):
        # Step 1: Estimate initial complex field
        init_complex = self.initializer(input_field)

        # Step 2: Apply modulation in Fourier domain
        frequency_repr = torch.fft.fftn(init_complex)
        modulated_freq = frequency_repr * H2
        slm_output = torch.fft.ifftn(modulated_freq)

        # Convert to proper complex tensor
        slm_output = torch.complex(slm_output.real.float(), slm_output.imag.float())

        # Step 3: Refine phase from modulated field
        final_output = self.refiner(slm_output)
        return final_output
