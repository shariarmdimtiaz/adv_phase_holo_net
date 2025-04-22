import torch
import numpy

def H(m=1920, n=1072, z=300, pitch=0.008, wavelength=0.000633):
    x = numpy.linspace(-n//2, n//2-1, n)
    y = numpy.linspace(-m//2, m//2-1, m)
    x=torch.from_numpy(x)
    y=torch.from_numpy(y)
    n=numpy.array(n)
    m=numpy.array(m)
    n = torch.from_numpy(n)
    m = torch.from_numpy(m)

    v = 1 / (n * pitch)
    u = 1 / (m * pitch)
    fx = x * v
    fy = y * u

    fX, fY = torch.meshgrid(fx, fy)

    H = (-1) * (2 * numpy.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX) ** 2 - (wavelength * fY) ** 2)
    return H


def H2(m, n, z, pitch, wavelength):
    x = numpy.linspace(-n // 2, n // 2 - 1, n)
    y = numpy.linspace(-m // 2, m // 2 - 1, m)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    n = numpy.array(n)
    m = numpy.array(m)
    n = torch.from_numpy(n)
    m = torch.from_numpy(m)

    v = 1 / (n * pitch)
    u = 1 / (m * pitch)
    fx = x * v
    fy = y * u

    fX, fY = torch.meshgrid(fx, fy)

    H_2 = (2 * numpy.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX) ** 2 - (wavelength * fY) ** 2)
    return H_2
