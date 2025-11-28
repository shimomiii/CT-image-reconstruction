# Bayesian CT Reconstruction

This repository provides a complete computational framework for CT image reconstruction,  
combining classical filtered backprojection (FBP) techniques with a Bayesian reconstruction
method based on free-energy minimization. The pipeline supports:

- Sinogram generation (C++ implementation)
- Filtered backprojection reconstruction
- Bayesian reconstruction (frequency-domain model)
- Hyperparameter estimation (γ, β, h) via n-section search
- Quantitative evaluation (PSNR, SSIM, profile comparison)

The project is structured for high-resolution numerical experiments, including Shepp–Logan
phantom reconstruction and noise/angle-reduction analysis.

---

## 1. Repository Structure
```
.
├── bayesian_ct_recon.ipynb # Main notebook for experiments and analysis
├── recon.cpp # C++ implementation (Radon transform & backprojection)
├── librecon_cpp.dylib # Compiled shared library for macOS
│
├── sinograms/ # Generated sinogram data (ignored by Git)
│ └── ... (CSV/NPY files)
│
├── results/ # Reconstruction results, figures, logs (ignored by Git)
│ └── ... (images, metrics)
│
└── README.md
```

---

## 2. Features

### Sinogram Generation
- Radon transform implemented in C++
- Supports:
  - Normal noise
  - Poisson noise
  - Delta (noise-free) mode
- Adjustable:
  - Number of projection angles (`N_theta`)
  - Detector sampling (`N_s`)
  - Number of backprojection steps (`nh`)
  - Noise standard deviation (`psd`)
  - Angular reduction factor (`T`) for sparse-view CT

### Reconstruction
- **Filtered Backprojection (FBP)**  
  Reconstruction using standard filters: Ramp, Hann, Hamming, Shepp–Logan.

- **Bayesian Reconstruction**  
  - Frequency-domain model  
  - Free-energy minimization  
  - Hyperparameters (γ, β, h) estimated via n-section search  
  - C++ and Python hybrid pipeline

### Evaluation
- PSNR / SSIM computation
- Reconstruction profiles
- Noise sensitivity analysis
- Angular subsampling (T-variation) performance

---

## 3. Build Instructions (macOS)

Compile the C++ reconstruction library:

```
clang++ -std=c++17 -O3 -dynamiclib -o librecon_cpp.dylib recon.cpp
```

This produces `librecon_cpp.dylib`, which is loaded from Python via `ctypes`.

---

## 4. Python Usage

### Load the shared library

```python
import ctypes

lib = ctypes.CDLL("./librecon_cpp.dylib")

```
Example: calling the backprojection function
```

import numpy as np

ny = nx = 256
out = np.zeros((ny, nx), dtype=np.float64)

lib.reconstruction_cpp(
    ny, nx,
    left, right,
    top, bottom,
    sinogram.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    N_theta, N_s, ds,
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
)

```
More complete examples are available in `bayesian_ct_recon.ipynb`.

---

## 5. Running the Full Pipeline

1. **Generate sinograms**  
   (via the notebook or your Python script)

2. **Choose reconstruction method**
   - FBP with various filters
   - Bayesian reconstruction

3. **Evaluate**
   - PSNR / SSIM
   - Visual comparison
   - Profile analysis

4. **Save results**  
   Output files are stored under `./results/`.

---

## 6. Requirements
```

python >= 3.9
numpy
scipy
matplotlib
tqdm

```
macOS (ARM or Intel) is supported for the C++ dynamic library.

---

## 7. License

Specify your preferred license (e.g., MIT, Apache 2.0).

---

## 8. Acknowledgements

This repository uses:

- Shepp–Logan phantom  
- Custom Radon & backprojection implementation in C++  
- Bayesian CT techniques inspired by Shouno et al. and related literature  

If you use this work in an academic project, please cite appropriately.# CT-image-reconstruction
