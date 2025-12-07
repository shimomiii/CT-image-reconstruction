# Bayesian CT Reconstruction Framework

This repository provides a computational framework for CT image reconstruction,  
combining classical Filtered Backprojection (FBP) with a Bayesian reconstruction method  
based on a Markov Random Field (MRF) prior model.  
A hybrid workflow of C++ (Radon transform, parameter estimation, reconstruction)  
and Python (numerical experiments and evaluation) enables efficient reconstruction  
even for high-resolution images.

---

## 1. Theoretical Background (Bayesian Reconstruction and Free Energy)

The Bayesian CT reconstruction method used in this project integrates  
a Gaussian observation model with an MRF-based image prior.  
This formulation yields a closed-form MAP estimator in the frequency domain.

---

### 1.1 Observation Model (Frequency Domain)

Let σ(x,y) be the original image, and τ(s,θ) its Radon transform.  
Assuming additive Gaussian noise, the observation model in frequency space is

$$ \tilde{\tau}(\tilde{s},\theta) = \tilde{\sigma}(\tilde{s},\theta) + \tilde{n}(\tilde{s},\theta). $$

The corresponding observation energy is

$$
H_{\mathrm{obs}} = 4\pi^2 \gamma \int d\theta \int d\tilde{s}\,
 |\tilde{\tau}-\tilde{\sigma}|^2,
$$

where γ is proportional to the inverse of the noise variance.

---

### 1.2 Prior Distribution (MRF Model)

To impose smoothness and amplitude regularization,  
the following prior energy is used:

$$
H_{\mathrm{pri}}(\sigma) = \beta \iint |\nabla\sigma|^2 \, dxdy + 4\pi h \iint |\sigma|^2 \, dxdy .
$$

In Fourier space,

$$
H_{\mathrm{pri}}
 = 4\pi^2 \iint (\beta(\tilde{x}^2+\tilde{y}^2)+h)\,
 |\tilde{\sigma}(\tilde{x},\tilde{y})|^2 \, d\tilde{x}d\tilde{y},
$$

and using polar coordinates $(\tilde{x},\tilde{y})=(\tilde{s}\cos\theta, \tilde{s}\sin\theta)$,

$$
H_{\mathrm{pri}}
 = 4\pi^2 \int d\theta \int d\tilde{s}\,
 (\beta\tilde{s}^2+h)\,|\tilde{s}|\,|\tilde{\sigma}(\tilde{s},\theta)|^2.
$$

The prior distribution is therefore

$$
p(\sigma\mid\beta,h)
 \propto \exp\!\left[
 -4\pi^2\!\iint(\beta\tilde{s}^2+h)\,|\tilde{s}|\,|\tilde{\sigma}|^2
 \right].
$$

---

### 1.3 MAP Estimator and Bayesian Filter (Closed-form Solution)

The posterior distribution is

$$
p(\sigma\mid\tau,\gamma,\beta,h)
 \propto \exp\!\left[
 -4\pi^2\!\int\!\!\int
 \left(
   \gamma|\tilde{\tau}-\tilde{\sigma}|^2
   +(\beta\tilde{s}^2+h)|\tilde{s}|\,|\tilde{\sigma}|^2
 \right)
 \right].
$$

Define

$$
F_{\tilde{s}}
 = (\beta\tilde{s}^2+h)|\tilde{s}|+\gamma.
$$

Then the MAP estimate has the closed-form solution

$$
\hat{\sigma}(\tilde{s},\theta)
 = \frac{\gamma}{F_{\tilde{s}}} \, \tilde{\tau}(\tilde{s},\theta).
$$

Here, γ/F_{\tilde{s}} acts as a **Bayesian frequency-domain filter**,  
which generalizes the classical FBP filters.

---

### 1.4 Hyperparameter Estimation (Free Energy Minimization)

The hyperparameters (γ, β, h) are obtained by minimizing the free energy

$$
\mathcal{F}(\gamma,\beta,h)
 = -\log\,p(\tau\mid\gamma,\beta,h),
$$

where the marginal likelihood is

$$
p(\tau\mid\gamma,\beta,h)
 = \int p(\tau\mid\sigma,\gamma)\,p(\sigma\mid\beta,h)\,d\sigma .
$$

After discretization, the free energy becomes

$$
F(\gamma,\beta,h) = -\frac{1}{2} \sum_{\tilde{k},l} \left[ \log\!\left( \frac{8\pi\Delta_\theta\Delta_s}{N_s}\, \gamma\left(1-\frac{\gamma}{F_{\tilde{s}}}\right) \right) - \frac{8\pi^2\Delta_\theta\Delta_s}{N_s}\, \gamma\left(1-\frac{\gamma}{F_{\tilde{s}}}\right) \tilde{\tau}_{\tilde{k},l}|^2 \right].
$$

This free energy is minimized using a **multi-dimensional grid search**,  
providing a consistent Bayesian estimation pipeline.

---

## 2. Repository Structure

```
.
├── bayesian_ct_recon.ipynb      # Main experiment notebook  
├── recon.cpp                     # C++ Radon transform / backprojection  
├── librecon_cpp.dylib            # Shared library (macOS)  
├── sinograms/                    # Generated sinograms  
├── results/                      # Reconstruction results  
└── README.md
```

---

## 3. Features

### 3.1 Sinogram Generation
- High-speed Radon transform in C++  
- Noise models: normal, poisson, delta  
- Adjustable parameters: psd, Ns, Nθ, T (sparse-view CT)

### 3.2 Reconstruction
**FBP Filters**
- Ramp  
- Shepp–Logan  
- Hann  
- Hamming  

**Bayesian Reconstruction**
- Closed-form MAP estimator  
- Hyperparameter estimation via free-energy minimization  
- Hybrid Python + C++ implementation

### 3.3 Evaluation Metrics
- PSNR, SSIM, RMSE  
- Visual evaluation  
- Profile comparison  

---

## 4. Building the C++ Library (macOS)

```
clang++ -std=c++17 -O3 -dynamiclib -o librecon_cpp.dylib recon.cpp
```

---

## 5. Using the Library from Python

```
import ctypes
lib = ctypes.CDLL("./librecon_cpp.dylib")
```

Example: calling the backprojection function

```
out = np.zeros((ny, nx), dtype=np.float64)
lib.reconstruction_cpp(
    ny, nx,
    left, right, top, bottom,
    sino.ctypes.data_as(...),
    Nθ, Ns, ds,
    out.ctypes.data_as(...)
)
```

---

## 6. Numerical Experiment Pipeline

```
records = numerical_experiment(
    psd_range=(0.5,4.0,0.5),
    N_sizes=256,
    N_theta=256,
    T_range=(1.0,10.5,1.0),
    filters=("bayes","ramp","shepp-logan","hann","hamming"),
    phantom=phantom,
    radon_transform_fn=radon_transform_cxx,
    free_energy_fn=free_energy,
    n_section_search_fn=n_section_search_cxx,
    image_reconstruction_fn=image_reconstruction,
)
```

### Runtime Example
- 256×256, 180 projections: **~0.7 seconds per reconstruction**  
- 2048×2048, 1800 projections: **~16 seconds per reconstruction**

---

## 7. Requirements

```
python >= 3.9  
numpy  
scipy  
matplotlib  
pandas  
tqdm  
scikit-image
```

---

## 8. References

1. A. C. Kak and M. Slaney,  
   *Principles of Computerized Tomographic Imaging*,  
   IEEE Press, 1988.

2. G. N. Ramachandran and A. V. Lakshminarayanan,  
   “Three-dimensional reconstruction from radiographs and electron micrographs:  
   Application of convolutions instead of Fourier transforms,”  
   *PNAS*, vol. 68, no. 9, pp. 2236–2240, 1971.

3. H. Shouno and M. Okada,  
   “Bayesian Image Restoration for Medical Images Using Radon Transform,”  
   *Journal of the Physical Society of Japan*, vol. 79, no. 7, p. 074004, 2010.

