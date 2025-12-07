#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

#include <limits>
#include <algorithm>

// ===== 逆投影（Reconstruction）=====
//
// out[ny, nx] ← reconstruction_cpp(ny, nx, left, right, top, bottom, g[nth, ns], nth, ns, ds)
//
// - g: sinogram after filtering, shape (nth, ns), row-major
// - out: reconstructed image, shape (ny, nx), row-major
//

extern "C"
__attribute__((visibility("default")))
void reconstruction_cpp(
    int ny, int nx,
    double left, double right,
    double top, double bottom,
    const double* g,   // (nth, ns)
    int nth, int ns,
    double ds,
    double* out        // (ny, nx)
) {
    const double dth = M_PI / static_cast<double>(nth);
    const double Smin = -((static_cast<double>(ns) - 1.0) / 2.0) * ds;
    const double Smax =  ((static_cast<double>(ns) - 1.0) / 2.0) * ds;
    const double eps  = 1e-12;

    const double dx = (right - left) / static_cast<double>(nx - 1);
    const double dy = (bottom - top) / static_cast<double>(ny - 1);

    std::vector<double> cos_th(nth), sin_th(nth);
    for (int it = 0; it < nth; ++it) {
        double th = static_cast<double>(it) * dth;
        cos_th[it] = std::cos(th);
        sin_th[it] = std::sin(th);
    }

    for (int iy = 0; iy < ny; ++iy) {
        double y = top + static_cast<double>(iy) * dy;
        for (int ix = 0; ix < nx; ++ix) {
            double x = left + static_cast<double>(ix) * dx;
            double ans = 0.0;

            for (int it = 0; it < nth; ++it) {
                double s = x * cos_th[it] + y * sin_th[it];

                if ((Smin <= s) && (s <= Smax - eps)) {
                    double rel = (s - Smin) / ds;
                    int k = static_cast<int>(std::floor(rel));
                    if (k < 0) k = 0;
                    if (k > ns - 2) k = ns - 2;

                    double sk = Smin + static_cast<double>(k) * ds;
                    double q  = (s - sk) / ds;

                    const double g0 = g[static_cast<std::size_t>(it) * ns + static_cast<std::size_t>(k)];
                    const double g1 = g[static_cast<std::size_t>(it) * ns + static_cast<std::size_t>(k + 1)];
                    ans += (1.0 - q) * g0 + q * g1;
                }
            }

            out[static_cast<std::size_t>(iy) * nx + static_cast<std::size_t>(ix)] = ans * dth;
        }
    }
}

// ===== helpers for Radon =====

static double rand_normal01(std::mt19937_64& gen) {
    static thread_local std::normal_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

static int rand_poisson(double lambda, std::mt19937_64& gen) {
    if (lambda <= 0.0) return 0;
    std::poisson_distribution<int> dist(lambda);
    return dist(gen);
}

// ===== helper: in_area =====
static inline bool in_area(
    double x, double y,
    double left, double right,
    double top, double bottom,
    bool periodic
) {
    if (periodic) return true;
    return (x >= left) && (x <= right) && (y >= bottom) && (y <= top);
}

// ===== helper: bilinear_value =====
static double bilinear_value(
    const double* img, int Ny, int Nx,
    double x, double y,
    double left, double right,
    double top, double bottom,
    double sx, double sy,
    bool periodic
) {
    if (periodic) {
        double Lx = right - left;
        double Ly = top - bottom;

        double tx = std::fmod(x - left, Lx);
        if (tx < 0.0) tx += Lx;
        x = left + tx;

        double ty = std::fmod(y - bottom, Ly);
        if (ty < 0.0) ty += Ly;
        y = bottom + ty;
    }

    double ix = (x - left) * sx;
    double iy = (y - bottom) * sy;

    if (ix < 0.0 || ix > static_cast<double>(Nx - 1) ||
        iy < 0.0 || iy > static_cast<double>(Ny - 1)) {
        return 0.0;
    }

    int x0 = static_cast<int>(std::floor(ix));
    int y0 = static_cast<int>(std::floor(iy));
    int x1 = x0 + 1; if (x1 > Nx - 1) x1 = Nx - 1;
    int y1 = y0 + 1; if (y1 > Ny - 1) y1 = Ny - 1;

    double dx = ix - static_cast<double>(x0);
    double dy = iy - static_cast<double>(y0);
    if (x1 == x0) dx = 0.0;
    if (y1 == y0) dy = 0.0;

    const double I00 = img[static_cast<std::size_t>(y0) * Nx + static_cast<std::size_t>(x0)];
    const double I01 = img[static_cast<std::size_t>(y1) * Nx + static_cast<std::size_t>(x0)];
    const double I10 = img[static_cast<std::size_t>(y0) * Nx + static_cast<std::size_t>(x1)];
    const double I11 = img[static_cast<std::size_t>(y1) * Nx + static_cast<std::size_t>(x1)];

    double I0 = I00 * (1.0 - dy) + I01 * dy;
    double I1 = I10 * (1.0 - dy) + I11 * dy;
    return I0 * (1.0 - dx) + I1 * dx;
}

// ===== helper: rfunc (noise) =====
static inline double rfunc(
    double val,
    int noise_mode,       // 0=delta,1=normal,2=poisson
    double psd,
    std::mt19937_64& gen
) {
    if (noise_mode == 0) {
        return val;
    } else if (noise_mode == 1) {
        double z = rand_normal01(gen);
        return val + psd * z;
    } else if (noise_mode == 2) {
        double lam = (val > 0.0) ? val : 0.0;
        int k = rand_poisson(lam, gen);
        return static_cast<double>(k);
    } else {
        return val;
    }
}

// ===== Radon transform (2D, C++実装, Cインターフェース) =====
//
// img: (Ny, Nx) row-major
// Tau: (Nth, Ns) row-major
//
extern "C"
__attribute__((visibility("default")))
void radon_transform_2d_cpp(
    const double* img, int Ny, int Nx,
    int Nth, int Ns,
    double left, double right,
    double top, double bottom,
    double psd,
    int nh,
    double ds,
    int noise_mode,    // 0=delta,1=normal,2=poisson
    int periodic,
    int unitary_like,
    int seed,
    double* Tau       // (Nth, Ns)
) {
    const double dtheta = M_PI / static_cast<double>(Nth);
    const double Lim = ((static_cast<double>(Ns) - 1.0) * ds) * std::sqrt(2.0) / 2.0;
    const double dh  = Lim / static_cast<double>(nh);

    const double sx = static_cast<double>(Nx - 1) / (right - left);
    const double sy = static_cast<double>(Ny - 1) / (top - bottom);

    const double s0 = -((static_cast<double>(Ns) - 1.0) / 2.0) * ds;

    std::mt19937_64 gen;
    if (seed == 0) {
        gen.seed(1234567ULL);
    } else {
        gen.seed(static_cast<std::uint64_t>(seed));
    }

    double theta = 0.0;
    for (int ith = 0; ith < Nth; ++ith) {
        double c = std::cos(theta);
        double s = std::sin(theta);

        for (int js = 0; js < Ns; ++js) {
            double sj = s0 + static_cast<double>(js) * ds;

            double plus = 0.0;
            double t = 0.0;
            while (t <= Lim + 1e-15) {
                double x = sj * c - t * s;
                double y = sj * s + t * c;
                if (!in_area(x, y, left, right, top, bottom, periodic != 0)) break;
                double val = bilinear_value(img, Ny, Nx, x, y, left, right, top, bottom, sx, sy, periodic != 0);
                plus += rfunc(val, noise_mode, psd, gen) * dh;
                t += dh;
            }

            double minus = 0.0;
            t = -dh;
            while (t >= -Lim - 1e-15) {
                double x = sj * c - t * s;
                double y = sj * s + t * c;
                if (!in_area(x, y, left, right, top, bottom, periodic != 0)) break;
                double val = bilinear_value(img, Ny, Nx, x, y, left, right, top, bottom, sx, sy, periodic != 0);
                minus += rfunc(val, noise_mode, psd, gen) * dh;
                t -= dh;
            }

            Tau[static_cast<std::size_t>(ith) * Ns + static_cast<std::size_t>(js)] = plus + minus;
        }

        theta += dtheta;
    }

    if (unitary_like) {
        double scale = std::sqrt((ds * dtheta) / (2.0 * M_PI));
        for (int ith = 0; ith < Nth; ++ith) {
            for (int js = 0; js < Ns; ++js) {
                Tau[static_cast<std::size_t>(ith) * Ns + static_cast<std::size_t>(js)] *= scale;
            }
        }
    }
}

// =========================
// n-section search for (gamma, beta, h)
// =========================

enum ScaleType { SCALE_LINEAR = 0, SCALE_LOG = 1 };

static void edges_and_centers(
    double lo, double hi, int n, ScaleType scale,
    std::vector<double>& edges,
    std::vector<double>& centers
) {
    edges.resize(n + 1);
    centers.resize(n);
    if (scale == SCALE_LOG) {
        if (lo <= 0.0 || hi <= 0.0) {
            throw std::runtime_error("log scale requires positive range");
        }
        double lo_t = std::log10(lo);
        double hi_t = std::log10(hi);
        double step = (hi_t - lo_t) / static_cast<double>(n);
        for (int i = 0; i <= n; ++i) {
            edges[i] = lo_t + step * static_cast<double>(i);
        }
        for (int i = 0; i < n; ++i) {
            centers[i] = 0.5 * (edges[i] + edges[i+1]);
        }
        // 10^t に戻す
        for (int i = 0; i <= n; ++i) edges[i] = std::pow(10.0, edges[i]);
        for (int i = 0; i < n; ++i)   centers[i] = std::pow(10.0, centers[i]);
    } else {
        double step = (hi - lo) / static_cast<double>(n);
        for (int i = 0; i <= n; ++i) {
            edges[i] = lo + step * static_cast<double>(i);
        }
        for (int i = 0; i < n; ++i) {
            centers[i] = 0.5 * (edges[i] + edges[i+1]);
        }
    }
}

// free_energy 
static double free_energy_local_cpp(
    const double* tau_abs2,  // (Nk * Ntheta), row-major: k*Ntheta + j
    int Nk,
    int Ntheta,
    const double* s_abs,     // (Nk)
    const double* s2,        // (Nk)
    double log_const,
    double const_,
    double gamma,
    double beta,
    double h
) {
    double E = 0.0;
    for (int k = 0; k < Nk; ++k) {
        double Fk = (beta * s2[k] + h) * s_abs[k] + gamma;

        double ratio = gamma / Fk;
        if (ratio < 1e-15) ratio = 1e-15;
        else if (ratio > 1.0 - 1e-15) ratio = 1.0 - 1e-15;

        double one_minus = 1.0 - ratio;
        double log_dom   = log_const * gamma * one_minus;
        double denom     = const_    * gamma * one_minus;

        double log_term = std::log(log_dom);

        const double* row = tau_abs2 + static_cast<std::size_t>(k) * Ntheta;
        double sum_tau2 = 0.0;
        for (int j = 0; j < Ntheta; ++j) {
            sum_tau2 += row[j];
        }

        E += -log_term * static_cast<double>(Ntheta)
             + denom * sum_tau2;
    }
    return 0.5 * E;
}

extern "C"
__attribute__((visibility("default")))
void n_section_search_cpp(
    const double* tau_abs2, int Nk, int Ntheta,
    const double* s_abs, const double* s2,
    double log_const, double const_,
    double g_lo_init, double g_hi_init,
    double b_lo_init, double b_hi_init,
    double h_lo_init, double h_hi_init,
    int ng, int nb, int nh,
    int iters,
    double tol_rel,
    double tol_abs,
    double fe_tol,
    int scale_gamma,
    int scale_beta,
    int scale_h,
    double* out_gamma,
    double* out_beta,
    double* out_h,
    double* out_minE
) {
    double g_lo = g_lo_init, g_hi = g_hi_init;
    double b_lo = b_lo_init, b_hi = b_hi_init;
    double h_lo = h_lo_init, h_hi = h_hi_init;

    double best_E = std::numeric_limits<double>::infinity();
    double best_g = 0.0, best_b = 0.0, best_h_ = 0.0;

    std::vector<double> g_edges, g_centers;
    std::vector<double> b_edges, b_centers;
    std::vector<double> h_edges, h_centers;

    ScaleType sg = (scale_gamma != 0) ? SCALE_LOG : SCALE_LINEAR;
    ScaleType sb = (scale_beta  != 0) ? SCALE_LOG : SCALE_LINEAR;
    ScaleType sh = (scale_h     != 0) ? SCALE_LOG : SCALE_LINEAR;

    for (int it = 0; it < iters; ++it) {
        edges_and_centers(g_lo, g_hi, ng, sg, g_edges, g_centers);
        edges_and_centers(b_lo, b_hi, nb, sb, b_edges, b_centers);
        edges_and_centers(h_lo, h_hi, nh, sh, h_edges, h_centers);

        double iter_best_E = std::numeric_limits<double>::infinity();
        int best_i = 0, best_j = 0, best_k = 0;
        double iter_best_g = 0.0, iter_best_b = 0.0, iter_best_h = 0.0;

        for (int ig = 0; ig < ng; ++ig) {
            double gamma = g_centers[ig];
            for (int jb = 0; jb < nb; ++jb) {
                double beta = b_centers[jb];
                for (int kh = 0; kh < nh; ++kh) {
                    double h = h_centers[kh];

                    double E = free_energy_local_cpp(
                        tau_abs2, Nk, Ntheta,
                        s_abs, s2,
                        log_const, const_,
                        gamma, beta, h
                    );
                    if (E < iter_best_E) {
                        iter_best_E = E;
                        iter_best_g = gamma;
                        iter_best_b = beta;
                        iter_best_h = h;
                        best_i = ig; best_j = jb; best_k = kh;
                    }
                }
            }
        }

        if (iter_best_E < best_E) {
            best_E = iter_best_E;
            best_g = iter_best_g;
            best_b = iter_best_b;
            best_h_ = iter_best_h;
        }

        auto rel_width = [](double lo, double hi) {
            double m = std::max({std::fabs(lo), std::fabs(hi), 1e-300});
            return (hi - lo) / m;
        };

        double w_g = g_hi - g_lo;
        double w_b = b_hi - b_lo;
        double w_h = h_hi - h_lo;

        bool stop_abs = (tol_abs > 0.0) &&
            (w_g <= tol_abs && w_b <= tol_abs && w_h <= tol_abs);
        bool stop_rel = (tol_rel > 0.0) &&
            (rel_width(g_lo, g_hi) <= tol_rel &&
             rel_width(b_lo, b_hi) <= tol_rel &&
             rel_width(h_lo, h_hi) <= tol_rel);
        bool stop_fe = (fe_tol > 0.0) &&
            (iter_best_E - best_E > -fe_tol);  

        if (stop_abs || stop_rel || stop_fe) {
            break;
        }

        g_lo = g_edges[best_i];
        g_hi = g_edges[best_i + 1];
        b_lo = b_edges[best_j];
        b_hi = b_edges[best_j + 1];
        h_lo = h_edges[best_k];
        h_hi = h_edges[best_k + 1];
    }

    *out_gamma = best_g;
    *out_beta  = best_b;
    *out_h     = best_h_;
    *out_minE  = best_E;
}


// clang++ -std=c++17 -O3 -dynamiclib -o librecon_cpp.dylib recon.cpp