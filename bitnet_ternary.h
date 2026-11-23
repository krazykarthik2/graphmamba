// bitnet_ternary.h
// Pure C++ implementation of BitNet b1.58 Ternary Math & Unpacking
// Directly mapped from ARCHITECTURE_AND_MATH.md specifications:
// - Discrete Ternary Weights: {-1, 0, +1}
// - AbsMean Gamma Scaling factor
// - 1.58-bit 4-weights-per-byte uint8 Bit-Packing / Unpacking
// - Dirichlet Graph Laplacian Spectral Inversion (W* = C_yx (C_xx + lambda*I)^-1)
// - Fast BitNet GEMM (ternary dot product using addition/subtraction without multiplications)

#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace bitnet {

// 1. Hard ternary quantization threshold:
// Quantize(z) = +1 (z > 0.5), 0 (-0.5 <= z <= 0.5), -1 (z < -0.5)
inline int8_t hard_ternary_scalar(float z, float threshold = 0.5f) {
    if (z > threshold) return 1;
    if (z < -threshold) return -1;
    return 0;
}

// 2. AbsMean Gamma Scaling: gamma = (1 / (d_out * d_in)) * sum(|W_ij|)
inline float compute_absmean_gamma(const std::vector<float>& w) {
    if (w.empty()) return 1.0f;
    double sum = 0.0;
    for (float v : w) {
        sum += std::abs(v);
    }
    float gamma = static_cast<float>(sum / w.size());
    return std::max(gamma, 1e-5f);
}

// 3. Pack ternary weights {-1, 0, +1} into 2-bit unsigned integers (4 weights / uint8 byte)
// Mapping: {-1 -> 0, 0 -> 1, +1 -> 2}
inline std::vector<uint8_t> pack_ternary(const std::vector<int8_t>& ternary_weights) {
    size_t n = ternary_weights.size();
    size_t packed_size = (n + 3) / 4;
    std::vector<uint8_t> packed(packed_size, 0);

    for (size_t i = 0; i < n; ++i) {
        // Map {-1, 0, 1} to {0, 1, 2}
        uint8_t code = static_cast<uint8_t>(ternary_weights[i] + 1) & 0x03;
        size_t byte_idx = i / 4;
        size_t bit_shift = (i % 4) * 2;
        packed[byte_idx] |= (code << bit_shift);
    }
    // Pad elements default to 0 in unused bits (or 1 depending on spec, here unmapped positions stay zeroed)
    return packed;
}

// 4. Unpack uint8 stream back to float32 {-1.0, 0.0, +1.0}
inline std::vector<float> unpack_ternary(const std::vector<uint8_t>& packed, size_t original_count) {
    std::vector<float> unpacked(original_count);
    for (size_t i = 0; i < original_count; ++i) {
        size_t byte_idx = i / 4;
        size_t bit_shift = (i % 4) * 2;
        uint8_t code = (packed[byte_idx] >> bit_shift) & 0x03;
        unpacked[i] = static_cast<float>(static_cast<int>(code) - 1);
    }
    return unpacked;
}

// 5. Memory-efficient low-RAM Ternary Linear Forward:
// y = x * (W_ternary * gamma)^T
// Computes matrix multiplication directly from packed 2-bit weights without full float tensor allocation!
// Out[j] = gamma * Sum_k(x[k] * W[j, k])
inline void ternary_gemv_packed(
    const float* __restrict x,
    const uint8_t* __restrict packed_w,
    float gamma,
    int in_features,
    int out_features,
    float* __restrict out
) {
    for (int j = 0; j < out_features; ++j) {
        double acc = 0.0;
        size_t row_start_idx = static_cast<size_t>(j) * in_features;

        for (int k = 0; k < in_features; ++k) {
            size_t weight_idx = row_start_idx + k;
            size_t byte_idx = weight_idx / 4;
            size_t bit_shift = (weight_idx % 4) * 2;
            int code = (packed_w[byte_idx] >> bit_shift) & 0x03;
            int w_val = code - 1; // {-1, 0, +1}

            // Zero multiplication: Ternary addition / subtraction!
            if (w_val == 1) {
                acc += x[k];
            } else if (w_val == -1) {
                acc -= x[k];
            }
        }
        out[j] = static_cast<float>(acc * gamma);
    }
}

// 6. Dirichlet Graph Laplacian Spectral Solver in C++:
// W* = C_yx * (C_xx + lambda * I)^(-1)
// Inverts positive semi-definite (C_xx + lambda*I) using Cholesky Decomposition (O(d^3), low RAM)
inline bool cholesky_decompose(int d, const std::vector<float>& A, std::vector<float>& L) {
    L.assign(d * d, 0.0f);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum += L[i * d + k] * L[j * d + k];
            }
            if (i == j) {
                float val = A[i * d + i] - sum;
                if (val <= 0.0f) return false;
                L[i * d + j] = std::sqrt(val);
            } else {
                L[i * d + j] = (A[i * d + j] - sum) / L[j * d + j];
            }
        }
    }
    return true;
}

// Solves A * X = B for X where A = L * L^T (Cholesky solve)
inline void cholesky_solve_matrix(int d, const std::vector<float>& L, const std::vector<float>& B, std::vector<float>& X) {
    X.assign(d * d, 0.0f);
    std::vector<float> Y(d, 0.0f);

    for (int col = 0; col < d; ++col) {
        // Forward substitution: L * Y = B[:, col]
        for (int i = 0; i < d; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < i; ++k) {
                sum += L[i * d + k] * Y[k];
            }
            Y[i] = (B[i * d + col] - sum) / L[i * d + i];
        }
        // Backward substitution: L^T * X[:, col] = Y
        for (int i = d - 1; i >= 0; --i) {
            float sum = 0.0f;
            for (int k = i + 1; k < d; ++k) {
                sum += L[k * d + i] * X[i * d + col];
            }
            X[i * d + col] = (Y[i] - sum) / L[i * d + i];
        }
    }
}

// Closed-form Harmonic Ternary Weight Solver
inline void solve_harmonic_ternary(
    int d,
    const std::vector<float>& cov_xx,
    const std::vector<float>& cov_yx,
    uint64_t total_tokens,
    float ridge,
    std::vector<int8_t>& out_ternary,
    float& out_gamma
) {
    // 1. Regularized A = (C_xx / N) + ridge * I
    std::vector<float> A(d * d, 0.0f);
    float inv_n = 1.0f / static_cast<float>(total_tokens);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            A[i * d + j] = cov_xx[i * d + j] * inv_n;
        }
        A[i * d + i] += ridge;
    }

    // 2. Cholesky Decomposition of A
    std::vector<float> L;
    if (!cholesky_decompose(d, A, L)) {
        throw std::runtime_error("Cholesky decomposition failed: Covariance matrix not strictly positive definite");
    }

    // 3. Compute A_inv by solving A * A_inv = I
    std::vector<float> I(d * d, 0.0f);
    for (int i = 0; i < d; ++i) I[i * d + i] = 1.0f;
    std::vector<float> A_inv(d * d, 0.0f);
    cholesky_solve_matrix(d, L, I, A_inv);

    // 4. W* = (C_yx / N) * A_inv
    std::vector<float> W_opt(d * d, 0.0f);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            double sum = 0.0;
            for (int k = 0; k < d; ++k) {
                sum += (cov_yx[i * d + k] * inv_n) * A_inv[k * d + j];
            }
            W_opt[i * d + j] = static_cast<float>(sum);
        }
    }

    // 5. Compute Gamma = mean(|W*|)
    out_gamma = compute_absmean_gamma(W_opt);

    // 6. Ternary projection: Round(Clip(W* / gamma, -1, 1))
    out_ternary.resize(d * d);
    for (size_t i = 0; i < W_opt.size(); ++i) {
        float scaled = W_opt[i] / out_gamma;
        out_ternary[i] = hard_ternary_scalar(scaled, 0.5f);
    }
}

} // namespace bitnet
