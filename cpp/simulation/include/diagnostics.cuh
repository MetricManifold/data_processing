#pragma once

/**
 * @file diagnostics.cuh
 * @brief GPU-side diagnostic measurement system for cell simulations
 * 
 * Computes physical observables:
 * - Energy components (gradient, bulk, interaction)
 * - Stress tensor (σ_xx, σ_yy, σ_xy)
 * - Pressure (P = -½ tr(σ))
 * - Coordination number (contacts per cell)
 * 
 * Enable via CMake: -DENABLE_DIAGNOSTICS=ON
 * or define DIAGNOSTICS_ENABLED before including
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace cellsim {

//=============================================================================
// DIAGNOSTICS_ENABLED is set by CMake or manually defined
//=============================================================================

#ifdef DIAGNOSTICS_ENABLED

/**
 * @brief GPU buffers for diagnostic accumulation
 */
struct DiagnosticBuffers {
    // Energy accumulators (per cell)
    float* d_E_gradient;      // [num_cells]
    float* d_E_bulk;          // [num_cells]
    float* d_E_interaction;   // [num_cells]
    
    // Stress tensor (global accumulators)
    float* d_sigma_xx;        // [1]
    float* d_sigma_yy;        // [1]
    float* d_sigma_xy;        // [1]
    float* d_sigma_isotropic; // [1]
    
    // Coordination (per cell)
    int* d_contacts;          // [num_cells]
    
    int num_cells;
    bool allocated;
    
    DiagnosticBuffers() : 
        d_E_gradient(nullptr), d_E_bulk(nullptr), d_E_interaction(nullptr),
        d_sigma_xx(nullptr), d_sigma_yy(nullptr), d_sigma_xy(nullptr),
        d_sigma_isotropic(nullptr), d_contacts(nullptr),
        num_cells(0), allocated(false) {}
};

/**
 * @brief Host-side output structure for a single diagnostic sample
 */
struct DiagnosticSample {
    float time;
    int step;
    
    // Energy components
    float E_gradient;
    float E_bulk;
    float E_interaction;
    float E_total;
    
    // Stress tensor
    float sigma_xx;
    float sigma_yy;
    float sigma_xy;
    float pressure;
    
    // Coordination statistics
    float z_mean;
    float z_min;
    float z_max;
    float z_std;
    
    DiagnosticSample() : time(0), step(0), 
        E_gradient(0), E_bulk(0), E_interaction(0), E_total(0),
        sigma_xx(0), sigma_yy(0), sigma_xy(0), pressure(0),
        z_mean(0), z_min(0), z_max(0), z_std(0) {}
};

//=============================================================================
// Allocation and management
//=============================================================================

inline cudaError_t diagnostics_allocate(DiagnosticBuffers& buffers, int num_cells) {
    if (buffers.allocated) return cudaSuccess;
    
    buffers.num_cells = num_cells;
    cudaError_t err;
    
    err = cudaMalloc(&buffers.d_E_gradient, num_cells * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buffers.d_E_bulk, num_cells * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buffers.d_E_interaction, num_cells * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buffers.d_sigma_xx, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buffers.d_sigma_yy, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buffers.d_sigma_xy, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buffers.d_sigma_isotropic, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buffers.d_contacts, num_cells * sizeof(int));
    if (err != cudaSuccess) return err;
    
    buffers.allocated = true;
    printf("[DIAG] Allocated buffers for %d cells\n", num_cells);
    
    return cudaSuccess;
}

inline void diagnostics_free(DiagnosticBuffers& buffers) {
    if (!buffers.allocated) return;
    
    cudaFree(buffers.d_E_gradient);
    cudaFree(buffers.d_E_bulk);
    cudaFree(buffers.d_E_interaction);
    cudaFree(buffers.d_sigma_xx);
    cudaFree(buffers.d_sigma_yy);
    cudaFree(buffers.d_sigma_xy);
    cudaFree(buffers.d_sigma_isotropic);
    cudaFree(buffers.d_contacts);
    
    buffers.allocated = false;
    buffers.num_cells = 0;
}

inline cudaError_t diagnostics_reset(DiagnosticBuffers& buffers) {
    if (!buffers.allocated) return cudaErrorNotReady;
    
    cudaMemset(buffers.d_E_gradient, 0, buffers.num_cells * sizeof(float));
    cudaMemset(buffers.d_E_bulk, 0, buffers.num_cells * sizeof(float));
    cudaMemset(buffers.d_E_interaction, 0, buffers.num_cells * sizeof(float));
    cudaMemset(buffers.d_sigma_xx, 0, sizeof(float));
    cudaMemset(buffers.d_sigma_yy, 0, sizeof(float));
    cudaMemset(buffers.d_sigma_xy, 0, sizeof(float));
    cudaMemset(buffers.d_sigma_isotropic, 0, sizeof(float));
    cudaMemset(buffers.d_contacts, 0, buffers.num_cells * sizeof(int));
    
    return cudaGetLastError();
}

//=============================================================================
// Collection and output
//=============================================================================

inline cudaError_t diagnostics_collect(
    const DiagnosticBuffers& buffers,
    DiagnosticSample& sample,
    float time, int step
) {
    sample.time = time;
    sample.step = step;
    
    int num_cells = buffers.num_cells;
    
    // Allocate host buffers
    float* h_E_grad = (float*)malloc(num_cells * sizeof(float));
    float* h_E_bulk = (float*)malloc(num_cells * sizeof(float));
    float* h_E_int = (float*)malloc(num_cells * sizeof(float));
    int* h_contacts = (int*)malloc(num_cells * sizeof(int));
    
    cudaMemcpy(h_E_grad, buffers.d_E_gradient, 
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E_bulk, buffers.d_E_bulk,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E_int, buffers.d_E_interaction,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    
    sample.E_gradient = 0.0f;
    sample.E_bulk = 0.0f;
    sample.E_interaction = 0.0f;
    
    for (int i = 0; i < num_cells; ++i) {
        sample.E_gradient += h_E_grad[i];
        sample.E_bulk += h_E_bulk[i];
        sample.E_interaction += h_E_int[i];
    }
    sample.E_total = sample.E_gradient + sample.E_bulk + sample.E_interaction;
    
    // Copy stress tensor
    cudaMemcpy(&sample.sigma_xx, buffers.d_sigma_xx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sample.sigma_yy, buffers.d_sigma_yy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sample.sigma_xy, buffers.d_sigma_xy, sizeof(float), cudaMemcpyDeviceToHost);
    
    float sigma_isotropic;
    cudaMemcpy(&sigma_isotropic, buffers.d_sigma_isotropic, sizeof(float), cudaMemcpyDeviceToHost);
    
    sample.sigma_xx += sigma_isotropic;
    sample.sigma_yy += sigma_isotropic;
    sample.pressure = -0.5f * (sample.sigma_xx + sample.sigma_yy);
    
    // Coordination statistics
    cudaMemcpy(h_contacts, buffers.d_contacts,
               num_cells * sizeof(int), cudaMemcpyDeviceToHost);
    
    float sum_z = 0.0f, sum_z2 = 0.0f;
    sample.z_min = static_cast<float>(h_contacts[0]);
    sample.z_max = static_cast<float>(h_contacts[0]);
    
    for (int i = 0; i < num_cells; ++i) {
        float z = static_cast<float>(h_contacts[i]);
        sum_z += z;
        sum_z2 += z * z;
        if (z < sample.z_min) sample.z_min = z;
        if (z > sample.z_max) sample.z_max = z;
    }
    
    sample.z_mean = sum_z / num_cells;
    float var = sum_z2 / num_cells - sample.z_mean * sample.z_mean;
    sample.z_std = sqrtf(var > 0 ? var : 0);
    
    // Free host buffers
    free(h_E_grad);
    free(h_E_bulk);
    free(h_E_int);
    free(h_contacts);
    
    return cudaGetLastError();
}

inline void diagnostics_write_header(FILE* file) {
    fprintf(file, "# time,step,E_grad,E_bulk,E_int,E_total,"
                  "sigma_xx,sigma_yy,sigma_xy,pressure,"
                  "z_mean,z_min,z_max,z_std\n");
    fflush(file);
}

inline void diagnostics_write(FILE* file, const DiagnosticSample& sample) {
    fprintf(file, "%.6f,%d,%.6e,%.6e,%.6e,%.6e,"
                  "%.6e,%.6e,%.6e,%.6e,"
                  "%.4f,%.0f,%.0f,%.4f\n",
            sample.time, sample.step,
            sample.E_gradient, sample.E_bulk, sample.E_interaction, sample.E_total,
            sample.sigma_xx, sample.sigma_yy, sample.sigma_xy, sample.pressure,
            sample.z_mean, sample.z_min, sample.z_max, sample.z_std);
    fflush(file);
}

#else // DIAGNOSTICS not enabled - stubs

struct DiagnosticBuffers { 
    bool allocated = false; 
    int num_cells = 0;
};
struct DiagnosticSample {};

inline cudaError_t diagnostics_allocate(DiagnosticBuffers&, int) { return cudaSuccess; }
inline void diagnostics_free(DiagnosticBuffers&) {}
inline cudaError_t diagnostics_reset(DiagnosticBuffers&) { return cudaSuccess; }
inline cudaError_t diagnostics_collect(const DiagnosticBuffers&, DiagnosticSample&, float, int) { return cudaSuccess; }
inline void diagnostics_write_header(FILE*) {}
inline void diagnostics_write(FILE*, const DiagnosticSample&) {}

#endif // DIAGNOSTICS_ENABLED

//=============================================================================
// STRESS FIELD COMPUTATION (spatial stress tensor fields)
// Enable via CMake: -DENABLE_STRESS_FIELDS=ON
//=============================================================================

#ifdef STRESS_FIELDS_ENABLED

/**
 * @brief GPU buffers for stress field computation
 * 
 * Stores spatial stress tensor fields σ_xx(x,y), σ_yy(x,y), σ_xy(x,y)
 * and derived pressure field P(x,y) = -½[σ_xx + σ_yy]
 */
struct StressFieldBuffers {
    float* d_sigma_xx_field;   // [Nx * Ny] - normal stress in x
    float* d_sigma_yy_field;   // [Nx * Ny] - normal stress in y
    float* d_sigma_xy_field;   // [Nx * Ny] - shear stress
    float* d_pressure_field;   // [Nx * Ny] - local pressure
    
    int Nx, Ny;
    bool allocated;
    
    StressFieldBuffers() : 
        d_sigma_xx_field(nullptr), d_sigma_yy_field(nullptr),
        d_sigma_xy_field(nullptr), d_pressure_field(nullptr),
        Nx(0), Ny(0), allocated(false) {}
};

inline cudaError_t stress_fields_allocate(StressFieldBuffers& buffers, int Nx, int Ny) {
    if (buffers.allocated) return cudaSuccess;
    
    buffers.Nx = Nx;
    buffers.Ny = Ny;
    size_t field_size = (size_t)Nx * Ny * sizeof(float);
    
    cudaError_t err;
    
    printf("[STRESS_FIELDS] Allocating buffers (%zu bytes each)...\n", field_size);
    
    err = cudaMalloc(&buffers.d_sigma_xx_field, field_size);
    if (err != cudaSuccess) {
        printf("[STRESS_FIELDS] Failed sigma_xx: %s\n", cudaGetErrorString(err));
        return err;
    }
    printf("[STRESS_FIELDS] sigma_xx at %p\n", (void*)buffers.d_sigma_xx_field);
    
    err = cudaMalloc(&buffers.d_sigma_yy_field, field_size);
    if (err != cudaSuccess) {
        printf("[STRESS_FIELDS] Failed sigma_yy: %s\n", cudaGetErrorString(err));
        cudaFree(buffers.d_sigma_xx_field);
        buffers.d_sigma_xx_field = nullptr;
        return err;
    }
    printf("[STRESS_FIELDS] sigma_yy at %p\n", (void*)buffers.d_sigma_yy_field);
    
    err = cudaMalloc(&buffers.d_sigma_xy_field, field_size);
    if (err != cudaSuccess) {
        printf("[STRESS_FIELDS] Failed sigma_xy: %s\n", cudaGetErrorString(err));
        cudaFree(buffers.d_sigma_xx_field);
        cudaFree(buffers.d_sigma_yy_field);
        buffers.d_sigma_xx_field = nullptr;
        buffers.d_sigma_yy_field = nullptr;
        return err;
    }
    printf("[STRESS_FIELDS] sigma_xy at %p\n", (void*)buffers.d_sigma_xy_field);
    
    err = cudaMalloc(&buffers.d_pressure_field, field_size);
    if (err != cudaSuccess) {
        printf("[STRESS_FIELDS] Failed pressure: %s\n", cudaGetErrorString(err));
        cudaFree(buffers.d_sigma_xx_field);
        cudaFree(buffers.d_sigma_yy_field);
        cudaFree(buffers.d_sigma_xy_field);
        buffers.d_sigma_xx_field = nullptr;
        buffers.d_sigma_yy_field = nullptr;
        buffers.d_sigma_xy_field = nullptr;
        return err;
    }
    printf("[STRESS_FIELDS] pressure at %p\n", (void*)buffers.d_pressure_field);
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[STRESS_FIELDS] Error after alloc sync: %s\n", cudaGetErrorString(err));
        // Clean up on error
        cudaFree(buffers.d_sigma_xx_field);
        cudaFree(buffers.d_sigma_yy_field);
        cudaFree(buffers.d_sigma_xy_field);
        cudaFree(buffers.d_pressure_field);
        buffers.d_sigma_xx_field = nullptr;
        buffers.d_sigma_yy_field = nullptr;
        buffers.d_sigma_xy_field = nullptr;
        buffers.d_pressure_field = nullptr;
        return err;
    }
    
    buffers.allocated = true;
    printf("[STRESS_FIELDS] All buffers allocated successfully\n");
    
    return cudaSuccess;
}

inline void stress_fields_free(StressFieldBuffers& buffers) {
    if (!buffers.allocated) return;
    
    cudaFree(buffers.d_sigma_xx_field);
    cudaFree(buffers.d_sigma_yy_field);
    cudaFree(buffers.d_sigma_xy_field);
    cudaFree(buffers.d_pressure_field);
    
    buffers.d_sigma_xx_field = nullptr;
    buffers.d_sigma_yy_field = nullptr;
    buffers.d_sigma_xy_field = nullptr;
    buffers.d_pressure_field = nullptr;
    buffers.allocated = false;
}

inline cudaError_t stress_fields_reset(StressFieldBuffers& buffers) {
    if (!buffers.allocated) return cudaErrorNotReady;
    
    size_t field_size = (size_t)buffers.Nx * buffers.Ny * sizeof(float);
    
    cudaMemset(buffers.d_sigma_xx_field, 0, field_size);
    cudaMemset(buffers.d_sigma_yy_field, 0, field_size);
    cudaMemset(buffers.d_sigma_xy_field, 0, field_size);
    cudaMemset(buffers.d_pressure_field, 0, field_size);
    
    return cudaGetLastError();
}

#else // STRESS_FIELDS not enabled - stubs

struct StressFieldBuffers {
    bool allocated = false;
    int Nx = 0, Ny = 0;
};

inline cudaError_t stress_fields_allocate(StressFieldBuffers&, int, int) { return cudaSuccess; }
inline void stress_fields_free(StressFieldBuffers&) {}
inline cudaError_t stress_fields_reset(StressFieldBuffers&) { return cudaSuccess; }

#endif // STRESS_FIELDS_ENABLED

} // namespace cellsim
