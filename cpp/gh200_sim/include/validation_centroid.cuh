#pragma once
// ===========================================================================
// Validation-only dual-centroid observable.
//
// This interface is deliberately separate from CellState and TrajPackedCell:
// the kernel reads the current phase-field frame and writes only its dedicated
// mapped output buffer.  It cannot alter solver state or the legacy trajectory
// ABI.
// ===========================================================================

#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace pf {

constexpr int kValidationCentroidThreads = 256;
constexpr uint32_t kCentroidPhiValid  = 1u << 0;
constexpr uint32_t kCentroidPhi2Valid = 1u << 1;

struct alignas(8) ValidationCentroidCell {
    int32_t  global_id;
    int32_t  cls;
    uint32_t valid_mask;
    uint32_t reserved;
    double   sum_phi;
    double   cx_phi;
    double   cy_phi;
    double   sum_phi2;
    double   cx_phi2;
    double   cy_phi2;
};
static_assert(sizeof(ValidationCentroidCell) == 64,
              "validation centroid record layout drift");

// One CTA per cell. `phi` must be the buffer holding phi^step at the sampling
// boundary. All three simulation inputs are const; only `out` is written.
void launch_validation_centroids(const float* phi,
                                 const CellState* cell,
                                 const uint8_t* cls,
                                 ValidationCentroidCell* out,
                                 int N, int L, cudaStream_t stream);

}  // namespace pf
