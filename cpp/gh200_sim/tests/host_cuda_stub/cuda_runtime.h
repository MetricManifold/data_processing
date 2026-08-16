#pragma once

#include <cstddef>

// The support-layout contract exercises ordinary constexpr/host code only.
// This minimal shim lets a CPU compiler parse the public CUDA header without
// emulating, linking, or executing any CUDA API.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

struct uint4 {
    unsigned int x, y, z, w;
};
using cudaStream_t = void*;
using std::size_t;
