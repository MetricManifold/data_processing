#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


//=============================================================================
// GPU Memory Tracker - Only active when SAFE_MODE is defined at compile time
//
// Usage:
//   - Build with -DSAFE_MODE to enable tracking
//   - Replace cudaMalloc/cudaFree with CUDA_MALLOC/CUDA_FREE macros
//   - Set GPU_MEMORY_LIMIT environment variable to override default 1GB limit
//
// In release builds (without -DSAFE_MODE), these macros expand to plain
// cudaMalloc/cudaFree with zero overhead.
//=============================================================================

#ifdef SAFE_MODE

namespace cellsim {
namespace gpu_memory {

// Global tracking state
inline size_t &allocated_bytes() {
  static size_t bytes = 0;
  return bytes;
}

inline size_t &peak_bytes() {
  static size_t bytes = 0;
  return bytes;
}

inline size_t &allocation_count() {
  static size_t count = 0;
  return count;
}

inline size_t get_memory_limit() {
  static size_t limit = 0;
  static bool initialized = false;
  if (!initialized) {
    // Default 1GB limit, can be overridden by environment variable
    limit = 1ULL * 1024 * 1024 * 1024;
    const char *env_limit = std::getenv("GPU_MEMORY_LIMIT");
    if (env_limit) {
      limit = std::strtoull(env_limit, nullptr, 10);
    }
    initialized = true;
    printf("[SAFE_MODE] GPU memory limit: %.2f MB\n",
           limit / (1024.0 * 1024.0));
  }
  return limit;
}

inline cudaError_t tracked_malloc(void **ptr, size_t size, const char *file,
                                  int line) {
  size_t limit = get_memory_limit();
  size_t current = allocated_bytes();

  // Check if this allocation would exceed the limit
  if (current + size > limit) {
    fprintf(stderr, "\n");
    fprintf(stderr, "======================================================\n");
    fprintf(stderr, "[SAFE_MODE] GPU MEMORY LIMIT EXCEEDED!\n");
    fprintf(stderr, "======================================================\n");
    fprintf(stderr, "  Location: %s:%d\n", file, line);
    fprintf(stderr, "  Requested: %.2f MB\n", size / (1024.0 * 1024.0));
    fprintf(stderr, "  Current usage: %.2f MB\n", current / (1024.0 * 1024.0));
    fprintf(stderr, "  Would be: %.2f MB\n",
            (current + size) / (1024.0 * 1024.0));
    fprintf(stderr, "  Limit: %.2f MB\n", limit / (1024.0 * 1024.0));
    fprintf(stderr, "======================================================\n");
    fprintf(stderr, "Aborting to prevent system hang.\n");
    fprintf(stderr, "Set GPU_MEMORY_LIMIT env var to increase limit.\n");
    fprintf(stderr,
            "======================================================\n\n");
    std::exit(1);
  }

  cudaError_t err = cudaMalloc(ptr, size);
  if (err == cudaSuccess) {
    allocated_bytes() += size;
    allocation_count()++;
    if (allocated_bytes() > peak_bytes()) {
      peak_bytes() = allocated_bytes();
    }
  }
  return err;
}

inline cudaError_t tracked_free(void *ptr, size_t size, const char *file,
                                int line) {
  (void)file;
  (void)line; // Unused but available for debugging
  cudaError_t err = cudaFree(ptr);
  if (err == cudaSuccess && ptr != nullptr) {
    if (size <= allocated_bytes()) {
      allocated_bytes() -= size;
    } else {
      // Shouldn't happen, but prevent underflow
      allocated_bytes() = 0;
    }
  }
  return err;
}

inline void print_stats() {
  printf("\n[SAFE_MODE] GPU Memory Statistics:\n");
  printf("  Current: %.2f MB\n", allocated_bytes() / (1024.0 * 1024.0));
  printf("  Peak: %.2f MB\n", peak_bytes() / (1024.0 * 1024.0));
  printf("  Total allocations: %zu\n", allocation_count());
}

} // namespace gpu_memory
} // namespace cellsim

// Macros for tracked allocation - include size tracking
#define CUDA_MALLOC(ptr, size)                                                 \
  cellsim::gpu_memory::tracked_malloc((void **)(ptr), (size), __FILE__,        \
                                      __LINE__)

// For free, we need to track the size - caller must provide it
#define CUDA_FREE(ptr, size)                                                   \
  cellsim::gpu_memory::tracked_free((ptr), (size), __FILE__, __LINE__)

// Print memory stats
#define CUDA_MEMORY_STATS() cellsim::gpu_memory::print_stats()

#else // !SAFE_MODE

// In release mode, these are just plain CUDA calls with zero overhead
#define CUDA_MALLOC(ptr, size) cudaMalloc((ptr), (size))
#define CUDA_FREE(ptr, size) cudaFree((ptr))
#define CUDA_MEMORY_STATS() ((void)0)

#endif // SAFE_MODE
