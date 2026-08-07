cmake_minimum_required(VERSION 3.20)
project(gh200_sim LANGUAGES CXX CUDA)

# Production builds may compile the alarm counters out entirely:
#   cmake .. -DPF_NO_ALARMS=ON
# support_clip alone fires on ~1.4% of cell-steps as an atomicAdd inside the hot
# kernel. Both it and class_exhausted have been measured as advisory-grade
# (truncated phi^2 mass ~1e-12 of total), so production loses no real signal.
# FLAG_CLASS_UNSUPPORTED survives regardless -- it is one atomic per CTA and
# means memory corruption, not physics.
option(PF_NO_ALARMS "compile out the in-kernel alarm counters" OFF)
if(PF_NO_ALARMS)
    add_compile_definitions(PF_NO_ALARMS)
    message(STATUS "gh200_sim: alarm counters COMPILED OUT (production build)")
endif()

# ---------------------------------------------------------------------------
# FUSE-1R -- GH200 phase-field cell engine.
#
# Target: NVIDIA GH200 120GB, sm_90 / sm_90a, CUDA 13.0 (nvhpc 26.3) or
# gcc 15.2 + cuda 13.0.2, host aarch64 Neoverse-V2.
#
# The host side is portable C++17: no x86 intrinsics, no __builtin_ia32_*, no
# SSE/AVX. Do NOT add -march=native or -mavx*; they do not exist on aarch64 and
# would silently disable the build on Grace.
# ---------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# 90 for portable sm_90 cubin+PTX; 90a enables Hopper-only instructions and
# emits sm_90-only code. Override on the command line:
#   cmake <src> -DCMAKE_CUDA_ARCHITECTURES=90a
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Determinism and diagnostics contract.
#
#   -fmad=true          explicit, so the FMA-contraction decisions that the
#                       golden-hash regression depends on are recorded in the
#                       build, not inherited from a default.
#   NEVER -use_fast_math / --ffast-math: it changes the interface profile, the
#                       reciprocals in the RHS and the reduction results.
#   -Xptxas -v          the zero-spill gate. The performance model assumes the
#                       85-register budget holds with no spill stores; grep the
#                       build log for "spill" before believing any timing.
# ---------------------------------------------------------------------------
set(PF_CUDA_FLAGS
  --expt-relaxed-constexpr
  -fmad=true
  -lineinfo
  -Xptxas=-v
)

add_library(pf_core STATIC
  src/kernels.cu
  src/sim.cu
  src/checkpoint.cu
)
# ../common holds checkpoint_format.h — the ONE definition of the on-disk
# checkpoint layout, shared with cpp/simulation. It is included, never copied:
# the old tree already drifted by duplicating physics constants and the file
# format is not going to repeat it.
target_include_directories(pf_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_compile_options(pf_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
set_target_properties(pf_core PROPERTIES CUDA_RUNTIME_LIBRARY Static)

add_executable(cell_gh200 src/main.cu)
target_link_libraries(cell_gh200 PRIVATE pf_core)
target_compile_options(cell_gh200 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)

add_executable(dump_phi tools/dump_phi.cu)
target_include_directories(dump_phi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_compile_options(dump_phi PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)

# Optional: the hardware probe that produced CALIBRATION.md.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gh200_probe.cu)
  add_executable(gh200_probe tools/gh200_probe.cu)
  target_compile_options(gh200_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-fmad=true>)
endif()

install(TARGETS cell_gh200 dump_phi RUNTIME DESTINATION bin)

message(STATUS "gh200_sim: build ${CMAKE_BUILD_TYPE}, "
               "CUDA arch ${CMAKE_CUDA_ARCHITECTURES}, "
               "host ${CMAKE_SYSTEM_PROCESSOR}")
