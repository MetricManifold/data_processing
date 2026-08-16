cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/PFLargeClassEdge.cmake)
project(gh200_sim LANGUAGES CXX CUDA)

# All CMake-built translation units see the marker. Any target that includes
# params.cuh but forgets to link pf_shape_config then fails at compile time
# instead of silently falling back to the compact direct-build default.
add_compile_definitions(PF_GH200_CMAKE_BUILD=1)
add_library(pf_shape_config INTERFACE)
target_compile_definitions(pf_shape_config INTERFACE
  PF_EXTENDED_SUPPORT_LAYOUT=${PF_EXTENDED_SUPPORT_LAYOUT_VALUE})

# Fatal in-kernel alarms are always enabled. The only optional counter is the
# high-frequency advisory support-clip diagnostic; opt in for validation with:
#   cmake .. -DPF_ALARMS=ON
#
# PF_ALARMS does not affect S_overflow, q_clamp, class_exhausted,
# S_other_negative, phi_nonfinite, V_nonpositive or class_unsupported. Those
# conditions invalidate a production trajectory and cannot be compiled out.
if(DEFINED PF_NO_ALARMS)
    message(FATAL_ERROR
        "-DPF_NO_ALARMS is retired: production-fatal alarms cannot be disabled.\n"
        "  To enable the advisory support_clip counter: -DPF_ALARMS=ON\n"
        "  Otherwise drop the flag entirely.")
endif()
option(PF_ALARMS "enable the advisory support_clip counter" OFF)
if(PF_ALARMS)
    add_compile_definitions(PF_ALARMS)
    message(STATUS "gh200_sim: fatal alarms ENABLED; advisory support_clip ENABLED")
else()
    message(STATUS "gh200_sim: fatal alarms ENABLED; advisory support_clip disabled")
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
  src/validation_centroid.cu
  src/sim.cu
  src/checkpoint.cu
)
target_link_libraries(pf_core PUBLIC pf_shape_config)
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

# ---------------------------------------------------------------------------
# EVERYTHING BELOW THIS LINE IS OPTIONAL.
#
# `cell_gh200` is the deployable artifact and it depends on nothing outside
# include/, src/, cmake/ and ../common. Tools and tests are built only when
# their sources are present, so a source drop that carries just the solver
# configures and builds cleanly instead of dying at generate time.
#
# This is not hypothetical tidiness: the tracked tree shipped for months with an
# unconditional `add_executable(dump_phi tools/dump_phi.cu)` and no tools/
# directory, and because CMake validates every declared target's sources during
# generation, that one missing FILE prevented `cell_gh200` from getting a build
# system at all. A missing validation artifact must never block the solver.
# ---------------------------------------------------------------------------
set(PF_OPTIONAL_FOUND "")
set(PF_OPTIONAL_MISSING "")

# The no-contraction policy is only needed by the host-side golden fixtures, so
# an unknown host compiler must not be fatal to a solver-only build.
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  set(PF_HOST_NO_CONTRACTION -Mnofma)
  set(PF_HOST_NO_CONTRACTION_KNOWN TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  set(PF_HOST_NO_CONTRACTION -ffp-contract=off)
  set(PF_HOST_NO_CONTRACTION_KNOWN TRUE)
else()
  set(PF_HOST_NO_CONTRACTION "")
  set(PF_HOST_NO_CONTRACTION_KNOWN FALSE)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/dump_phi.cu)
  add_executable(dump_phi tools/dump_phi.cu)
  target_link_libraries(dump_phi PRIVATE pf_shape_config)
  target_include_directories(dump_phi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)
  target_compile_options(dump_phi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
  list(APPEND PF_OPTIONAL_FOUND dump_phi)
else()
  list(APPEND PF_OPTIONAL_MISSING dump_phi)
endif()

# Host-only published-placement generator. It writes the one accepted-centre
# table that both branches of a matched pair must consume. The accepted float
# table is a committed bit-level fixture, so it needs an explicit
# no-contraction flag; without one the table would silently change.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/palmieri_centres.cpp)
  if(PF_HOST_NO_CONTRACTION_KNOWN)
    add_executable(palmieri_centres tools/palmieri_centres.cpp)
    target_include_directories(palmieri_centres PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(palmieri_centres PRIVATE ${PF_HOST_NO_CONTRACTION})
    list(APPEND PF_OPTIONAL_FOUND palmieri_centres)
  else()
    message(STATUS
      "gh200_sim: skipping palmieri_centres -- host compiler "
      "'${CMAKE_CXX_COMPILER_ID}' has no verified no-contraction flag, and the "
      "accepted centre table is a bit-level fixture. The solver is unaffected.")
    list(APPEND PF_OPTIONAL_MISSING palmieri_centres)
  endif()
else()
  list(APPEND PF_OPTIONAL_MISSING palmieri_centres)
endif()

include(CTest)
# Tests are opt-out AND existence-gated: BUILD_TESTING alone is not enough,
# because a solver-only source drop has no tests/ directory to compile.
if(BUILD_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests
                 AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools
                 AND PF_HOST_NO_CONTRACTION_KNOWN)
  # Not REQUIRED: a missing interpreter disables the python tests, it does not
  # fail the configure of a solver the user may only want to run.
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(NOT Python3_FOUND)
    message(STATUS "gh200_sim: Python3 not found; python CPU tests skipped")
  endif()
  add_executable(palmieri_initializer_test tools/palmieri_initializer_test.cpp)
  target_include_directories(palmieri_initializer_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(palmieri_initializer_test PRIVATE
    ${PF_HOST_NO_CONTRACTION})
  add_test(NAME palmieri_initializer_cpu COMMAND palmieri_initializer_test)
  if(Python3_FOUND AND TARGET palmieri_centres)
    add_test(NAME palmieri_pairing_cpu
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/palmieri_pairing_test.py
        --tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/palmieri_pairing.py
        --generator $<TARGET_FILE:palmieri_centres>)
  endif()
  foreach(layout IN ITEMS 0 1)
    if(layout EQUAL 0)
      set(expected_tile 256)
      set(expected_edge 208)
      set(layout_name compact)
    else()
      set(expected_tile 288)
      set(expected_edge 224)
      set(layout_name extended)
    endif()
    add_executable(support_layout_${layout_name}_test
      tests/shape_class_edge_test.cpp)
    target_include_directories(support_layout_${layout_name}_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/host_cuda_stub
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    target_compile_definitions(support_layout_${layout_name}_test PRIVATE
      PF_EXTENDED_SUPPORT_LAYOUT=${layout}
      PF_EXPECTED_TILE_PITCH=${expected_tile}
      PF_EXPECTED_LARGE_CLASS_EDGE=${expected_edge})
    add_test(NAME support_layout_${layout_name}_cpu
      COMMAND support_layout_${layout_name}_test)
  endforeach()
  if(Python3_FOUND)
    add_test(NAME support_layout_one_factor_cpu
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/shape_class_edge_compare.py
        --compact $<TARGET_FILE:support_layout_compact_test>
        --extended $<TARGET_FILE:support_layout_extended_test>)
    add_test(NAME support_layout_source_contract_cpu
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/support_layout_source_contract.py)
  endif()
  add_executable(checkpoint_repack_cpu_test
    tests/checkpoint_repack_cpu_test.cu)
  target_link_libraries(checkpoint_repack_cpu_test PRIVATE pf_shape_config)
  target_include_directories(checkpoint_repack_cpu_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)
  target_compile_options(checkpoint_repack_cpu_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
  add_test(NAME checkpoint_repack_cpu COMMAND checkpoint_repack_cpu_test)
endif()

# Optional: the hardware probe that produced CALIBRATION.md.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gh200_probe.cu)
  add_executable(gh200_probe tools/gh200_probe.cu)
  target_link_libraries(gh200_probe PRIVATE pf_shape_config)
  target_compile_options(gh200_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-fmad=true>)
endif()

# Install the solver unconditionally; install each optional tool only if it was
# actually configured, so `cmake --install` never fails on a solver-only drop.
install(TARGETS cell_gh200 RUNTIME DESTINATION bin)
foreach(pf_tool dump_phi palmieri_centres)
  if(TARGET ${pf_tool})
    install(TARGETS ${pf_tool} RUNTIME DESTINATION bin)
  endif()
endforeach()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/palmieri_pairing.py)
  install(PROGRAMS tools/palmieri_pairing.py DESTINATION bin)
endif()

if(PF_OPTIONAL_MISSING)
  message(STATUS "gh200_sim: optional components NOT built (solver unaffected): "
                 "${PF_OPTIONAL_MISSING}")
endif()
message(STATUS "gh200_sim: solver cell_gh200 + optional [${PF_OPTIONAL_FOUND}]")
message(STATUS "gh200_sim: build ${CMAKE_BUILD_TYPE}, "
               "CUDA arch ${CMAKE_CUDA_ARCHITECTURES}, "
               "host ${CMAKE_SYSTEM_PROCESSOR}, "
               "support layout ${PF_SUPPORT_TILE_PITCH}/${PF_SUPPORT_LARGE_EDGE}")
