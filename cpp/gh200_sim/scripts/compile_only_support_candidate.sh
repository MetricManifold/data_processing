#!/usr/bin/env bash
set -euo pipefail

# Login-node compile/CPU-test harness. This script contains no Slurm or GPU
# execution command. It refuses to reuse a build directory so every receipt is
# tied to a clean configure with identical flags except for the layout option.
candidate_root="/scratch/project_2019216/stevsilb/gh200_extended_support_candidate_20260815"
source_root="${candidate_root}/cpp/gh200_sim"
compact_build="${candidate_root}/build_compact_nvhpc26_3_compileonly_20260815"
extended_build="${candidate_root}/build_extended_nvhpc26_3_compileonly_20260815"

test ! -e "${compact_build}"
test ! -e "${extended_build}"

source /usr/share/lmod/lmod/init/bash
module purge
module use /appl/modulefiles/manual/general/aarch64
module load nvhpc

{
    date -Is
    hostname
    uname -a
    module -t list
    command -v cmake
    cmake --version
    command -v nvc++
    nvc++ --version
    command -v nvcc
    nvcc --version
} 2>&1 | tee "${candidate_root}/TOOLCHAIN_COMPILE_ONLY.txt"

cmake -S "${source_root}" -B "${compact_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DPF_EXTENDED_SUPPORT_LAYOUT=OFF \
    -DBUILD_TESTING=ON \
    2>&1 | tee "${candidate_root}/CMAKE_CONFIGURE_COMPACT.log"
cmake --build "${compact_build}" -j2 \
    2>&1 | tee "${candidate_root}/CMAKE_BUILD_COMPACT.log"
ctest --test-dir "${compact_build}" --output-on-failure \
    2>&1 | tee "${candidate_root}/CTEST_COMPACT.log"

cmake -S "${source_root}" -B "${extended_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DPF_EXTENDED_SUPPORT_LAYOUT=ON \
    -DBUILD_TESTING=ON \
    2>&1 | tee "${candidate_root}/CMAKE_CONFIGURE_EXTENDED.log"
cmake --build "${extended_build}" -j2 \
    2>&1 | tee "${candidate_root}/CMAKE_BUILD_EXTENDED.log"
ctest --test-dir "${extended_build}" --output-on-failure \
    2>&1 | tee "${candidate_root}/CTEST_EXTENDED.log"
