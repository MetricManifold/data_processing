#pragma once
#include "types.cuh"

void launch_hash_build(CellArrays& c, int Nx, int Ny);
void launch_pre_step(CellArrays& c, const SimParams& p, int step,
                     int& cache_w, int& cache_h);
void launch_fused(CellArrays& c, const SimParams& p,
                  int max_w, int max_h, int step);
void launch_swap(CellArrays& c, int Nx, int Ny);
void launch_polar(CellArrays& c, const SimParams& p);
void launch_rng_init(CellArrays& c, unsigned long seed);
void launch_initial_reduce(CellArrays& c, const SimParams& p,
                           int max_w, int max_h);
void launch_initial_velocity(CellArrays& c, const SimParams& p,
                             int max_w, int max_h);
