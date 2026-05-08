#pragma once
// ---------------------------------------------------------------------------
// Multi-GPU wrapper. Thin abstraction around NCCL so the rest of the codebase
// does not need to depend on (or even include) <nccl.h>.
//
// Two build modes, controlled by the ENABLE_MULTI_GPU compile flag wired up
// in CMakeLists.txt:
//
//   ENABLE_MULTI_GPU = OFF (default, also Windows local builds without NCCL):
//     mg_available() -> false. mg_init_world() -> false. The orchestrator
//     in main.cu refuses --gpus > 1 with a clear error message. The single-
//     GPU path (--gpus 1) is bit-identical to a no-NCCL baseline build.
//
//   ENABLE_MULTI_GPU = ON (cluster H100 nodes):
//     Real NCCL implementation is linked in. Single process, one host
//     thread, multiple devices via cudaSetDevice. NCCL group brackets
//     coalesce per-rank issue into one collective.
//
// The data-parallel scheme implemented here is:
//   - Cells are partitioned across G ranks: rank g owns indices
//     [cell_offset[g], cell_offset[g] + cells_local[g]).
//   - The global S(x,y) field is FULLY REPLICATED on every rank.
//   - Per step: each rank scatters phi_n^2 of its cells into its local S.
//     Then ncclAllReduce(SUM) makes the full population's S identical on
//     every rank. Then each rank's evolve reads its own slice's phi and
//     writes its own phi_out. No tile / phi data ever crosses the wire.
//
// This is the only collective we issue per step. Sized for the target
// 12800-cell run (Nx ~ 10358, S ~ 410 MB) it is ~1-2 ms on NVLink on
// 4xH100 — comfortably less than the per-step compute cost at that N.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

// Opaque per-rank communicator handle. Real definition lives in
// src/multi_gpu.cu when ENABLE_MULTI_GPU=ON; otherwise treated as void*.
struct MgComm;

// Bundle of per-rank state owned by the orchestrator. One MgWorld per
// process. world_size == 1 is a valid configuration (no-op collectives).
struct MgWorld {
    int world_size = 1;
    std::vector<MgComm*>      comms;    // size = world_size
    std::vector<cudaStream_t> streams;  // dedicated allreduce stream/rank
    std::vector<int>          devices;  // CUDA device id per rank
};

// Compile-time capability probe.
//   - ENABLE_MULTI_GPU=OFF  -> returns false.
//   - ENABLE_MULTI_GPU=ON   -> returns true (NCCL is linked).
bool mg_available();

// Initialize NCCL communicators for `world_size` devices in this process.
// Devices used are 0..world_size-1 by default; callers can constrain via
// CUDA_VISIBLE_DEVICES. Streams are non-blocking, one per rank, used for
// the S allreduce so it can overlap with the rank's own kernel work.
//
// Returns true on success. On failure, prints to stderr and leaves `out`
// in a destroyable-but-empty state.
bool mg_init_world(int world_size, MgWorld& out);

// Destroy NCCL comms and streams. Safe to call on a partially initialised
// world or on an empty world.
void mg_finalize_world(MgWorld& w);

// In-place SUM all-reduce of `n_floats` float32s in `buf` for one rank.
// MUST be issued inside an mg_group_start()/mg_group_end() pair when
// driving multiple ranks from a single host thread, otherwise NCCL will
// deadlock waiting for peers.
void mg_allreduce_sum_f32(MgComm* comm, float* buf, std::size_t n_floats,
                          cudaStream_t stream);

// Send `n_floats` from `src` (this rank, device memory) to `peer_rank`,
// and receive `n_floats` from `peer_rank` into `dst` (this rank, device
// memory). Both must be issued inside an mg_group_start/mg_group_end
// pair. NCCL pairs sends and recvs by appearance order within a group,
// so callers on both ends must match the order. Used by the slab
// halo exchange: rank g sends its top band to rank prev and rank g
// recvs prev's bottom band into a temp staging buffer.
void mg_send_recv_f32(MgComm* comm,
                      const float* src, int peer_send,
                      float* dst, int peer_recv,
                      std::size_t n_floats,
                      cudaStream_t stream);

// Generic byte send/recv pair. Used by cell migration where the payload
// is a packed struct (phi tile + header). Internally just calls ncclSend
// and ncclRecv with ncclChar — NCCL is byte-transparent. Same group-call
// requirements as mg_send_recv_f32.
void mg_send_recv_bytes(MgComm* comm,
                        const void* src, int peer_send,
                        void* dst, int peer_recv,
                        std::size_t n_bytes,
                        cudaStream_t stream);

// Pure send / recv halves of mg_send_recv_bytes, for when the four
// directions of a migration exchange (up-send, down-send, prev-recv,
// next-recv) need independent counts. Caller must put them inside one
// mg_group_start/mg_group_end pair so NCCL can pair them across ranks.
void mg_send_bytes(MgComm* comm,
                   const void* src, int peer,
                   std::size_t n_bytes,
                   cudaStream_t stream);
void mg_recv_bytes(MgComm* comm,
                   void* dst, int peer,
                   std::size_t n_bytes,
                   cudaStream_t stream);

// Send a single int32 to `peer_send` and receive an int32 from
// `peer_recv` into *dst. Used to exchange migration counts before the
// payload-bytes exchange. Same group-call requirements.
void mg_send_recv_i32(MgComm* comm,
                      const int* src, int peer_send,
                      int* dst, int peer_recv,
                      cudaStream_t stream);

// NCCL group brackets. Required around per-rank issuing in single-process
// multi-device mode: every rank's ncclAllReduce call between Start/End is
// coalesced into one collective. No-op when ENABLE_MULTI_GPU=OFF.
void mg_group_start();
void mg_group_end();

// ---------------------------------------------------------------------------
// Cell partitioning helper (no NCCL dependency; lives here because the
// multi-GPU runner is its only consumer). Slices N_global cells across
// G ranks: ranks [0, N % G) get one extra cell each.
// ---------------------------------------------------------------------------
inline void mg_slice_cells(int N_global, int gpus, int rank,
                           int& offset_out, int& count_out)
{
    if (gpus <= 1) { offset_out = 0; count_out = N_global; return; }
    int base = N_global / gpus;
    int rem  = N_global % gpus;
    int extra_before = (rank < rem) ? rank : rem;
    offset_out = rank * base + extra_before;
    count_out  = base + ((rank < rem) ? 1 : 0);
}
