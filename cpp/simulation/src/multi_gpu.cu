// ---------------------------------------------------------------------------
// Multi-GPU NCCL implementation. Single TU, two branches selected at
// compile time by ENABLE_MULTI_GPU.
//
//   ENABLE_MULTI_GPU=ON  : real NCCL.
//   ENABLE_MULTI_GPU=OFF : stubs that report unavailability.
//
// The stubs let main.cu and sim.cu always include "multi_gpu.cuh" and call
// mg_available() unconditionally; the actual --gpus>1 dispatch is gated
// in main.cu.
// ---------------------------------------------------------------------------

#include "multi_gpu.cuh"
#include "sim.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MG_CUDA_CK(call) do {                                                  \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
        fprintf(stderr, "[multi_gpu] CUDA %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(e));                                        \
        return false;                                                          \
    }                                                                          \
} while(0)

#ifdef ENABLE_MULTI_GPU

#include <nccl.h>

struct MgComm { ncclComm_t c; };

#define MG_NCCL_CK(call) do {                                                  \
    ncclResult_t r = (call);                                                   \
    if (r != ncclSuccess) {                                                    \
        fprintf(stderr, "[multi_gpu] NCCL %s:%d: %s\n", __FILE__, __LINE__,    \
                ncclGetErrorString(r));                                        \
        return false;                                                          \
    }                                                                          \
} while(0)

bool mg_available() { return true; }

bool mg_init_world(int world_size, MgWorld& out) {
    out = MgWorld{};
    out.world_size = world_size;
    if (world_size <= 0) {
        fprintf(stderr, "[multi_gpu] world_size must be >= 1 (got %d)\n", world_size);
        return false;
    }

    int dev_count = 0;
    MG_CUDA_CK(cudaGetDeviceCount(&dev_count));

    // Loopback test mode: pin every rank to the same physical device.
    // Lets us validate the multi-GPU code path on single-GPU hardware
    // (WSL laptop) without real multi-GPU. NCCL supports same-device
    // peers and uses a fast in-memory channel.
    //   CELL_SIM_LOOPBACK_DEVICE=0 (or any valid device id)
    //   -> all ranks attached to that device.
    // Not for production runs; the env var is the deliberate signal.
    const char* loopback_env = std::getenv("CELL_SIM_LOOPBACK_DEVICE");
    int loopback_dev = -1;
    if (loopback_env && loopback_env[0] != '\0') {
        loopback_dev = std::atoi(loopback_env);
        if (loopback_dev < 0 || loopback_dev >= dev_count) {
            fprintf(stderr,
                "[multi_gpu] CELL_SIM_LOOPBACK_DEVICE=%d invalid (have %d devices)\n",
                loopback_dev, dev_count);
            return false;
        }
        fprintf(stdout,
            "[multi_gpu] LOOPBACK MODE: all %d ranks on device %d\n",
            world_size, loopback_dev);
    } else if (dev_count < world_size) {
        fprintf(stderr,
                "[multi_gpu] requested %d GPUs but only %d visible. "
                "Set CUDA_VISIBLE_DEVICES, reduce --gpus, or set "
                "CELL_SIM_LOOPBACK_DEVICE=<id> for testing.\n",
                world_size, dev_count);
        return false;
    }

    out.devices.resize(world_size);
    for (int g = 0; g < world_size; ++g) {
        out.devices[g] = (loopback_dev >= 0) ? loopback_dev : g;
    }

    out.streams.resize(world_size, nullptr);
    for (int g = 0; g < world_size; ++g) {
        MG_CUDA_CK(cudaSetDevice(out.devices[g]));
        MG_CUDA_CK(cudaStreamCreateWithFlags(&out.streams[g],
                                             cudaStreamNonBlocking));
    }

    // Single-process, multi-device init: ncclCommInitAll handles the unique
    // ID exchange internally and creates one comm per device.
    std::vector<ncclComm_t> raw(world_size);
    MG_NCCL_CK(ncclCommInitAll(raw.data(), world_size, out.devices.data()));

    out.comms.resize(world_size, nullptr);
    for (int g = 0; g < world_size; ++g) {
        out.comms[g] = new MgComm{raw[g]};
    }

    fprintf(stdout, "[multi_gpu] NCCL world initialized: %d devices\n",
            world_size);
    return true;
}

void mg_finalize_world(MgWorld& w) {
    for (auto* c : w.comms) {
        if (c) {
            ncclCommDestroy(c->c);
            delete c;
        }
    }
    w.comms.clear();
    for (size_t g = 0; g < w.streams.size(); ++g) {
        if (w.streams[g]) {
            cudaSetDevice(w.devices[g]);
            cudaStreamDestroy(w.streams[g]);
        }
    }
    w.streams.clear();
    w.devices.clear();
    w.world_size = 1;
}

void mg_allreduce_sum_f32(MgComm* comm, float* buf, std::size_t n_floats,
                          cudaStream_t stream)
{
    if (!comm || n_floats == 0) return;
    ncclResult_t r = ncclAllReduce((const void*)buf, (void*)buf,
                                   n_floats, ncclFloat32, ncclSum,
                                   comm->c, stream);
    if (r != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclAllReduce: %s\n",
                ncclGetErrorString(r));
        std::exit(1);
    }
}

void mg_allreduce_sum_i32(MgComm* comm, int32_t* buf, std::size_t n_ints,
                          cudaStream_t stream)
{
    if (!comm || n_ints == 0) return;
    ncclResult_t r = ncclAllReduce((const void*)buf, (void*)buf,
                                   n_ints, ncclInt32, ncclSum,
                                   comm->c, stream);
    if (r != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclAllReduce(i32): %s\n",
                ncclGetErrorString(r));
        std::exit(1);
    }
}

void mg_send_recv_f32(MgComm* comm,
                      const float* src, int peer_send,
                      float* dst, int peer_recv,
                      std::size_t n_floats,
                      cudaStream_t stream)
{
    if (!comm || n_floats == 0) return;
    // Issue both the send and the recv. Caller is responsible for being
    // inside an mg_group_start/mg_group_end pair so NCCL can match this
    // with the symmetric calls on the peer rank.
    ncclResult_t rs = ncclSend((const void*)src, n_floats, ncclFloat32,
                               peer_send, comm->c, stream);
    if (rs != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclSend: %s\n",
                ncclGetErrorString(rs));
        std::exit(1);
    }
    ncclResult_t rr = ncclRecv((void*)dst, n_floats, ncclFloat32,
                               peer_recv, comm->c, stream);
    if (rr != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclRecv: %s\n",
                ncclGetErrorString(rr));
        std::exit(1);
    }
}

void mg_send_recv_bytes(MgComm* comm,
                        const void* src, int peer_send,
                        void* dst, int peer_recv,
                        std::size_t n_bytes,
                        cudaStream_t stream)
{
    if (!comm || n_bytes == 0) return;
    ncclResult_t rs = ncclSend(src, n_bytes, ncclChar,
                               peer_send, comm->c, stream);
    if (rs != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclSend(bytes): %s\n",
                ncclGetErrorString(rs));
        std::exit(1);
    }
    ncclResult_t rr = ncclRecv(dst, n_bytes, ncclChar,
                               peer_recv, comm->c, stream);
    if (rr != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclRecv(bytes): %s\n",
                ncclGetErrorString(rr));
        std::exit(1);
    }
}

void mg_send_bytes(MgComm* comm,
                   const void* src, int peer,
                   std::size_t n_bytes,
                   cudaStream_t stream)
{
    if (!comm || n_bytes == 0) return;
    ncclResult_t r = ncclSend(src, n_bytes, ncclChar, peer, comm->c, stream);
    if (r != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclSend(bytes): %s\n",
                ncclGetErrorString(r));
        std::exit(1);
    }
}

void mg_recv_bytes(MgComm* comm,
                   void* dst, int peer,
                   std::size_t n_bytes,
                   cudaStream_t stream)
{
    if (!comm || n_bytes == 0) return;
    ncclResult_t r = ncclRecv(dst, n_bytes, ncclChar, peer, comm->c, stream);
    if (r != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclRecv(bytes): %s\n",
                ncclGetErrorString(r));
        std::exit(1);
    }
}

void mg_send_recv_i32(MgComm* comm,
                      const int* src, int peer_send,
                      int* dst, int peer_recv,
                      cudaStream_t stream)
{
    if (!comm) return;
    ncclResult_t rs = ncclSend((const void*)src, 1, ncclInt32,
                               peer_send, comm->c, stream);
    if (rs != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclSend(i32): %s\n",
                ncclGetErrorString(rs));
        std::exit(1);
    }
    ncclResult_t rr = ncclRecv((void*)dst, 1, ncclInt32,
                               peer_recv, comm->c, stream);
    if (rr != ncclSuccess) {
        fprintf(stderr, "[multi_gpu] ncclRecv(i32): %s\n",
                ncclGetErrorString(rr));
        std::exit(1);
    }
}

void mg_group_start() { ncclGroupStart(); }
void mg_group_end()   { ncclGroupEnd(); }

#else  // ENABLE_MULTI_GPU OFF

// Stub branch: never link NCCL. mg_available() returns false; any other
// call exits with a clear message. The orchestrator gates --gpus>1 in
// main.cu so these unreachable-call stubs are paranoia, not the primary
// safety net.

struct MgComm { int dummy; };

bool mg_available() { return false; }

bool mg_init_world(int /*world_size*/, MgWorld& out) {
    out = MgWorld{};
    fprintf(stderr,
            "[multi_gpu] this binary was built without ENABLE_MULTI_GPU. "
            "Rebuild with cmake -DENABLE_MULTI_GPU=ON to enable --gpus>1.\n");
    return false;
}

void mg_finalize_world(MgWorld& /*w*/) {}

void mg_allreduce_sum_f32(MgComm* /*c*/, float* /*buf*/,
                          std::size_t /*n*/, cudaStream_t /*s*/) {
    fprintf(stderr, "[multi_gpu] mg_allreduce_sum_f32 called in stub build\n");
    std::exit(1);
}

void mg_allreduce_sum_i32(MgComm* /*c*/, int32_t* /*buf*/,
                          std::size_t /*n*/, cudaStream_t /*s*/) {
    fprintf(stderr, "[multi_gpu] mg_allreduce_sum_i32 called in stub build\n");
    std::exit(1);
}

void mg_send_recv_f32(MgComm* /*c*/,
                      const float* /*src*/, int /*peer_send*/,
                      float* /*dst*/, int /*peer_recv*/,
                      std::size_t /*n*/, cudaStream_t /*s*/) {
    fprintf(stderr, "[multi_gpu] mg_send_recv_f32 called in stub build\n");
    std::exit(1);
}

void mg_send_recv_bytes(MgComm* /*c*/,
                        const void* /*src*/, int /*peer_send*/,
                        void* /*dst*/, int /*peer_recv*/,
                        std::size_t /*n*/, cudaStream_t /*s*/) {
    fprintf(stderr, "[multi_gpu] mg_send_recv_bytes called in stub build\n");
    std::exit(1);
}

void mg_send_bytes(MgComm*, const void*, int, std::size_t, cudaStream_t) {
    fprintf(stderr, "[multi_gpu] mg_send_bytes called in stub build\n");
    std::exit(1);
}

void mg_recv_bytes(MgComm*, void*, int, std::size_t, cudaStream_t) {
    fprintf(stderr, "[multi_gpu] mg_recv_bytes called in stub build\n");
    std::exit(1);
}

void mg_send_recv_i32(MgComm* /*c*/,
                      const int* /*src*/, int /*peer_send*/,
                      int* /*dst*/, int /*peer_recv*/,
                      cudaStream_t /*s*/) {
    fprintf(stderr, "[multi_gpu] mg_send_recv_i32 called in stub build\n");
    std::exit(1);
}

void mg_group_start() {}
void mg_group_end()   {}

// run_multi_gpu is implemented in sim.cu under the same ENABLE_MULTI_GPU
// guard. main.cu's --gpus>1 path is gated on mg_available() so this stub
// is unreachable in normal use; it exists only so the build link succeeds
// when the orchestrator's full implementation is excluded.
int run_multi_gpu(const MultiGpuRunArgs& /*args*/) {
    fprintf(stderr,
            "[multi_gpu] run_multi_gpu called in stub build "
            "(ENABLE_MULTI_GPU=OFF). This is a bug — main.cu should have "
            "rejected --gpus>1 before reaching here.\n");
    std::exit(1);
}

#endif  // ENABLE_MULTI_GPU
