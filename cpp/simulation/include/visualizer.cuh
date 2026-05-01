#pragma once

// ---------------------------------------------------------------------------
// Live GPU visualizer (CUDA + OpenGL interop).
//
// Off by default. Build with -DENABLE_VISUALIZER=ON to enable. Requires
// GLFW3 and OpenGL on the build host. Intended ONLY for local debugging /
// demos -- never enable on cluster builds.
//
// Renders the global S(x,y) = sum_n phi_n(x,y)^2 field directly from device
// memory each call (no host copies). Single-window viridis colormap; user
// closes the window or presses ESC to terminate the viewer (the simulation
// keeps running). Designed for sim_v3 tile-pool layout.
//
// Overlays (drawn from device data, no host roundtrip):
//   * Red tint on soft cells (gamma_cell < soft_gamma_threshold).
//   * Cyan dashed bbox on soft cells (active rect).
//   * White velocity arrows at every cell centroid.
// ---------------------------------------------------------------------------

#ifdef ENABLE_VISUALIZER

struct GLFWwindow;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

namespace cellsim {

class Visualizer {
public:
    Visualizer() = default;
    ~Visualizer();

    // Open a window sized to the field (capped to 900 px tall, 1600 px wide,
    // minimum 600 px tall so a small field is still visible).
    bool init(int field_width, int field_height,
              const char* title = "cell_sim live");

    // Composite + present one frame.
    //
    //   d_S       : device pointer, [Nx*Ny] float, the global sum-field.
    //   d_phi_in  : device pointer, [N*TILE_AREA] float, current phi pool.
    //   d_origin  : device pointer, [2*N] int, (gx0, gy0) per cell.
    //   d_rect    : device pointer, [4*N] int, (rx0, ry0, rw, rh) per cell.
    //   d_Cx,d_Cy : device pointers, [N] float, tile-local sum(phi^2*lx/ly).
    //   d_volumes : device pointer, [N] float, sum(phi) per cell.
    //   d_vx,d_vy : device pointers, [N] float, cell velocities.
    //   d_gamma   : device pointer, [N] float, per-cell surface tension.
    //
    // soft_gamma_threshold: cells with gamma < this are treated as "soft"
    //   (red tint, bbox drawn). Default 0.5 covers gamma=0.25 vs 1.0.
    void update(const float* d_S,
                const float* d_phi_in,
                const int*   d_origin,
                const int*   d_rect,
                const float* d_Cx, const float* d_Cy,
                const float* d_Cxx, const float* d_Cyy,
                const float* d_volumes,
                const float* d_vx, const float* d_vy,
                const float* d_gamma,
                const float* d_tgt_radius,
                int   num_cells,
                int   Nx, int   Ny,
                double current_time,
                float  bbox_K = 2.0f,
                float  lambda = 7.0f,
                float  soft_gamma_threshold = 0.5f);

    bool should_close() const;
    void shutdown();
    bool is_initialized() const { return initialized; }

private:
    GLFWwindow* window = nullptr;
    unsigned int gl_texture = 0;
    cudaGraphicsResource_t cuda_resource = nullptr;
    bool initialized = false;
    int tex_width = 0, tex_height = 0;
};

} // namespace cellsim

#endif // ENABLE_VISUALIZER
