// visualizer.cu -- CUDA + OpenGL interop live viewer for cell_sim.
//
// Off by default. Compiled only when ENABLE_VISUALIZER is defined (set by
// the CMake option of the same name). The build will then link against
// GLFW3 + OpenGL.
//
// SHOULD NOT BE ENABLED ON CLUSTER BUILDS. Headless nodes have no display
// and GLFW init will fail at runtime. Local debugging / demo use only.

#ifdef ENABLE_VISUALIZER

// GL must precede the CUDA-GL interop header.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "visualizer.cuh"
#include "types.cuh"   // for TILE_T / TILE_AREA used by the soft-cell tint kernel
#include <cstdio>
#include <cstring>
#include <cmath>

namespace cellsim {

// ---------------------------------------------------------------------------
// Viridis-like colormap (6 control points, linearly interpolated on GPU).
// ---------------------------------------------------------------------------
__constant__ float3 d_cmap[6] = {
    {0.267f, 0.004f, 0.329f},
    {0.283f, 0.141f, 0.458f},
    {0.127f, 0.357f, 0.510f},
    {0.204f, 0.553f, 0.396f},
    {0.565f, 0.749f, 0.173f},
    {0.992f, 0.906f, 0.145f},
};

__device__ static uchar4 viridis(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float idx = v * 5.0f;
    int lo = (int)idx;
    if (lo >= 5) lo = 4;
    float frac = idx - (float)lo;
    float3 a = d_cmap[lo];
    float3 b = d_cmap[lo + 1];
    float r  = a.x + frac * (b.x - a.x);
    float g  = a.y + frac * (b.y - a.y);
    float bl = a.z + frac * (b.z - a.z);
    return make_uchar4((unsigned char)(r * 255.0f),
                       (unsigned char)(g * 255.0f),
                       (unsigned char)(bl * 255.0f), 255);
}

// S = sum phi^2 in [0, ~1]; sqrt for phi-like contrast.
__global__ static void kernel_colormap(uchar4* pixels, const float* S,
                                       int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    float v = S[y * Nx + x];
    v = sqrtf(fminf(v, 1.0f));
    pixels[y * Nx + x] = viridis(v);
}

// Single-pixel write helper (periodic-safe).
__device__ static void put_pixel(uchar4* pixels, int px, int py,
                                 int Nx, int Ny, uchar4 c) {
    px = ((px % Nx) + Nx) % Nx;
    py = ((py % Ny) + Ny) % Ny;
    pixels[py * Nx + px] = c;
}

// ---------------------------------------------------------------------------
// Soft-cell red tint. One thread per pixel of the cell's active rect.
// Blends the existing pixel with red, alpha = sqrt(phi). Cell is included
// only when gamma_cell[n] < soft_threshold.
// ---------------------------------------------------------------------------
__global__ static void kernel_tint_soft(
    uchar4* pixels, int Nx, int Ny,
    const float* phi_in,
    const int* origin, const int* rect,
    const float* gamma_cell,
    float soft_threshold) {

    int n = blockIdx.z;
    if (gamma_cell[n] >= soft_threshold) return;

    int rx0 = rect[4*n + 0];
    int ry0 = rect[4*n + 1];
    int rw  = rect[4*n + 2];
    int rh  = rect[4*n + 3];

    int lx = rx0 + blockIdx.x * blockDim.x + threadIdx.x;
    int ly = ry0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= rx0 + rw || ly >= ry0 + rh) return;

    const float* tile = phi_in + (size_t)n * TILE_AREA;
    float phi = tile[ly * TILE_T + lx];
    if (phi < 0.05f) return;
    float a = sqrtf(fminf(phi, 1.0f)) * 0.7f;  // alpha

    int gx = ((origin[2*n + 0] + lx) % Nx + Nx) % Nx;
    int gy = ((origin[2*n + 1] + ly) % Ny + Ny) % Ny;
    uchar4 cur = pixels[gy * Nx + gx];

    // Blend toward saturated red.
    float r = cur.x * (1.0f - a) + 255.0f * a;
    float g = cur.y * (1.0f - a) +  20.0f * a;
    float b = cur.z * (1.0f - a) +  20.0f * a;
    pixels[gy * Nx + gx] = make_uchar4(
        (unsigned char)fminf(r, 255.0f),
        (unsigned char)fminf(g, 255.0f),
        (unsigned char)fminf(b, 255.0f), 255);
}

// ---------------------------------------------------------------------------
// Velocity arrows. One thread per cell. Centroid computed inline from
// (origin, Cx, Cy, volumes). Skips cells whose velocity is sub-pixel.
// ---------------------------------------------------------------------------
__global__ static void kernel_draw_arrows(
    uchar4* pixels, int Nx, int Ny,
    const int* origin,
    const float* Cx, const float* Cy, const float* volumes,
    const float* vx, const float* vy,
    int num_cells, float arrow_scale) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float V = volumes[i];
    if (V < 1e-6f) return;
    float invV = 1.0f / V;
    float cx = origin[2*i + 0] + Cx[i] * invV;
    float cy = origin[2*i + 1] + Cy[i] * invV;
    int x0 = (int)roundf(cx);
    int y0 = (int)roundf(cy);

    float dvx = vx[i] * arrow_scale;
    float dvy = vy[i] * arrow_scale;
    float len = sqrtf(dvx*dvx + dvy*dvy);
    if (len < 1.0f) return;

    uchar4 col = make_uchar4(255, 255, 255, 255);

    int steps = (int)ceilf(len);
    for (int s = 0; s <= steps; ++s) {
        float t = (float)s / fmaxf((float)steps, 1.0f);
        int px = x0 + (int)roundf(dvx * t);
        int py = y0 + (int)roundf(dvy * t);
        put_pixel(pixels, px,     py,     Nx, Ny, col);
        put_pixel(pixels, px + 1, py,     Nx, Ny, col);
        put_pixel(pixels, px,     py + 1, Nx, Ny, col);
    }

    // Arrowhead.
    float nx1 = -dvx * 0.3f + dvy * 0.2f;
    float ny1 = -dvy * 0.3f - dvx * 0.2f;
    float nx2 = -dvx * 0.3f - dvy * 0.2f;
    float ny2 = -dvy * 0.3f + dvx * 0.2f;
    int tx = x0 + (int)roundf(dvx);
    int ty = y0 + (int)roundf(dvy);
    for (int s = 0; s <= 5; ++s) {
        float t = (float)s / 5.0f;
        put_pixel(pixels, tx + (int)(nx1*t), ty + (int)(ny1*t), Nx, Ny, col);
        put_pixel(pixels, tx + (int)(nx2*t), ty + (int)(ny2*t), Nx, Ny, col);
    }
}

// ---------------------------------------------------------------------------
// bbox draw policy: which cells get cyan/orange boxes drawn?
//   - all soft cells (gamma_cell < soft_threshold), and
//   - a fixed watch list of normal cells (for comparison: see what a stiff
//     cell's rect looks like vs the soft one).
// Adjust WATCH_NORMAL[] if you want a different sample.
// ---------------------------------------------------------------------------
__device__ static bool should_draw_box(int i, const float* gamma_cell,
                                       float soft_threshold,
                                       int num_cells) {
    if (gamma_cell[i] < soft_threshold) return true;
    // A handful of normal cells for visual comparison.
    int watch[] = {1, 17, 35, 53};
    for (int k = 0; k < 4; ++k) {
        if (watch[k] < num_cells && i == watch[k]) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Soft-cell bbox. One thread per cell; only soft cells draw.
// Dashed cyan rectangle aligned to the cell's active rect (origin + rect).
// ---------------------------------------------------------------------------
__global__ static void kernel_draw_soft_bbox(
    uchar4* pixels, int Nx, int Ny,
    const int* origin, const int* rect,
    const float* gamma_cell,
    float soft_threshold,
    int num_cells) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;
    if (!should_draw_box(i, gamma_cell, soft_threshold, num_cells)) return;

    int gx = origin[2*i + 0] + rect[4*i + 0];
    int gy = origin[2*i + 1] + rect[4*i + 1];
    int w  = rect[4*i + 2];
    int h  = rect[4*i + 3];

    uchar4 col = make_uchar4(100, 220, 255, 255);  // cyan

    for (int x = 0; x < w; x += 2) {
        put_pixel(pixels, gx + x, gy,         Nx, Ny, col);
        put_pixel(pixels, gx + x, gy + h - 1, Nx, Ny, col);
    }
    for (int y = 0; y < h; y += 2) {
        put_pixel(pixels, gx,         gy + y, Nx, Ny, col);
        put_pixel(pixels, gx + w - 1, gy + y, Nx, Ny, col);
    }
}

// ---------------------------------------------------------------------------
// Soft-cell shape extent box. Shows the *actual estimated cell extent*
// (NOT the padded sim rect) using the same second moments. For a smooth
// phase-field cell the visible edge sits at roughly sigma + lambda/2 from
// the centroid (sigma ~ R/2 for a disk; the +lambda/2 captures the tanh
// interface decay). So the orange box is a tight estimate of the visible
// cell footprint -- compare to cyan (the sim rect) to see how much
// padding the rebind heuristic gives.
// ---------------------------------------------------------------------------
__global__ static void kernel_draw_sigma_box(
    uchar4* pixels, int Nx, int Ny,
    const int* origin,
    const float* Cx, const float* Cy,
    const float* Cxx, const float* Cyy,
    const float* volumes,
    const float* gamma_cell,
    const float* tgt_radius,
    float K, float lambda, float soft_threshold,
    int num_cells) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;
    if (!should_draw_box(i, gamma_cell, soft_threshold, num_cells)) return;

    float V = volumes[i];
    if (V < 1e-6f) return;
    float invV = 1.0f / V;
    float mx = Cx[i] * invV;
    float my = Cy[i] * invV;
    float varx = Cxx[i] * invV - mx * mx;
    float vary = Cyy[i] * invV - my * my;
    if (varx < 0.0f) varx = 0.0f;
    if (vary < 0.0f) vary = 0.0f;
    float sigx = sqrtf(varx);
    float sigy = sqrtf(vary);

    // Estimated visible cell extent: sigma + lambda/2 along each axis.
    // sigma alone is the phi^2-weighted second moment radius; the visible
    // edge of a phase-field cell sits ~lambda/2 outside that.
    float pad = 0.5f * lambda;
    int hwx = (int)ceilf(sigx + pad);
    int hwy = (int)ceilf(sigy + pad);

    int cx = origin[2*i + 0] + (int)roundf(mx);
    int cy = origin[2*i + 1] + (int)roundf(my);

    uchar4 col = make_uchar4(255, 165, 0, 255);  // orange

    // Dashed rectangle centered on (cx, cy) with half-widths (hwx, hwy).
    for (int x = -hwx; x <= hwx; x += 2) {
        put_pixel(pixels, cx + x, cy - hwy, Nx, Ny, col);
        put_pixel(pixels, cx + x, cy + hwy, Nx, Ny, col);
    }
    for (int y = -hwy; y <= hwy; y += 2) {
        put_pixel(pixels, cx - hwx, cy + y, Nx, Ny, col);
        put_pixel(pixels, cx + hwx, cy + y, Nx, Ny, col);
    }
}

// ---------------------------------------------------------------------------
// Visualizer
// ---------------------------------------------------------------------------
Visualizer::~Visualizer() { shutdown(); }

bool Visualizer::init(int field_width, int field_height, const char* title) {
    if (initialized) return true;

    tex_width = field_width;
    tex_height = field_height;

    if (!glfwInit()) {
        fprintf(stderr, "[viz] GLFW init failed\n");
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE);

    float aspect = (float)field_width / (float)field_height;
    int win_h = (field_height > 900) ? 900 : field_height;
    int win_w = (int)(win_h * aspect);
    if (win_w > 1600) { win_w = 1600; win_h = (int)(win_w / aspect); }
    // Ensure a minimum size so a small field doesn't open a tiny window
    // that's easy to miss.
    if (win_h < 600) { win_h = 600; win_w = (int)(win_h * aspect); }

    window = glfwCreateWindow(win_w, win_h, title, nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "[viz] window creation failed\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // no vsync — don't throttle the sim.
    glfwShowWindow(window);
    glfwFocusWindow(window);

    // ESC closes the window.
    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int) {
        if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(w, GLFW_TRUE);
    });

    glGenTextures(1, &gl_texture);
    glBindTexture(GL_TEXTURE_2D, gl_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex_width, tex_height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaError_t err = cudaGraphicsGLRegisterImage(
        &cuda_resource, gl_texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        fprintf(stderr, "[viz] CUDA-GL register failed: %s\n",
                cudaGetErrorString(err));
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }

    initialized = true;
    printf("[viz] %dx%d window (field %dx%d)\n",
           win_w, win_h, tex_width, tex_height);
    return true;
}

void Visualizer::update(const float* d_S,
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
                        float  bbox_K,
                        float  lambda,
                        float  soft_gamma_threshold) {
    if (!initialized || !d_S) return;

    cudaGraphicsMapResources(1, &cuda_resource);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);

    // Persistent scratch buffer for RGBA pixels (small, GPU-resident).
    static uchar4* d_pixels = nullptr;
    static int     alloc_size = 0;
    int needed = Nx * Ny;
    if (needed > alloc_size) {
        if (d_pixels) cudaFree(d_pixels);
        cudaMalloc(&d_pixels, needed * sizeof(uchar4));
        alloc_size = needed;
    }

    // 1. Base colormap from S.
    {
        dim3 block(16, 16);
        dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);
        kernel_colormap<<<grid, block>>>(d_pixels, d_S, Nx, Ny);
    }

    // 2. Soft-cell red tint over the active rect of every soft cell.
    if (d_phi_in && d_origin && d_rect && d_gamma && num_cells > 0) {
        dim3 block(16, 16);
        // The active rect is at most TILE_T x TILE_T; 320 / 16 = 20 blocks
        // per axis. Z dim iterates cells.
        dim3 grid((TILE_T + 15) / 16, (TILE_T + 15) / 16, num_cells);
        kernel_tint_soft<<<grid, block>>>(
            d_pixels, Nx, Ny, d_phi_in, d_origin, d_rect, d_gamma,
            soft_gamma_threshold);
    }

    // 3. Soft-cell bbox (dashed cyan, only for soft cells).
    if (d_origin && d_rect && d_gamma && num_cells > 0) {
        int threads = 256;
        int blocks  = (num_cells + threads - 1) / threads;
        kernel_draw_soft_bbox<<<blocks, threads>>>(
            d_pixels, Nx, Ny, d_origin, d_rect, d_gamma,
            soft_gamma_threshold, num_cells);
    }

    // 3b. Soft-cell sigma box (dashed orange) -- the unaligned/unclamped
    // half-width k_rebind would compute from current second moments. Lets
    // you compare the actual rect (cyan, possibly stale between rebinds)
    // against the live shape extent.
    if (d_origin && d_Cx && d_Cy && d_Cxx && d_Cyy && d_volumes
        && d_gamma && d_tgt_radius && num_cells > 0) {
        int threads = 256;
        int blocks  = (num_cells + threads - 1) / threads;
        kernel_draw_sigma_box<<<blocks, threads>>>(
            d_pixels, Nx, Ny, d_origin, d_Cx, d_Cy, d_Cxx, d_Cyy,
            d_volumes, d_gamma, d_tgt_radius,
            bbox_K, lambda, soft_gamma_threshold, num_cells);
    }

    // 4. Velocity arrows on every cell.
    if (d_origin && d_Cx && d_volumes && d_vx && num_cells > 0) {
        int threads = 256;
        int blocks  = (num_cells + threads - 1) / threads;
        const float arrow_scale = 600.0f;  // scaled for v ~ 0.02
        kernel_draw_arrows<<<blocks, threads>>>(
            d_pixels, Nx, Ny, d_origin, d_Cx, d_Cy, d_volumes,
            d_vx, d_vy, num_cells, arrow_scale);
    }

    cudaMemcpy2DToArray(array, 0, 0, d_pixels,
                        Nx * sizeof(uchar4),
                        Nx * sizeof(uchar4), Ny,
                        cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_resource);

    // Update window title with current sim time.
    char title[64];
    snprintf(title, sizeof(title), "cell_sim live  t=%.2f  N=%d", current_time, num_cells);
    glfwSetWindowTitle(window, title);

    int win_w, win_h;
    glfwGetFramebufferSize(window, &win_w, &win_h);
    glViewport(0, 0, win_w, win_h);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, gl_texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1,  1);
    glTexCoord2f(0, 0); glVertex2f(-1,  1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool Visualizer::should_close() const {
    if (!initialized || !window) return true;
    return glfwWindowShouldClose(window);
}

void Visualizer::shutdown() {
    if (!initialized) return;
    if (cuda_resource) {
        cudaGraphicsUnregisterResource(cuda_resource);
        cuda_resource = nullptr;
    }
    if (gl_texture) {
        glDeleteTextures(1, &gl_texture);
        gl_texture = 0;
    }
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
    initialized = false;
}

} // namespace cellsim

#endif // ENABLE_VISUALIZER
