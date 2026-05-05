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
// 5x7 bitmap font for in-frame text overlays. Each glyph = 7 bytes, low 5
// bits per row are the columns (MSB = leftmost). Charset (16 glyphs):
//   0-9, '.', 't', '=', 'N', 'x', ' '
// ---------------------------------------------------------------------------
__constant__ unsigned char d_font[16][7] = {
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, //  0: 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, //  1: 1
    {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}, //  2: 2
    {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, //  3: 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, //  4: 4
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, //  5: 5
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, //  6: 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, //  7: 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, //  8: 8
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, //  9: 9
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, // 10: .
    {0x04,0x0E,0x04,0x04,0x04,0x05,0x02}, // 11: t
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, // 12: =
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, // 13: N
    {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}, // 14: x
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 15: space
};

__device__ static int char_to_font_idx(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c == '.') return 10;
    if (c == 't') return 11;
    if (c == '=') return 12;
    if (c == 'N') return 13;
    if (c == 'x') return 14;
    return 15; // space / unknown
}

// Draw a single glyph at (x0,y0) at integer scale. Used by both the
// host-string and per-cell label kernels.
__device__ static void draw_glyph(uchar4* pixels, int Nx, int Ny,
                                  int glyph_idx, int x0, int y0, int scale,
                                  uchar4 fg) {
    for (int row = 0; row < 7; ++row) {
        unsigned char bits = d_font[glyph_idx][row];
        for (int col = 0; col < 5; ++col) {
            if (!(bits & (0x10 >> col))) continue;
            for (int sy = 0; sy < scale; ++sy)
                for (int sx = 0; sx < scale; ++sx)
                    put_pixel(pixels,
                              x0 + col * scale + sx,
                              y0 + row * scale + sy,
                              Nx, Ny, fg);
        }
    }
}

__device__ static void draw_text_bg(uchar4* pixels, int Nx, int Ny,
                                    int x0, int y0,
                                    int w_px, int h_px, uchar4 bg) {
    for (int dy = -2; dy < h_px + 2; ++dy)
        for (int dx = -2; dx < w_px + 2; ++dx) {
            int px = x0 + dx, py = y0 + dy;
            if (px >= 0 && px < Nx && py >= 0 && py < Ny)
                pixels[py * Nx + px] = bg;
        }
}

// Single-threaded kernel: draw a host-formatted string at (x0,y0).
__global__ static void kernel_draw_text(uchar4* pixels, int Nx, int Ny,
                                        const char* text, int text_len,
                                        int x0, int y0, int scale) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uchar4 fg = make_uchar4(255, 255, 255, 255);
    uchar4 bg = make_uchar4(0,   0,   0,   200);

    int total_w = text_len * 6 * scale;
    int total_h = 7 * scale;
    draw_text_bg(pixels, Nx, Ny, x0, y0, total_w, total_h, bg);

    for (int ci = 0; ci < text_len; ++ci) {
        int g = char_to_font_idx(text[ci]);
        draw_glyph(pixels, Nx, Ny, g,
                   x0 + ci * 6 * scale, y0, scale, fg);
    }
}

// Per-cell kernel: draws "WxH" at the top-left corner of each drawn cell's
// active rect. Same draw policy as the bbox kernels (soft + watch list).
__global__ static void kernel_draw_bbox_size(
    uchar4* pixels, int Nx, int Ny,
    const int* origin, const int* rect,
    const float* gamma_cell,
    float soft_threshold, int num_cells) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;
    if (gamma_cell[i] >= soft_threshold) return;  // soft cells only

    int rx0 = rect[4*i + 0];
    int ry0 = rect[4*i + 1];
    int w   = rect[4*i + 2];
    int h   = rect[4*i + 3];
    int gx  = origin[2*i + 0] + rx0;
    int gy  = origin[2*i + 1] + ry0;

    // Format "WxH" into a small buffer of glyph indices.
    int glyphs[12];
    int n = 0;
    int tmp = w;
    int wd[5]; int wn = 0;
    if (tmp == 0) wd[wn++] = 0;
    else while (tmp > 0) { wd[wn++] = tmp % 10; tmp /= 10; }
    for (int k = wn - 1; k >= 0; --k) glyphs[n++] = wd[k];
    glyphs[n++] = 14; // 'x'
    int hd[5]; int hn = 0;
    tmp = h;
    if (tmp == 0) hd[hn++] = 0;
    else while (tmp > 0) { hd[hn++] = tmp % 10; tmp /= 10; }
    for (int k = hn - 1; k >= 0; --k) glyphs[n++] = hd[k];

    int scale = 1;
    int total_w = n * 6 * scale;
    int total_h = 7 * scale;
    int x0 = gx + 1;
    int y0 = gy - total_h - 3;  // sit just above the bbox top edge

    uchar4 fg = make_uchar4(255, 255, 255, 255);
    uchar4 bg = make_uchar4(0,   0,   0,   180);
    draw_text_bg(pixels, Nx, Ny, x0, y0, total_w, total_h, bg);
    for (int ci = 0; ci < n; ++ci)
        draw_glyph(pixels, Nx, Ny, glyphs[ci],
                   x0 + ci * 6 * scale, y0, scale, fg);
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

    // Single-pixel shaft, supersampled at 2x so the line stays connected
    // for shallow slopes without introducing visible thickness.
    int steps = (int)ceilf(len * 2.0f);
    for (int s = 0; s <= steps; ++s) {
        float t = (float)s / fmaxf((float)steps, 1.0f);
        int px = x0 + (int)roundf(dvx * t);
        int py = y0 + (int)roundf(dvy * t);
        put_pixel(pixels, px, py, Nx, Ny, col);
    }

    // Arrowhead (smaller, single-pixel).
    float nx1 = -dvx * 0.25f + dvy * 0.15f;
    float ny1 = -dvy * 0.25f - dvx * 0.15f;
    float nx2 = -dvx * 0.25f - dvy * 0.15f;
    float ny2 = -dvy * 0.25f + dvx * 0.15f;
    int tx = x0 + (int)roundf(dvx);
    int ty = y0 + (int)roundf(dvy);
    for (int s = 0; s <= 6; ++s) {
        float t = (float)s / 6.0f;
        put_pixel(pixels, tx + (int)roundf(nx1*t), ty + (int)roundf(ny1*t), Nx, Ny, col);
        put_pixel(pixels, tx + (int)roundf(nx2*t), ty + (int)roundf(ny2*t), Nx, Ny, col);
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
    // One normal cell for visual comparison.
    int watch[] = {1};
    for (int k = 0; k < 1; ++k) {
        if (watch[k] < num_cells && i == watch[k]) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Tile (TILE_T x TILE_T) outline. One thread per cell; only drawn cells.
// Dashed magenta rectangle aligned to the cell's tile origin. Shows the
// outer scratch container that bounds where any phi values for this cell
// can ever live (the active rect is a sub-region of this tile).
// ---------------------------------------------------------------------------
__global__ static void kernel_draw_tile_box(
    uchar4* pixels, int Nx, int Ny,
    const int* origin,
    const float* gamma_cell,
    float soft_threshold,
    int num_cells) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;
    if (!should_draw_box(i, gamma_cell, soft_threshold, num_cells)) return;

    int gx = origin[2*i + 0];
    int gy = origin[2*i + 1];
    int w  = TILE_T;
    int h  = TILE_T;

    uchar4 col = make_uchar4(255, 80, 220, 255);  // magenta

    // Long-dash (4 on, 4 off) so it's distinguishable from cyan/orange.
    for (int x = 0; x < w; ++x) {
        if (((x >> 2) & 1) == 0) {
            put_pixel(pixels, gx + x, gy,         Nx, Ny, col);
            put_pixel(pixels, gx + x, gy + h - 1, Nx, Ny, col);
        }
    }
    for (int y = 0; y < h; ++y) {
        if (((y >> 2) & 1) == 0) {
            put_pixel(pixels, gx,         gy + y, Nx, Ny, col);
            put_pixel(pixels, gx + w - 1, gy + y, Nx, Ny, col);
        }
    }
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

    // Orange = 2*sigma along each axis (raw second-moment shape estimate,
    // independent of per-cell K-scaling). Cyan = the actual k_rebind rect
    // including K*sigma + R/2 padding and 16-px alignment, so cyan is
    // always >= orange by the padding+align overhead.
    const float K_show = 2.0f;
    int hwx = (int)ceilf(K_show * sigx);
    int hwy = (int)ceilf(K_show * sigy);
    (void)K;
    (void)lambda;
    (void)tgt_radius;

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
    glfwSwapInterval(1);  // vsync on — prevents tearing. Sim isn't throttled
                          // because update() only blocks on swap when called.
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

    // Non-blocking stream so viz kernels never implicit-sync against the
    // sim's default-stream work. Sync points are explicit via sim_done /
    // viz_done events.
    cudaStreamCreateWithFlags(&viz_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&sim_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&viz_done, cudaEventDisableTiming);

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

    // Frame-skip: if the previous viz frame is still in flight, drop this
    // one. Sim never blocks waiting for viz to catch up.
    if (viz_in_flight) {
        cudaError_t st = cudaEventQuery(viz_done);
        if (st == cudaErrorNotReady) {
            // Pump the window so it stays responsive even when we drop frames.
            glfwPollEvents();
            return;
        }
        viz_in_flight = false;
    }

    // Hand-off from sim → viz: record on default stream (where sim runs),
    // then make the viz stream wait on that event. Sim's default stream is
    // never stalled by viz work — only viz_stream serializes against sim.
    cudaEventRecord(sim_done, 0);
    cudaStreamWaitEvent(viz_stream, sim_done, 0);

    cudaGraphicsMapResources(1, &cuda_resource, viz_stream);
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
        kernel_colormap<<<grid, block, 0, viz_stream>>>(d_pixels, d_S, Nx, Ny);
    }

    // 2. Soft-cell red tint over the active rect of every soft cell.
    if (d_phi_in && d_origin && d_rect && d_gamma && num_cells > 0) {
        dim3 block(16, 16);
        // The active rect is at most TILE_T x TILE_T; 320 / 16 = 20 blocks
        // per axis. Z dim iterates cells.
        dim3 grid((TILE_T + 15) / 16, (TILE_T + 15) / 16, num_cells);
        kernel_tint_soft<<<grid, block, 0, viz_stream>>>(
            d_pixels, Nx, Ny, d_phi_in, d_origin, d_rect, d_gamma,
            soft_gamma_threshold);
    }

    // 3. Soft-cell bbox (dashed cyan, only for soft cells).
    if (d_origin && d_rect && d_gamma && num_cells > 0) {
        int threads = 256;
        int blocks  = (num_cells + threads - 1) / threads;
        kernel_draw_soft_bbox<<<blocks, threads, 0, viz_stream>>>(
            d_pixels, Nx, Ny, d_origin, d_rect, d_gamma,
            soft_gamma_threshold, num_cells);
    }

    // 3a. TILE_T outline (dashed magenta) -- the outer scratch container
    // for each drawn cell. Same draw policy as the cyan/orange boxes
    // (soft cells + watch list).
    if (d_origin && d_gamma && num_cells > 0) {
        int threads = 256;
        int blocks  = (num_cells + threads - 1) / threads;
        kernel_draw_tile_box<<<blocks, threads, 0, viz_stream>>>(
            d_pixels, Nx, Ny, d_origin, d_gamma,
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
        kernel_draw_sigma_box<<<blocks, threads, 0, viz_stream>>>(
            d_pixels, Nx, Ny, d_origin, d_Cx, d_Cy, d_Cxx, d_Cyy,
            d_volumes, d_gamma, d_tgt_radius,
            bbox_K, lambda, soft_gamma_threshold, num_cells);
    }

    // 4. Velocity arrows on every cell.
    if (d_origin && d_Cx && d_volumes && d_vx && num_cells > 0) {
        int threads = 256;
        int blocks  = (num_cells + threads - 1) / threads;
        const float arrow_scale = 300.0f;  // half-size; v ~ 0.02 -> 6 px
        kernel_draw_arrows<<<blocks, threads, 0, viz_stream>>>(
            d_pixels, Nx, Ny, d_origin, d_Cx, d_Cy, d_volumes,
            d_vx, d_vy, num_cells, arrow_scale);
    }

    // 5. Per-soft-cell "WxH" label at top-left of each cyan bbox.
    if (d_origin && d_rect && d_gamma && num_cells > 0) {
        int threads = 256;
        int blocks  = (num_cells + threads - 1) / threads;
        kernel_draw_bbox_size<<<blocks, threads, 0, viz_stream>>>(
            d_pixels, Nx, Ny, d_origin, d_rect, d_gamma,
            soft_gamma_threshold, num_cells);
    }

    // 6. Top-left timestamp overlay "t=NNNN.NN" rendered into the texture.
    {
        char buf[32];
        int t_int  = (int)current_time;
        int t_frac = (int)((current_time - (double)t_int) * 100.0 + 0.5);
        if (t_frac >= 100) { t_int += 1; t_frac -= 100; }
        int len = 0;
        buf[len++] = 't';
        buf[len++] = '=';
        char digits[12]; int nd = 0;
        if (t_int == 0) digits[nd++] = '0';
        else { int tmp = t_int; while (tmp > 0) { digits[nd++] = '0' + (tmp % 10); tmp /= 10; } }
        for (int i = nd - 1; i >= 0; --i) buf[len++] = digits[i];
        buf[len++] = '.';
        buf[len++] = '0' + (t_frac / 10);
        buf[len++] = '0' + (t_frac % 10);

        static char* d_text = nullptr;
        if (!d_text) cudaMalloc(&d_text, 32);
        cudaMemcpyAsync(d_text, buf, len, cudaMemcpyHostToDevice, viz_stream);
        kernel_draw_text<<<1, 1, 0, viz_stream>>>(
            d_pixels, Nx, Ny, d_text, len, 6, 6, /*scale=*/2);
    }

    cudaMemcpy2DToArrayAsync(array, 0, 0, d_pixels,
                             Nx * sizeof(uchar4),
                             Nx * sizeof(uchar4), Ny,
                             cudaMemcpyDeviceToDevice, viz_stream);

    cudaGraphicsUnmapResources(1, &cuda_resource, viz_stream);
    cudaEventRecord(viz_done, viz_stream);

    // GL needs the texture to be consistent before the swap. Sync ONLY the
    // viz stream — sim's default stream keeps running ahead.
    cudaStreamSynchronize(viz_stream);
    viz_in_flight = false;

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

    // Mark next call as having an in-flight frame so that future calls can
    // skip cleanly if the GL/swap pipeline is congested. We've already
    // synced viz_done above so the bookkeeping flag stays consistent: this
    // is a "frame is presented" marker for the next iteration's drop logic
    // (we leave it false because viz_done has already completed). If at a
    // later point we move the sync after swap, set this to true.
    viz_in_flight = false;
}

bool Visualizer::should_close() const {
    if (!initialized || !window) return true;
    return glfwWindowShouldClose(window);
}

void Visualizer::shutdown() {
    if (!initialized) return;
    if (viz_stream) {
        cudaStreamSynchronize(viz_stream);
        cudaStreamDestroy(viz_stream);
        viz_stream = 0;
    }
    if (sim_done) { cudaEventDestroy(sim_done); sim_done = nullptr; }
    if (viz_done) { cudaEventDestroy(viz_done); viz_done = nullptr; }
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
