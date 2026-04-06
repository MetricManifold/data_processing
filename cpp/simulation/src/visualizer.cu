#ifdef ENABLE_VISUALIZER

// Must include GL before CUDA GL interop
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "visualizer.cuh"
#include <cstdio>
#include <cmath>

namespace cellsim {

//=============================================================================
// Viridis-like colormap (6 control points, linearly interpolated on GPU)
//=============================================================================
__constant__ float3 d_cmap[6] = {
    {0.267f, 0.004f, 0.329f},  // 0.0 - dark purple
    {0.283f, 0.141f, 0.458f},  // 0.2
    {0.127f, 0.357f, 0.510f},  // 0.4
    {0.204f, 0.553f, 0.396f},  // 0.6
    {0.565f, 0.749f, 0.173f},  // 0.8
    {0.992f, 0.906f, 0.145f},  // 1.0 - yellow
};

__device__ uchar4 viridis(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float idx = v * 5.0f;  // 6 control points, 5 intervals
    int lo = (int)idx;
    if (lo >= 5) lo = 4;
    float frac = idx - (float)lo;
    float3 a = d_cmap[lo];
    float3 b = d_cmap[lo + 1];
    float r = a.x + frac * (b.x - a.x);
    float g = a.y + frac * (b.y - a.y);
    float bl = a.z + frac * (b.z - a.z);
    return make_uchar4((unsigned char)(r * 255.0f),
                       (unsigned char)(g * 255.0f),
                       (unsigned char)(bl * 255.0f), 255);
}

//=============================================================================
// Kernel: sum field → RGBA pixels (writes to mapped PBO)
//=============================================================================
__global__ void kernel_colormap(uchar4 *pixels, const float *sum_field,
                                 int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    float v = sum_field[y * Nx + x];
    // sqrt for better contrast (phi^2 field → phi-like visual)
    v = sqrtf(fminf(v, 1.0f));
    pixels[y * Nx + x] = viridis(v);
}

//=============================================================================
// Overlay kernel: draw a single pixel with alpha blending
//=============================================================================
__device__ void draw_pixel(uchar4 *pixels, int px, int py, int Nx, int Ny,
                           uchar4 color) {
    // Wrap periodic
    px = ((px % Nx) + Nx) % Nx;
    py = ((py % Ny) + Ny) % Ny;
    pixels[py * Nx + px] = color;
}

//=============================================================================
// Minimal 5x7 bitmap font for GPU-side text rendering
// Each glyph is 5 columns x 7 rows, packed as 7 bytes (1 bit per pixel col)
//=============================================================================
// Charset: 0-9 . t = e + (space)   — indices 0-15
__constant__ unsigned char d_font[16][7] = {
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
    {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}, // 2
    {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, // 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, // 10: .
    {0x04,0x0E,0x04,0x04,0x04,0x05,0x02}, // 11: t
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, // 12: =
    {0x0E,0x11,0x1F,0x10,0x10,0x11,0x0E}, // 13: e
    {0x04,0x04,0x04,0x1F,0x04,0x04,0x04}, // 14: +
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 15: space
};

//=============================================================================
// Kernel: draw velocity arrows (one thread per cell)
//=============================================================================
__global__ void kernel_draw_arrows(
    uchar4 *pixels, int Nx, int Ny,
    const float *cx, const float *cy,
    const float *vx, const float *vy,
    int num_cells, float arrow_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int x0 = (int)roundf(cx[i]);
    int y0 = (int)roundf(cy[i]);
    float dvx = vx[i] * arrow_scale;
    float dvy = vy[i] * arrow_scale;
    float len = sqrtf(dvx * dvx + dvy * dvy);
    if (len < 1.0f) return;

    uchar4 col = make_uchar4(255, 50, 50, 255);  // red arrows

    // Bresenham-style line from centroid to tip
    int steps = (int)ceilf(len);
    for (int s = 0; s <= steps; ++s) {
        float t = (float)s / fmaxf((float)steps, 1.0f);
        int px = x0 + (int)roundf(dvx * t);
        int py = y0 + (int)roundf(dvy * t);
        draw_pixel(pixels, px, py, Nx, Ny, col);
        // Thicken: draw neighboring pixels
        draw_pixel(pixels, px + 1, py, Nx, Ny, col);
        draw_pixel(pixels, px, py + 1, Nx, Ny, col);
    }

    // Arrowhead: two short lines at ~30 degrees from tip
    float tip_x = dvx, tip_y = dvy;
    float nx1 = -tip_x * 0.3f + tip_y * 0.2f;
    float ny1 = -tip_y * 0.3f - tip_x * 0.2f;
    float nx2 = -tip_x * 0.3f - tip_y * 0.2f;
    float ny2 = -tip_y * 0.3f + tip_x * 0.2f;
    int tx = x0 + (int)roundf(dvx);
    int ty = y0 + (int)roundf(dvy);
    for (int s = 0; s <= 5; ++s) {
        float t = (float)s / 5.0f;
        draw_pixel(pixels, tx + (int)(nx1*t), ty + (int)(ny1*t), Nx, Ny, col);
        draw_pixel(pixels, tx + (int)(nx2*t), ty + (int)(ny2*t), Nx, Ny, col);
    }

    // Draw cell index at centroid
    uchar4 text_col = make_uchar4(255, 255, 255, 255);
    uchar4 text_bg  = make_uchar4(0, 0, 0, 180);
    // Convert index to digits
    char digits[5]; int nd = 0;
    int tmp = i;
    if (tmp == 0) { digits[nd++] = '0'; }
    else { while (tmp > 0) { digits[nd++] = '0' + (tmp % 10); tmp /= 10; } }
    // Draw background
    int tx0 = x0 - nd * 3;
    int ty0 = y0 - 4;
    for (int dy = -1; dy < 8; ++dy)
        for (int dx = -1; dx < nd * 6 + 1; ++dx) {
            int px = tx0 + dx, py = ty0 + dy;
            if (px >= 0 && px < Nx && py >= 0 && py < Ny)
                draw_pixel(pixels, px, py, Nx, Ny, text_bg);
        }
    // Draw digits (reversed order)
    for (int ci = 0; ci < nd; ++ci) {
        int glyph = digits[nd - 1 - ci] - '0';
        for (int row = 0; row < 7; ++row) {
            unsigned char bits = d_font[glyph][row];
            for (int c = 0; c < 5; ++c) {
                if (bits & (0x10 >> c)) {
                    int px = tx0 + ci * 6 + c;
                    int py = ty0 + row;
                    if (px >= 0 && px < Nx && py >= 0 && py < Ny)
                        draw_pixel(pixels, px, py, Nx, Ny, text_col);
                }
            }
        }
    }
}
__global__ void kernel_draw_bboxes(
    uchar4 *pixels, int Nx, int Ny,
    const int *offsets_x, const int *offsets_y,
    const int *widths, const int *heights,
    int num_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int ox = offsets_x[i], oy = offsets_y[i];
    int w = widths[i], h = heights[i];
    uchar4 col = make_uchar4(100, 200, 255, 255);  // light blue

    // Draw top and bottom edges
    for (int x = 0; x < w; x += 2) {  // dashed
        draw_pixel(pixels, ox + x, oy, Nx, Ny, col);
        draw_pixel(pixels, ox + x, oy + h - 1, Nx, Ny, col);
    }
    // Draw left and right edges
    for (int y = 0; y < h; y += 2) {  // dashed
        draw_pixel(pixels, ox, oy + y, Nx, Ny, col);
        draw_pixel(pixels, ox + w - 1, oy + y, Nx, Ny, col);
    }
}

// Draw red box showing raw 3σ extent from second moments (diagnostic)
__global__ void kernel_draw_moment_boxes(
    uchar4 *pixels, int Nx, int Ny,
    const float *centroids_x, const float *centroids_y,
    const float *second_moment_x, const float *second_moment_y,
    const float *volumes, float dA,
    int num_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float phi2_sum = volumes[i] / dA;
    if (phi2_sum < 1.0f) return;
    float sigma_x = sqrtf(fmaxf(second_moment_x[i] / phi2_sum, 1.0f));
    float sigma_y = sqrtf(fmaxf(second_moment_y[i] / phi2_sum, 1.0f));

    int cx = (int)roundf(centroids_x[i]);
    int cy = (int)roundf(centroids_y[i]);
    int half_w = (int)ceilf(3.0f * sigma_x);
    int half_h = (int)ceilf(3.0f * sigma_y);

    uchar4 col = make_uchar4(255, 50, 50, 255);  // red

    // Draw dashed red box edges
    for (int x = -half_w; x <= half_w; x += 2) {
        draw_pixel(pixels, cx + x, cy - half_h, Nx, Ny, col);
        draw_pixel(pixels, cx + x, cy + half_h, Nx, Ny, col);
    }
    for (int y = -half_h; y <= half_h; y += 2) {
        draw_pixel(pixels, cx - half_w, cy + y, Nx, Ny, col);
        draw_pixel(pixels, cx + half_w, cy + y, Nx, Ny, col);
    }
}

__device__ int char_to_font_idx(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c == '.') return 10;
    if (c == 't') return 11;
    if (c == '=') return 12;
    if (c == 'e') return 13;
    if (c == '+') return 14;
    return 15; // space
}

// Draw a string at (x0, y0) with scale factor. Single-threaded kernel.
__global__ void kernel_draw_text(uchar4 *pixels, int Nx, int Ny,
                                  const char *text, int text_len,
                                  int x0, int y0, int scale) {
    // Single thread draws all characters
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uchar4 fg = make_uchar4(255, 255, 255, 255);
    uchar4 bg = make_uchar4(0, 0, 0, 200);

    // Background rectangle
    int total_w = text_len * 6 * scale + 4;
    int total_h = 7 * scale + 4;
    for (int dy = -2; dy < total_h; ++dy) {
        for (int dx = -2; dx < total_w; ++dx) {
            int px = x0 + dx, py = y0 + dy;
            if (px >= 0 && px < Nx && py >= 0 && py < Ny)
                pixels[py * Nx + px] = bg;
        }
    }

    // Characters
    for (int ci = 0; ci < text_len; ++ci) {
        int glyph = char_to_font_idx(text[ci]);
        for (int row = 0; row < 7; ++row) {
            unsigned char bits = d_font[glyph][row];
            for (int col = 0; col < 5; ++col) {
                if (bits & (0x10 >> col)) {
                    for (int sy = 0; sy < scale; ++sy) {
                        for (int sx = 0; sx < scale; ++sx) {
                            int px = x0 + ci * 6 * scale + col * scale + sx;
                            int py = y0 + row * scale + sy;
                            if (px >= 0 && px < Nx && py >= 0 && py < Ny)
                                pixels[py * Nx + px] = fg;
                        }
                    }
                }
            }
        }
    }
}

Visualizer::Visualizer()
    : window(nullptr), gl_texture(0), cuda_resource(nullptr),
      initialized(false), tex_width(0), tex_height(0) {}

Visualizer::~Visualizer() { shutdown(); }

bool Visualizer::init(int field_width, int field_height, const char *title) {
    if (initialized) return true;

    tex_width = field_width;
    tex_height = field_height;

    if (!glfwInit()) {
        fprintf(stderr, "Visualizer: GLFW init failed\n");
        return false;
    }

    // Request OpenGL 2.1 (minimal, just need textures)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    // Window size: cap at 900px, maintain aspect ratio
    float aspect = (float)field_width / (float)field_height;
    int win_h = (field_height > 900) ? 900 : field_height;
    int win_w = (int)(win_h * aspect);
    if (win_w > 1600) { win_w = 1600; win_h = (int)(win_w / aspect); }

    window = glfwCreateWindow(win_w, win_h, title, nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Visualizer: Window creation failed\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // No vsync — don't limit simulation speed

    // Store pointer to this Visualizer for key callback
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, [](GLFWwindow *w, int key, int, int action, int) {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
        auto *vis = static_cast<Visualizer *>(glfwGetWindowUserPointer(w));
        if (key == GLFW_KEY_UP)   vis->bbox_cell_id++;
        if (key == GLFW_KEY_DOWN) vis->bbox_cell_id = (vis->bbox_cell_id > 0) ? vis->bbox_cell_id - 1 : 0;
        if (key == GLFW_KEY_B)    vis->show_bboxes = !vis->show_bboxes;
        if (key == GLFW_KEY_A)    vis->show_arrows = !vis->show_arrows;
    });

    // Create OpenGL texture
    glGenTextures(1, &gl_texture);
    glBindTexture(GL_TEXTURE_2D, gl_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex_width, tex_height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register texture with CUDA
    cudaError_t err = cudaGraphicsGLRegisterImage(
        &cuda_resource, gl_texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        fprintf(stderr, "Visualizer: CUDA-GL register failed: %s\n",
                cudaGetErrorString(err));
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }

    initialized = true;
    printf("Visualizer: %dx%d window (field %dx%d)\n",
           win_w, win_h, tex_width, tex_height);
    return true;
}

void Visualizer::update(const float *d_sum_field, int Nx, int Ny) {
    if (!initialized || !d_sum_field) return;

    // Map CUDA resource → get array
    cudaGraphicsMapResources(1, &cuda_resource);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);

    // We can't write directly to a cudaArray with a kernel easily.
    // Instead, use a temporary device buffer → cudaMemcpy2DToArray.
    // The buffer is small (Nx*Ny*4 bytes) and stays on GPU.
    static uchar4 *d_pixels = nullptr;
    static int alloc_size = 0;
    int needed = Nx * Ny;
    if (needed > alloc_size) {
        if (d_pixels) cudaFree(d_pixels);
        cudaMalloc(&d_pixels, needed * sizeof(uchar4));
        alloc_size = needed;
    }

    // Run colormap kernel
    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);
    kernel_colormap<<<grid, block>>>(d_pixels, d_sum_field, Nx, Ny);

    // Copy pixels to GL texture array (GPU → GPU)
    cudaMemcpy2DToArray(array, 0, 0, d_pixels,
                        Nx * sizeof(uchar4), Nx * sizeof(uchar4), Ny,
                        cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_resource);

    // Render fullscreen quad with texture
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

void Visualizer::update(const float *d_sum_field, int Nx, int Ny,
                        const float *d_centroids_x, const float *d_centroids_y,
                        const float *d_velocities_x, const float *d_velocities_y,
                        const int *d_offsets_x, const int *d_offsets_y,
                        const int *d_widths, const int *d_heights,
                        const float *d_second_moment_x, const float *d_second_moment_y,
                        const float *d_volumes, float dA,
                        int num_cells, float current_time,
                        bool draw_arrows, bool draw_bboxes) {
    if (!initialized || !d_sum_field) return;

    cudaGraphicsMapResources(1, &cuda_resource);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);

    static uchar4 *d_pixels = nullptr;
    static int alloc_size = 0;
    int needed = Nx * Ny;
    if (needed > alloc_size) {
        if (d_pixels) cudaFree(d_pixels);
        cudaMalloc(&d_pixels, needed * sizeof(uchar4));
        alloc_size = needed;
    }

    // 1. Colormap
    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);
    kernel_colormap<<<grid, block>>>(d_pixels, d_sum_field, Nx, Ny);

    // 2. Overlay: bounding boxes for selected cells (cycle with up/down keys)
    if (draw_bboxes && d_offsets_x && d_widths && num_cells > 0) {
        int watch[] = {1, 4, 47, 32, 62};
        for (int k = 0; k < 5; ++k) {
            int bid = watch[k];
            if (bid < num_cells) {
                kernel_draw_bboxes<<<1, 1>>>(
                    d_pixels, Nx, Ny,
                    d_offsets_x + bid, d_offsets_y + bid,
                    d_widths + bid, d_heights + bid, 1);
            }
        }
    }

    // 2b. Red dashed boxes: raw 3σ extent from second moments (same cells as blue boxes)
    if (draw_bboxes && d_second_moment_x && d_volumes && num_cells > 0) {
        int watch[] = {1, 4, 47, 32, 62};
        for (int k = 0; k < 5; ++k) {
            int bid = watch[k];
            if (bid < num_cells) {
                kernel_draw_moment_boxes<<<1, 1>>>(
                    d_pixels, Nx, Ny,
                    d_centroids_x + bid, d_centroids_y + bid,
                    d_second_moment_x + bid, d_second_moment_y + bid,
                    d_volumes + bid, dA, 1);
            }
        }
    }

    // 3. Overlay: velocity arrows
    if (draw_arrows && d_centroids_x && d_velocities_x && num_cells > 0) {
        int threads = 256;
        int blocks_1d = (num_cells + threads - 1) / threads;
        float arrow_scale = 15.0f;  // polarization is unit vector → 15px arrows
        kernel_draw_arrows<<<blocks_1d, threads>>>(
            d_pixels, Nx, Ny, d_centroids_x, d_centroids_y,
            d_velocities_x, d_velocities_y, num_cells, arrow_scale);
    }

    // 4. Overlay: time text (top-left corner)
    {
        char buf[32];
        int len = 0;
        // Format "t=XXXX.XX" on host, copy to device
        buf[len++] = 't';
        buf[len++] = '=';
        // Integer part
        int t_int = (int)current_time;
        char digits[10]; int nd = 0;
        if (t_int == 0) { digits[nd++] = '0'; }
        else { int tmp = t_int; while (tmp > 0) { digits[nd++] = '0' + (tmp % 10); tmp /= 10; } }
        for (int i = nd - 1; i >= 0; --i) buf[len++] = digits[i];

        static char *d_text = nullptr;
        if (!d_text) cudaMalloc(&d_text, 32);
        cudaMemcpy(d_text, buf, len, cudaMemcpyHostToDevice);

        int scale = 2;
        kernel_draw_text<<<1, 1>>>(d_pixels, Nx, Ny, d_text, len,
                                    4, 4, scale);
    }

    cudaMemcpy2DToArray(array, 0, 0, d_pixels,
                        Nx * sizeof(uchar4), Nx * sizeof(uchar4), Ny,
                        cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cuda_resource);

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

void Visualizer::poll_events() {
    if (initialized && window) glfwPollEvents();
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
