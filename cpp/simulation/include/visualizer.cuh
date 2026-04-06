#pragma once

#ifdef ENABLE_VISUALIZER

namespace cellsim {

class Visualizer {
public:
  Visualizer();
  ~Visualizer();

  // Initialize window and GL/CUDA interop. Returns false if window creation fails.
  bool init(int field_width, int field_height, const char *title = "Cell Simulation");

  // Update the displayed image from a GPU sum field. Non-blocking.
  // sum_field: device pointer to float[Ny * Nx] (the composite phi^2 field)
  void update(const float *d_sum_field, int Nx, int Ny);

  // Update with overlays (velocity arrows + subdomain boxes + time text)
  void update(const float *d_sum_field, int Nx, int Ny,
              const float *d_centroids_x, const float *d_centroids_y,
              const float *d_velocities_x, const float *d_velocities_y,
              const int *d_offsets_x, const int *d_offsets_y,
              const int *d_widths, const int *d_heights,
              const float *d_second_moment_x, const float *d_second_moment_y,
              const float *d_volumes, float dA,
              int num_cells, float current_time,
              bool show_arrows, bool show_bboxes);

  // Poll events and check if window should close
  bool should_close() const;

  // Process pending window events (call periodically)
  void poll_events();

  // Cleanup
  void shutdown();

  bool is_initialized() const { return initialized; }
  bool show_arrows = true;
  bool show_bboxes = true;
  int bbox_cell_id = 0;       // Which cell to show bbox for (cycle with up/down arrows)

private:
  struct GLFWwindow *window;
  unsigned int gl_texture;
  struct cudaGraphicsResource *cuda_resource;
  bool initialized;
  int tex_width, tex_height;
};

} // namespace cellsim

#endif // ENABLE_VISUALIZER
