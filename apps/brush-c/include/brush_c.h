#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Viewer API — load and render Gaussian splat .ply files
// ---------------------------------------------------------------------------

/// Opaque handle to a viewer instance.
typedef struct BrushViewer BrushViewer;

/// Per-frame stats.
typedef struct {
    uint32_t num_splats;
    bool has_splats;
} BrushViewerStats;

/// Create a viewer that asynchronously loads the .ply file at `file_path`.
/// Returns NULL on failure. Free with `brush_viewer_destroy`.
BrushViewer *brush_viewer_create(const char *file_path);

/// Destroy a viewer and free all associated resources.
void brush_viewer_destroy(BrushViewer *viewer);

/// Returns true once at least one splat frame has been loaded.
bool brush_viewer_has_splats(const BrushViewer *viewer);

/// Set the render surface size in pixels.
void brush_viewer_set_size(BrushViewer *viewer, uint32_t width, uint32_t height);

/// Set the vertical field-of-view in degrees (default: 60).
void brush_viewer_set_fov(BrushViewer *viewer, float fov_degrees);

/// Orbit the camera: delta_x = horizontal (radians), delta_y = vertical (radians).
void brush_viewer_orbit(BrushViewer *viewer, float delta_x, float delta_y);

/// Pan the camera: delta_x/y are in screen pixels, scaled to the scene automatically.
void brush_viewer_pan(BrushViewer *viewer, float delta_x, float delta_y);

/// Recenter the orbit pivot under a screen pixel (px/py from the top-left).
void brush_viewer_recenter_at(BrushViewer *viewer, float px, float py);

/// Zoom: positive delta zooms in (reduces orbit distance by ~10% per unit).
void brush_viewer_zoom(BrushViewer *viewer, float delta);

/// Reset the camera to the initial auto-framed view.
void brush_viewer_reset_view(BrushViewer *viewer);

/// Fill out_xyz (9 floats) with camera-space directions of world X,Y,Z axes:
/// per axis (screen_x, screen_y, depth). For drawing an orientation gizmo.
void brush_viewer_get_axes(const BrushViewer *viewer, float *out_xyz);

/// Project count world points to screen pixels. out receives 3 floats/point
/// (screen_x, screen_y, depth); behind-camera points get NaN x/y, depth<=0.
void brush_viewer_project_points(const BrushViewer *viewer, const float *in_xyz,
                                 uint32_t count, float *out);

/// Set background color (each component 0.0–1.0).
void brush_viewer_set_background(BrushViewer *viewer, float r, float g, float b);

/// Enable/disable MIP anti-aliasing (reduces flicker when zoomed out).
void brush_viewer_set_mip(BrushViewer *viewer, bool enabled);

/// Set the crop box (0–1 fractions of scene bounds per axis). enabled=false clears it.
void brush_viewer_set_crop(BrushViewer *viewer,
                           float fxmin, float fymin, float fzmin,
                           float fxmax, float fymax, float fzmax,
                           bool enabled);

/// Export current splats (cropped if a crop is active) to a .ply. Returns true on success.
bool brush_viewer_export_cropped(const BrushViewer *viewer, const char *out_path);

/// Set splat scale multiplier. Pass 0.0 to use the default (1.0).
void brush_viewer_set_splat_scale(BrushViewer *viewer, float scale);

/// Return current stats.
BrushViewerStats brush_viewer_get_stats(const BrushViewer *viewer);

/// Render the scene into `rgba_out` (caller must provide `width * height * 4` bytes).
/// Blocks until GPU render and CPU readback are complete.
/// Returns true on success, false if no splats are loaded or an error occurs.
bool brush_viewer_render_frame(
    const BrushViewer *viewer,
    uint32_t width,
    uint32_t height,
    uint8_t *rgba_out
);

// ---------------------------------------------------------------------------
// Training API — train a Gaussian splat model from a dataset
// ---------------------------------------------------------------------------

/// Exit code returned by training functions.
typedef enum {
    TrainExitCode_Success = 0,
    TrainExitCode_Error = 1,
} TrainExitCode;

/// Training progress messages delivered via callback.
typedef enum {
    ProgressMessage_NewProcess,
    ProgressMessage_Training,  ///< `iter` field is valid
    ProgressMessage_DoneTraining,
} ProgressMessageKind;

/// Training configuration passed to `train_and_save`.
typedef struct {
    uint32_t total_train_steps;
    uint32_t refine_every;
    uint32_t max_resolution;
    uint32_t export_every;
    uint32_t max_splats;     ///< Cap on total splats (0 = default). Lower → smaller files.
    uint32_t sh_degree;      ///< Spherical-harmonics degree 0–3 (0 = default). Lower → smaller files.
    const char *output_path; ///< NULL means no export path
} TrainOptions;

typedef struct {
    ProgressMessageKind kind;
    uint32_t iter; ///< Only valid when kind == ProgressMessage_Training
} TrainProgressMessage;

typedef void (*TrainProgressCallback)(TrainProgressMessage message, void *user_data);

/// Train a Gaussian splat model from a dataset directory.
/// Blocks until training is complete. Calls `callback` with progress updates.
///
/// `dataset_path`  Path to a COLMAP or Nerfstudio dataset directory.
/// `options`       Pointer to training configuration; must not be NULL.
/// `callback`      Called on each training step and at completion.
/// `user_data`     Passed through to `callback` unchanged.
TrainExitCode train_and_save(
    const char *dataset_path,
    const TrainOptions *options,
    TrainProgressCallback callback,
    void *user_data
);

#ifdef __cplusplus
} // extern "C"
#endif
