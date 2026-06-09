use crate::ensure_burn_setup;
use brush_process::{DataSource, RunningProcess, create_process};
use brush_process::message::ProcessMessage;
use brush_render::{TextureMode, camera::Camera, gaussian_splats::Splats};
use brush_process::slot::Slot;
use burn::Tensor;
use burn::tensor::TensorData;
use glam::{Quat, UVec2, Vec2, Vec3};
use std::ffi::{CStr, c_char};
use std::f32::consts::PI;
use std::panic::AssertUnwindSafe;
use tokio_stream::StreamExt;

/// Default 3/4 (angled) home orientation, so scenes don't open dead-on.
const HOME_YAW: f32 = 0.6;    // ~34° around
const HOME_PITCH: f32 = 0.45; // ~26° looking down

/// Orbit-style camera state (position computed from yaw/pitch/distance/center).
struct OrbitCamera {
    center: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    fov_y: f32,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            center: Vec3::ZERO,
            distance: 3.0,
            yaw: HOME_YAW,
            pitch: HOME_PITCH,
            fov_y: 60.0_f32.to_radians(),
        }
    }

    fn to_camera(&self, width: u32, height: u32) -> Camera {
        let pitch = self.pitch.clamp(-PI / 2.0 + 0.01, PI / 2.0 - 0.01);
        // Brush's camera convention: +Z is forward, +Y is down (COLMAP).
        // Orbit yaw around world -Y (which is visual up in Y-down convention).
        let rotation = Quat::from_axis_angle(Vec3::NEG_Y, self.yaw)
            * Quat::from_rotation_x(pitch);
        // Camera sits behind the focal point along -Z so it looks toward center.
        let position = self.center - rotation * Vec3::Z * self.distance;
        let fov_y = self.fov_y as f64;
        // Derive fov_x from fov_y via the pinhole model: keep focal_y fixed,
        // then compute the horizontal FOV that fills the given width.
        // focal = (px/2) / tan(fov/2)  →  fov = 2 * atan((px/2) / focal)
        let focal_y = (height.max(1) as f64 * 0.5) / (fov_y * 0.5).tan();
        let fov_x = 2.0 * ((width.max(1) as f64 * 0.5) / focal_y).atan();
        Camera {
            position,
            rotation,
            fov_x,
            fov_y,
            center_uv: Vec2::new(0.5, 0.5),
            ..Camera::default()
        }
    }
}

/// Opaque viewer handle exposed through the C FFI.
pub struct BrushViewer {
    runtime: tokio::runtime::Runtime,
    splat_view: Slot<Splats>,
    orbit: OrbitCamera,
    background: Vec3,
    splat_scale: Option<f32>,
    width: u32,
    height: u32,
    /// Splat center positions [x,y,z,...], cached at load for click-picking.
    splat_means: Vec<f32>,
    /// Initial auto-framed pivot/distance, for the "Home" reset.
    home_center: Vec3,
    home_distance: f32,
    /// MIP anti-aliasing (reduces sub-pixel splat flicker when zoomed out).
    mip: bool,
    /// Scene axis-aligned bounds (from splat means), for normalized crop mapping.
    scene_min: Vec3,
    scene_max: Vec3,
    /// Active crop box in world coords; None = show everything.
    crop: Option<(Vec3, Vec3)>,
    /// Cached splats filtered to the crop box (rebuilt when the crop changes).
    cropped: Option<Splats>,
}

/// Stats returned per-frame.
#[repr(C)]
pub struct BrushViewerStats {
    pub num_splats: u32,
    pub has_splats: bool,
}

/// Create a viewer that asynchronously loads a .ply splat file.
///
/// Returns a heap-allocated `BrushViewer` on success, null on failure.
/// The caller must eventually call `brush_viewer_destroy` to free it.
///
/// # Safety
/// `file_path` must be a valid, null-terminated C string for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_create(file_path: *const c_char) -> *mut BrushViewer {
    if file_path.is_null() {
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(|| {
        let path_str = unsafe { CStr::from_ptr(file_path).to_string_lossy().into_owned() };

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        let (splat_view, auto_orbit, splat_means) = runtime.block_on(async {
            ensure_burn_setup().await;

            let source = DataSource::Path(path_str);
            let RunningProcess { mut stream, splat_view } =
                create_process(source, async |cfg| Some(cfg));

            // Drive stream to completion synchronously.
            while let Some(msg) = stream.next().await {
                match msg {
                    Ok(ProcessMessage::DoneLoading) => break,
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("[BrushViewer] stream error: {e}");
                        break;
                    }
                }
            }

            // Auto-fit camera from splat positions. Reads means from GPU → CPU to
            // compute the scene bounding box, then sets center and distance. The
            // means are kept for click-to-recenter picking.
            let mut means_vec: Vec<f32> = Vec::new();
            let auto_orbit = if let Some(splats) = splat_view.latest() {
                if splats.num_splats() > 0 {
                    let means = splats.means(); // [N, 3]
                    match means.into_data_async().await {
                        Ok(data) => match data.into_vec::<f32>() {
                            Ok(vals) => {
                                let n = vals.len() / 3;
                                let mut min = [f32::INFINITY; 3];
                                let mut max = [f32::NEG_INFINITY; 3];
                                for i in 0..n {
                                    for j in 0..3 {
                                        let v = vals[i * 3 + j];
                                        if v < min[j] { min[j] = v; }
                                        if v > max[j] { max[j] = v; }
                                    }
                                }
                                let center = Vec3::new(
                                    (min[0] + max[0]) * 0.5,
                                    (min[1] + max[1]) * 0.5,
                                    (min[2] + max[2]) * 0.5,
                                );
                                let extent = Vec3::new(
                                    max[0] - min[0],
                                    max[1] - min[1],
                                    max[2] - min[2],
                                );
                                // Stand back 1.5× the scene's half-diagonal
                                let distance = (extent.length() * 0.75).max(0.5);
                                eprintln!("[BrushViewer] auto-fit: center={center:?}, dist={distance:.3}");
                                means_vec = vals;
                                Some((center, distance))
                            }
                            Err(_) => None,
                        },
                        Err(_) => None,
                    }
                } else {
                    None
                }
            } else {
                None
            };

            (splat_view, auto_orbit, means_vec)
        });

        let mut orbit = OrbitCamera::new();
        if let Some((center, distance)) = auto_orbit {
            orbit.center = center;
            orbit.distance = distance;
        }
        let home_center = orbit.center;
        let home_distance = orbit.distance;
        let (scene_min, scene_max) = bounds_of_means(&splat_means);

        Box::into_raw(Box::new(BrushViewer {
            runtime,
            splat_view,
            orbit,
            background: Vec3::new(0.2, 0.2, 0.2),
            splat_scale: None,
            width: 800,
            height: 600,
            splat_means,
            home_center,
            home_distance,
            mip: true,
            scene_min,
            scene_max,
            crop: None,
            cropped: None,
        }))
    });

    result.unwrap_or(std::ptr::null_mut())
}

/// Destroy a viewer created by `brush_viewer_create`.
///
/// # Safety
/// `viewer` must have been returned by `brush_viewer_create` and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_destroy(viewer: *mut BrushViewer) {
    if !viewer.is_null() {
        drop(unsafe { Box::from_raw(viewer) });
    }
}

/// Returns true once at least one splat frame has been loaded.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_has_splats(viewer: *const BrushViewer) -> bool {
    if viewer.is_null() {
        return false;
    }
    !unsafe { &*viewer }.splat_view.is_empty()
}

/// Update the render surface size.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_set_size(
    viewer: *mut BrushViewer,
    width: u32,
    height: u32,
) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    v.width = width.max(1);
    v.height = height.max(1);
}

/// Set the vertical field-of-view in degrees.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_set_fov(viewer: *mut BrushViewer, fov_degrees: f32) {
    if viewer.is_null() {
        return;
    }
    unsafe { &mut *viewer }.orbit.fov_y = fov_degrees.to_radians();
}

/// Orbit the camera around the scene center.
/// `delta_x` and `delta_y` are in radians.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_orbit(
    viewer: *mut BrushViewer,
    delta_x: f32,
    delta_y: f32,
) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    v.orbit.yaw += delta_x;
    v.orbit.pitch += delta_y;
}

/// Pan the camera (translate both position and center).
/// `delta_x` and `delta_y` are in **screen pixels**; they are scaled to world
/// units based on the pivot distance and field of view, so panning feels the
/// same whether the scene is tiny or huge.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_pan(
    viewer: *mut BrushViewer,
    delta_x: f32,
    delta_y: f32,
) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    let cam = v.orbit.to_camera(v.width, v.height);
    let right = cam.rotation * Vec3::X;
    // Brush: camera +Y is down, so -Y is visual up.
    let up = cam.rotation * Vec3::NEG_Y;
    // World units spanned by one pixel at the pivot plane: the visible height
    // there is 2 * distance * tan(fov_y/2), divided across the viewport height.
    let world_per_pixel =
        (2.0 * v.orbit.distance * (v.orbit.fov_y * 0.5).tan()) / (v.height.max(1) as f32);
    v.orbit.center -= right * (delta_x * world_per_pixel) + up * (delta_y * world_per_pixel);
}

/// Recenter the orbit pivot on whatever lies under the given screen pixel.
/// `px`/`py` are in pixels from the top-left of the rendered image. The pivot
/// is placed along the ray through that pixel at the current viewing distance,
/// which brings the clicked spot to the middle of the view.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_recenter_at(viewer: *mut BrushViewer, px: f32, py: f32) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    let w = v.width.max(1) as f32;
    let h = v.height.max(1) as f32;
    let cam = v.orbit.to_camera(v.width, v.height);

    let thfy = (v.orbit.fov_y * 0.5).tan();
    let thfx = (w / h) * thfy;
    // Pixel → normalized device coords. +X right, +Y down (Brush convention).
    let ndc_x = ((px + 0.5) / w) * 2.0 - 1.0;
    let ndc_y = ((py + 0.5) / h) * 2.0 - 1.0;
    let cam_ray = Vec3::new(ndc_x * thfx, ndc_y * thfy, 1.0).normalize();
    let dir = (cam.rotation * cam_ray).normalize();
    let origin = cam.position;

    // Pick the front-most splat whose center lies near the click ray, so the
    // pivot lands on the actual surface rather than floating at a guessed depth.
    // Acceptance cone ≈ a few pixels wide at the splat's depth.
    let cone_per_unit = (2.0 * thfy / h) * 6.0; // ~6px tolerance
    let mut best_t: Option<f32> = None;
    let mut fallback_t: Option<f32> = None; // nearest-to-ray if none within cone
    let mut fallback_perp = f32::INFINITY;

    let n = v.splat_means.len() / 3;
    for i in 0..n {
        let m = Vec3::new(
            v.splat_means[i * 3],
            v.splat_means[i * 3 + 1],
            v.splat_means[i * 3 + 2],
        );
        let rel = m - origin;
        let t = rel.dot(dir);
        if t <= 0.01 {
            continue; // behind the camera
        }
        let perp = (rel - dir * t).length();
        if perp < t * cone_per_unit {
            if best_t.map_or(true, |b| t < b) {
                best_t = Some(t);
            }
        } else if perp < fallback_perp {
            fallback_perp = perp;
            fallback_t = Some(t);
        }
    }

    // Depth of the picked surface point along the ray; fall back to the current
    // viewing distance if the scene has no cached means.
    let t_pick = best_t.or(fallback_t).unwrap_or(v.orbit.distance);
    let picked = origin + dir * t_pick;

    // Re-aim the camera at the picked point WITHOUT moving it: keep `origin`
    // fixed, point along `dir`, set distance to the surface depth. The orbit
    // model recomputes position = center - forward*distance == origin.
    v.orbit.yaw = (-dir.x).atan2(dir.z);
    v.orbit.pitch = (-dir.y).asin();
    v.orbit.distance = t_pick.max(0.01);
    v.orbit.center = picked;
}

/// Zoom the camera by adjusting its distance from the center.
/// Positive `delta` zooms in.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_zoom(viewer: *mut BrushViewer, delta: f32) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    v.orbit.distance = (v.orbit.distance * (1.0 - delta * 0.1)).max(0.01);
}

/// Reset the camera to the initial auto-framed view (pivot, distance, angles).
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_reset_view(viewer: *mut BrushViewer) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    v.orbit.center = v.home_center;
    v.orbit.distance = v.home_distance;
    v.orbit.yaw = HOME_YAW;
    v.orbit.pitch = HOME_PITCH;
}

/// Fill `out_xyz` (9 floats) with the camera-space directions of the world
/// X, Y, Z axes: for each axis, (right·e, up·e, forward·e). The first two give
/// the on-screen 2D direction; the third is depth (negative = toward viewer).
///
/// # Safety
/// `viewer` must be valid; `out_xyz` must point to 9 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_get_axes(viewer: *const BrushViewer, out_xyz: *mut f32) {
    if viewer.is_null() || out_xyz.is_null() {
        return;
    }
    let v = unsafe { &*viewer };
    let cam = v.orbit.to_camera(v.width, v.height);
    let right = cam.rotation * Vec3::X;
    let up = cam.rotation * Vec3::NEG_Y; // visual up (Brush camera +Y is down)
    let fwd = cam.rotation * Vec3::Z;
    let out = unsafe { std::slice::from_raw_parts_mut(out_xyz, 9) };
    for (i, e) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
        out[i * 3] = e.dot(right);
        out[i * 3 + 1] = e.dot(up);
        out[i * 3 + 2] = e.dot(fwd);
    }
}

/// Set background color (each component 0.0–1.0).
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_set_background(
    viewer: *mut BrushViewer,
    r: f32,
    g: f32,
    b: f32,
) {
    if viewer.is_null() {
        return;
    }
    unsafe { &mut *viewer }.background = Vec3::new(r, g, b);
}

/// Enable/disable MIP anti-aliasing (reduces flicker when zoomed out).
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_set_mip(viewer: *mut BrushViewer, enabled: bool) {
    if viewer.is_null() {
        return;
    }
    unsafe { &mut *viewer }.mip = enabled;
}

/// Axis-aligned bounds of a flat `[x,y,z, x,y,z, ...]` means array.
fn bounds_of_means(means: &[f32]) -> (Vec3, Vec3) {
    if means.len() < 3 {
        return (Vec3::splat(-1.0), Vec3::splat(1.0));
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for chunk in means.chunks_exact(3) {
        let p = Vec3::new(chunk[0], chunk[1], chunk[2]);
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

/// Indices of splats whose center lies inside the world-space box.
fn keep_indices_in_box(means: &[f32], wmin: Vec3, wmax: Vec3) -> Vec<i32> {
    let mut keep = Vec::new();
    for (i, chunk) in means.chunks_exact(3).enumerate() {
        let (x, y, z) = (chunk[0], chunk[1], chunk[2]);
        if x >= wmin.x && x <= wmax.x && y >= wmin.y && y <= wmax.y && z >= wmin.z && z <= wmax.z {
            keep.push(i as i32);
        }
    }
    keep
}

/// Build a Splats containing only the gaussians inside the box (GPU `select`).
/// Returns None if the box excludes everything.
fn crop_splats(splats: &Splats, means: &[f32], wmin: Vec3, wmax: Vec3) -> Option<Splats> {
    let keep = keep_indices_in_box(means, wmin, wmax);
    if keep.is_empty() {
        return None;
    }
    let device = splats.device();
    let idx = Tensor::from_data(TensorData::new(keep.clone(), [keep.len()]), &device);
    let mut out = splats.clone();
    out.transforms = out.transforms.map(|t| t.select(0, idx.clone()));
    out.sh_coeffs = out.sh_coeffs.map(|c| c.select(0, idx.clone()));
    out.raw_opacities = out.raw_opacities.map(|o| o.select(0, idx.clone()));
    out.min_scale = out.min_scale.map(|f| f.select(0, idx.clone()));
    Some(out)
}

/// Set the crop box. `fmin`/`fmax` are 0–1 fractions of the scene bounds per
/// axis; `enabled = false` clears the crop. Lets you trim stray "floater"
/// gaussians around the edges of a capture.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_set_crop(
    viewer: *mut BrushViewer,
    fxmin: f32, fymin: f32, fzmin: f32,
    fxmax: f32, fymax: f32, fzmax: f32,
    enabled: bool,
) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    if !enabled {
        v.crop = None;
        v.cropped = None;
        return;
    }
    let size = v.scene_max - v.scene_min;
    let wmin = v.scene_min + Vec3::new(fxmin, fymin, fzmin) * size;
    let wmax = v.scene_min + Vec3::new(fxmax, fymax, fzmax) * size;
    v.crop = Some((wmin, wmax));
    v.cropped = v
        .splat_view
        .latest()
        .and_then(|s| crop_splats(&s, &v.splat_means, wmin, wmax));
}

/// Export the current splats (cropped, if a crop is active) to a `.ply` file.
/// Returns true on success.
///
/// # Safety
/// `viewer` must be valid; `out_path` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_export_cropped(
    viewer: *const BrushViewer,
    out_path: *const c_char,
) -> bool {
    if viewer.is_null() || out_path.is_null() {
        return false;
    }
    let v = unsafe { &*viewer };
    let path = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };
    let splats = if v.crop.is_some() {
        v.cropped.clone()
    } else {
        v.splat_view.latest()
    };
    let Some(splats) = splats else { return false };

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        v.runtime.block_on(async move {
            match brush_serde::splat_to_ply(splats).await {
                Ok(bytes) => std::fs::write(&path, bytes).is_ok(),
                Err(e) => {
                    eprintln!("[BrushViewer] export failed: {e}");
                    false
                }
            }
        })
    }));
    result.unwrap_or(false)
}

/// Set splat scale multiplier. Pass 1.0 for original size, 0.0 to use the default.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_set_splat_scale(viewer: *mut BrushViewer, scale: f32) {
    if viewer.is_null() {
        return;
    }
    let v = unsafe { &mut *viewer };
    v.splat_scale = if scale <= 0.0 { None } else { Some(scale) };
}

/// Get current stats.
///
/// # Safety
/// `viewer` must be a valid pointer returned by `brush_viewer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_get_stats(viewer: *const BrushViewer) -> BrushViewerStats {
    if viewer.is_null() {
        return BrushViewerStats { num_splats: 0, has_splats: false };
    }
    let v = unsafe { &*viewer };
    match v.splat_view.latest() {
        Some(splats) => BrushViewerStats {
            num_splats: splats.num_splats(),
            has_splats: true,
        },
        None => BrushViewerStats { num_splats: 0, has_splats: false },
    }
}

/// Render the current scene into a caller-provided RGBA buffer.
///
/// `rgba_out` must point to at least `width * height * 4` bytes.
/// Returns `true` on success, `false` if no splats are loaded yet or on error.
///
/// This function blocks until the GPU render and readback are complete.
///
/// # Safety
/// - `viewer` must be a valid pointer returned by `brush_viewer_create`.
/// - `rgba_out` must point to at least `width * height * 4` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_viewer_render_frame(
    viewer: *const BrushViewer,
    width: u32,
    height: u32,
    rgba_out: *mut u8,
) -> bool {
    if viewer.is_null() || rgba_out.is_null() {
        return false;
    }
    let v = unsafe { &*viewer };
    // Render the cropped subset when a crop is active, else the full scene.
    let chosen = if v.crop.is_some() {
        v.cropped.clone()
    } else {
        v.splat_view.latest()
    };
    let Some(mut splats) = chosen else {
        return false;
    };
    // Toggle MIP anti-aliasing on the per-frame splats clone.
    splats.render_mip = v.mip;

    let w = width.max(1);
    let h = height.max(1);
    let img_size = UVec2::new(w, h);
    let camera = v.orbit.to_camera(w, h);
    let background = v.background;
    let splat_scale = v.splat_scale;

    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v.runtime.block_on(async move {
            let (tensor, aux) = brush_render::render_splats(
                splats,
                &camera,
                img_size,
                background,
                splat_scale,
                TextureMode::Float,
            )
            .await;

            eprintln!("[BrushViewer] num_visible={} num_intersections={} img={}x{}",
                aux.num_visible, aux.num_intersections, img_size.x, img_size.y);

            let data = match tensor.into_data_async().await {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[BrushViewer] into_data_async failed: {e}");
                    panic!("readback failed");
                }
            };
            let floats = match data.into_vec::<f32>() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[BrushViewer] into_vec::<f32> failed: {e}");
                    panic!("vec conversion failed");
                }
            };
            eprintln!("[BrushViewer] render ok: {} floats, first 8: {:?}", floats.len(), &floats[..8.min(floats.len())]);

            unsafe {
                let out = std::slice::from_raw_parts_mut(rgba_out, (w * h * 4) as usize);
                for (i, &f) in floats.iter().take(out.len()).enumerate() {
                    // Force alpha=255 so non-splat background pixels are opaque
                    out[i] = if i % 4 == 3 {
                        255
                    } else {
                        (f * 255.0).clamp(0.0, 255.0) as u8
                    };
                }
            }
        });
    }));

    if ok.is_err() {
        eprintln!("[BrushViewer] render_frame panicked (caught)");
    }
    ok.is_ok()
}
