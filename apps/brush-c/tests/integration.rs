#![cfg(not(target_family = "wasm"))]

use std::ffi::{CString, c_void};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use brush_c::{
    ProgressMessage, TrainExitCode, TrainOptions, train_and_save,
    brush_viewer_create, brush_viewer_destroy, brush_viewer_has_splats, brush_viewer_render_frame,
};

#[repr(C)]
struct CallbackState {
    call_count: AtomicUsize,
    finished_called: std::sync::atomic::AtomicBool,
}

extern "C" fn test_progress_callback(process_message: ProgressMessage, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: user_data is a pointer to a CallbackState struct
    let state = unsafe { (user_data as *const CallbackState).as_ref().unwrap() };
    state.call_count.fetch_add(1, Ordering::SeqCst);

    match process_message {
        ProgressMessage::NewProcess => {
            println!("FFI Test: Training starting...");
        }
        ProgressMessage::Training { iter } => {
            println!("FFI Test: Training iteration: {iter:.2}%");
        }
        ProgressMessage::DoneTraining => {
            println!("FFI Test: Training finished!");
            state
                .finished_called
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[test]
fn test_train_and_save_ffi_short() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dataset_path = Path::new(manifest_dir)
        .join("tests")
        .join("data")
        .join("test_dataset");

    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();

    let dataset_path_cstr = CString::new(dataset_path.to_str().unwrap()).unwrap();

    let mut callback_state = CallbackState {
        call_count: AtomicUsize::new(0),
        finished_called: std::sync::atomic::AtomicBool::new(false),
    };

    let options = TrainOptions {
        total_train_steps: 10,
        refine_every: 5,
        export_every: 10,
        max_resolution: 50,
        max_splats: 0,
        sh_degree: 0,
        output_path: output_path_cstr.as_ptr(),
    };

    // SAFETY: paths are valid, user_data is valid for lifetime of callback_state
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Success));
    assert!(callback_state.call_count.load(Ordering::SeqCst) > 2);

    let output_files: Vec<_> = fs::read_dir(output_path)
        .unwrap()
        .filter_map(Result::ok)
        .collect();
    assert!(!output_files.is_empty(), "No output file was created");
}

#[test]
fn test_train_and_save_ffi_invalid_path() {
    let invalid_dataset_path = "/path/that/does/not/exist/and/should/fail";
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_invalid_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();

    let dataset_path_cstr = CString::new(invalid_dataset_path).unwrap();
    let mut callback_state = CallbackState {
        call_count: AtomicUsize::new(0),
        finished_called: std::sync::atomic::AtomicBool::new(false),
    };

    let options = TrainOptions {
        total_train_steps: 10,
        refine_every: 5,
        export_every: 10,
        max_resolution: 50,
        max_splats: 0,
        sh_degree: 0,
        output_path: output_path_cstr.as_ptr(),
    };

    // SAFETY: The paths are valid, and the callback state is alive for the duration of the call.
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Error));
}

#[test]
fn test_train_and_save_ffi_null_options() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dataset_path = Path::new(manifest_dir)
        .join("tests")
        .join("data")
        .join("test_dataset");

    let dataset_path_cstr = CString::new(dataset_path.to_str().unwrap()).unwrap();

    let mut callback_state = CallbackState {
        call_count: AtomicUsize::new(0),
        finished_called: std::sync::atomic::AtomicBool::new(false),
    };

    // SAFETY: The paths are valid, and the callback state is alive for the duration of the call.
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            std::ptr::null(),
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Error));
}

#[test]
fn test_viewer_loads_and_renders() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let ply_path = Path::new(manifest_dir)
        .join("tests/data/test_dataset/init.ply");

    let ply_cstr = CString::new(ply_path.to_str().unwrap()).unwrap();

    let viewer = unsafe { brush_viewer_create(ply_cstr.as_ptr()) };
    assert!(!viewer.is_null(), "brush_viewer_create returned null");

    // Poll until splats are loaded (up to 10 s)
    let mut has = false;
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if unsafe { brush_viewer_has_splats(viewer) } {
            has = true;
            break;
        }
    }
    assert!(has, "brush_viewer_has_splats never became true");

    // Render a small frame
    const W: u32 = 64;
    const H: u32 = 64;
    let mut buf = vec![0u8; (W * H * 4) as usize];
    let ok = unsafe { brush_viewer_render_frame(viewer, W, H, buf.as_mut_ptr()) };
    assert!(ok, "brush_viewer_render_frame returned false");

    // Expect at least some non-zero pixels (background is gray 0.2)
    let non_zero = buf.iter().any(|&b| b > 0);
    assert!(non_zero, "rendered frame is all zeros");
    println!("render ok — first 16 bytes: {:?}", &buf[..16]);

    unsafe { brush_viewer_destroy(viewer); }
}

#[test]
fn test_train_and_save_ffi_null_dataset() {
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_invalid_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();

    let options = TrainOptions {
        total_train_steps: 10,
        refine_every: 5,
        export_every: 10,
        max_resolution: 50,
        max_splats: 0,
        sh_degree: 0,
        output_path: output_path_cstr.as_ptr(),
    };

    // SAFETY: The paths are valid, and the callback state is null.
    let status_null_dataset = unsafe {
        train_and_save(
            std::ptr::null(),
            &options,
            test_progress_callback,
            std::ptr::null_mut(),
        )
    };

    assert!(matches!(status_null_dataset, TrainExitCode::Error));
}
