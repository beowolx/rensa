use std::env;
use std::path::PathBuf;

fn main() {
  let crate_dir = PathBuf::from(
    env::var("CARGO_MANIFEST_DIR")
      .expect("cargo must set CARGO_MANIFEST_DIR for build scripts"),
  );
  let header_path = crate_dir.join("include").join("rensa_ffi.h");
  let config_path = crate_dir.join("cbindgen.toml");

  println!("cargo:rerun-if-changed=src/lib.rs");
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed={}", config_path.display());

  std::fs::create_dir_all(
    header_path
      .parent()
      .expect("generated header path must have a parent directory"),
  )
  .expect("failed to create include directory for generated header");

  let config = cbindgen::Config::from_file(&config_path)
    .expect("failed to load cbindgen.toml for rensa-ffi");
  cbindgen::Builder::new()
    .with_crate(crate_dir)
    .with_config(config)
    .generate()
    .expect("failed to generate rensa-ffi C header with cbindgen")
    .write_to_file(header_path);
}
