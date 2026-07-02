fn main() {
  // Emits `PyPy`/`GraalPy`/`Py_LIMITED_API`/... cfg flags (plus their
  // check-cfg declarations) so interpreter-specific modules can be gated.
  pyo3_build_config::use_pyo3_cfgs();
}
