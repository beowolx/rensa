pub fn read_env_usize_clamped(
  key: &str,
  default: usize,
  min: usize,
  max: usize,
) -> usize {
  std::env::var(key)
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .map_or(default, |parsed| parsed.clamp(min, max))
}

pub fn read_env_f64_clamped(
  key: &str,
  default: f64,
  min: f64,
  max: f64,
) -> f64 {
  std::env::var(key)
    .ok()
    .and_then(|value| value.parse::<f64>().ok())
    .map_or(default, |parsed| parsed.clamp(min, max))
}
