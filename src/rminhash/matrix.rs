#[derive(Clone)]
pub struct RhoDigestSidecar {
  pub non_empty_counts: Vec<u16>,
  pub source_token_counts: Vec<u16>,
  pub sparse_occupancy_threshold: usize,
  pub sparse_verify_perm: usize,
  pub sparse_verify_signatures: Vec<u32>,
  pub sparse_verify_active: Vec<u8>,
}

impl crate::rminhash::RMinHashDigestMatrix {
  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  #[inline]
  pub(crate) const fn rows(&self) -> usize {
    self.rows
  }

  #[inline]
  pub(crate) fn row(&self, row_index: usize) -> &[u32] {
    let start = row_index * self.num_perm;
    &self.data[start..start + self.num_perm]
  }

  pub(crate) fn rho_non_empty_count(&self, row_index: usize) -> Option<usize> {
    let sidecar = self.rho_sidecar.as_ref()?;
    sidecar
      .non_empty_counts
      .get(row_index)
      .copied()
      .map(usize::from)
  }

  pub(crate) fn rho_source_token_count(
    &self,
    row_index: usize,
  ) -> Option<usize> {
    let sidecar = self.rho_sidecar.as_ref()?;
    sidecar
      .source_token_counts
      .get(row_index)
      .copied()
      .map(usize::from)
  }

  pub(crate) fn rho_sparse_occupancy_threshold(&self) -> Option<usize> {
    self
      .rho_sidecar
      .as_ref()
      .map(|sidecar| sidecar.sparse_occupancy_threshold)
  }

  pub(crate) fn rho_sparse_verify_perm(&self) -> usize {
    self
      .rho_sidecar
      .as_ref()
      .map_or(0, |sidecar| sidecar.sparse_verify_perm)
  }

  pub(crate) fn rho_sparse_verify_signature(
    &self,
    row_index: usize,
  ) -> Option<&[u32]> {
    let sidecar = self.rho_sidecar.as_ref()?;
    if sidecar.sparse_verify_perm == 0 {
      return None;
    }
    if sidecar.sparse_verify_active.get(row_index).copied() != Some(1) {
      return None;
    }
    let start = row_index.checked_mul(sidecar.sparse_verify_perm)?;
    let end = start.checked_add(sidecar.sparse_verify_perm)?;
    sidecar.sparse_verify_signatures.get(start..end)
  }
}
