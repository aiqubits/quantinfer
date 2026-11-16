use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use crate::Result;

use super::invalid;

pub(crate) const MAGIC: &[u8; 4] = b"QSIG";
pub(crate) const VERSION: u32 = 1;
pub(crate) const ALGORITHM_ED25519: u32 = 1;
pub(crate) const LENGTH: usize = 108;

pub(crate) fn encode(root_hash: &[u8; 32], key: &SigningKey) -> [u8; LENGTH] {
    let signature = key.sign(root_hash);
    let mut envelope = [0_u8; LENGTH];
    envelope[0..4].copy_from_slice(MAGIC);
    envelope[4..8].copy_from_slice(&VERSION.to_le_bytes());
    envelope[8..12].copy_from_slice(&ALGORITHM_ED25519.to_le_bytes());
    envelope[12..44].copy_from_slice(key.verifying_key().as_bytes());
    envelope[44..108].copy_from_slice(&signature.to_bytes());
    envelope
}

pub(crate) fn verify(root_hash: &[u8; 32], envelope: &[u8]) -> Result<()> {
    if envelope.len() != LENGTH
        || &envelope[..4] != MAGIC
        || u32::from_le_bytes(envelope[4..8].try_into().unwrap()) != VERSION
        || u32::from_le_bytes(envelope[8..12].try_into().unwrap()) != ALGORITHM_ED25519
    {
        return Err(invalid("invalid QIF signature envelope"));
    }
    let key = VerifyingKey::from_bytes(&envelope[12..44].try_into().unwrap())
        .map_err(|_| invalid("invalid QIF signature public key"))?;
    let signature = Signature::from_bytes(&envelope[44..108].try_into().unwrap());
    key.verify(root_hash, &signature)
        .map_err(|_| invalid("QIF signature verification failed"))
}
