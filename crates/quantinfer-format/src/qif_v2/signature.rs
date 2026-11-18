use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

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
    if key.to_edwards().compress().as_bytes() != &envelope[12..44] {
        return Err(invalid("noncanonical QIF signature public key"));
    }
    let signature = Signature::from_bytes(&envelope[44..108].try_into().unwrap());
    key.verify_strict(root_hash, &signature)
        .map_err(|_| invalid("QIF signature verification failed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_weak_key_forgery_for_arbitrary_roots() {
        let mut envelope = encode(&[0; 32], &SigningKey::from_bytes(&[42; 32]));
        envelope[12..108].fill(0);
        envelope[12] = 1;
        envelope[44] = 1;
        for root in [[0; 32], [1; 32], [255; 32]] {
            assert!(verify(&root, &envelope).is_err());
        }
    }

    #[test]
    fn independent_point_filters_cover_every_small_order_y() {
        let mut canonical_points = std::collections::BTreeSet::new();
        for encoding in [
            "0000000000000000000000000000000000000000000000000000000000000000",
            "0100000000000000000000000000000000000000000000000000000000000000",
            "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
            "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
            "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
        ] {
            let mut bytes = [0_u8; 32];
            for (out, pair) in bytes.iter_mut().zip(encoding.as_bytes().as_chunks::<2>().0) {
                *out = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
            }
            for sign in [0, 128] {
                bytes[31] = (bytes[31] & 127) | sign;
                let key = VerifyingKey::from_bytes(&bytes).unwrap();
                assert!(key.is_weak(), "{encoding}");
                canonical_points.insert(key.to_edwards().compress().to_bytes());
            }
        }
        assert_eq!(canonical_points.len(), 8);
    }
}
