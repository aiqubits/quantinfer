//! GGML physical layout helpers.

/// Map an adjacent-pair GGML Q/K row to the checkpoint's half-split channel.
pub fn qk_channel(row: usize, head_dim: usize) -> usize {
    let channel = row % head_dim;
    row - channel + channel / 2 + (channel % 2) * (head_dim / 2)
}

#[cfg(test)]
mod tests {
    use super::qk_channel;

    #[test]
    fn maps_adjacent_pairs_to_half_split_channels() {
        assert_eq!(
            (0..8).map(|row| qk_channel(row, 8)).collect::<Vec<_>>(),
            [0, 4, 1, 5, 2, 6, 3, 7]
        );
        assert_eq!(qk_channel(9, 8), 12);
    }
}
