.load build/macosx/arm64/release/libsqlite_vector
.mode list

-- ── Int8 distance functions ──────────────────────────────────────────────

-- L2: sqrt((1-3)^2 + (2-4)^2 + (3-5)^2) = sqrt(12) ≈ 3.464
SELECT 'i8_l2', ROUND(vec_distance_l2_i8('[1,2,3]', '[3,4,5]'), 3);

-- Cosine: identical vectors → 0
SELECT 'i8_cos_same', ROUND(vec_distance_cosine_i8('[10,20,30]', '[10,20,30]'), 6);

-- Cosine: opposite vectors → 2
SELECT 'i8_cos_opp', ROUND(vec_distance_cosine_i8('[127,0]', '[-128,0]'), 1);

-- IP: -(1*4 + 2*5 + 3*6) = -32
SELECT 'i8_ip', vec_distance_ip_i8('[1,2,3]', '[4,5,6]');

-- L1: |1-4| + |2-5| + |3-6| = 9
SELECT 'i8_l1', vec_distance_l1_i8('[1,2,3]', '[4,5,6]');

-- Null handling
SELECT 'i8_null', vec_distance_l2_i8(NULL, '[1,2,3]') IS NULL;

-- ── Binary distance functions ────────────────────────────────────────────

-- Hamming: X'FF' vs X'00' = 8 bits differ
SELECT 'bit_hamming_8', vec_distance_hamming_bit(X'FF', X'00');

-- Hamming: identical = 0
SELECT 'bit_hamming_0', vec_distance_hamming_bit(X'AA', X'AA');

-- Hamming: X'FF00' vs X'F000' = 4 bits differ
SELECT 'bit_hamming_4', vec_distance_hamming_bit(X'FF00', X'F000');

-- Jaccard: identical = 0
SELECT 'bit_jaccard_0', ROUND(vec_distance_jaccard_bit(X'FF', X'FF'), 1);

-- Null handling
SELECT 'bit_null', vec_distance_hamming_bit(NULL, X'FF') IS NULL;
