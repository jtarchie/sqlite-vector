-- test/query_patterns.sql
-- Tests all query paths through vec0 xBestIndex/xFilter:
--   idxNum 0: full scan
--   idxNum 1: MATCH kNN
--   idxNum 2: rowid point lookup
--   idxNum 151-156: operator-based kNN
--   Flags: ef_search, LIMIT, distance threshold (LT/LE/GT/GE)

.load build/macosx/arm64/release/libsqlite_vector
.bail on

-- ── Setup ─────────────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=l2);
INSERT INTO v(vector) VALUES('[1,0,0,0]');   -- rowid 1, L2 to query=0.0
INSERT INTO v(vector) VALUES('[0,1,0,0]');   -- rowid 2, L2 to query=sqrt(2)≈1.414
INSERT INTO v(vector) VALUES('[0,0,1,0]');   -- rowid 3, L2 to query=sqrt(2)
INSERT INTO v(vector) VALUES('[0,0,0,1]');   -- rowid 4, L2 to query=sqrt(2)
INSERT INTO v(vector) VALUES('[1,1,0,0]');   -- rowid 5, L2 to query=1.0

-- ── Full scan (idxNum 0) ──────────────────────────────────────────────────
SELECT '-- full_scan_count';
SELECT COUNT(*) FROM v;
-- expect: 5

-- ── Rowid point lookup (idxNum 2) ─────────────────────────────────────────
SELECT '-- rowid_lookup';
SELECT rowid FROM v WHERE rowid = 3;
-- expect: 3

SELECT '-- rowid_lookup_miss';
SELECT COUNT(*) FROM v WHERE rowid = 999;
-- expect: 0

-- ── MATCH kNN (idxNum 1) ────────────────────────────────────────────────
SELECT '-- knn_match_3';
SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' LIMIT 3;
-- expect: 1, 5, then one of {2,3,4}

SELECT '-- knn_match_distance_ordered';
SELECT ROUND(distance, 4) FROM v WHERE v MATCH '[1,0,0,0]' LIMIT 3;
-- first two must be 0.0 and 1.0

-- ── MATCH + ef_search override ────────────────────────────────────────────
SELECT '-- knn_ef_search';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[0.5,0.5,0,0]' AND ef_search=50 LIMIT 5);
-- expect: 5

-- ── MATCH + LIMIT variations ─────────────────────────────────────────────
SELECT '-- knn_limit_1';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' LIMIT 1);
-- expect: 1

SELECT '-- knn_limit_exceeds_count';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' LIMIT 100);
-- expect: 5  (only 5 rows exist)

-- ── Distance threshold: LE ───────────────────────────────────────────────
SELECT '-- dist_le_1';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' AND distance <= 1.0 LIMIT 10);
-- expect: 2  (distances 0.0 and 1.0)

-- ── Distance threshold: LT ───────────────────────────────────────────────
SELECT '-- dist_lt_1';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' AND distance < 1.0 LIMIT 10);
-- expect: 1  (distance 0.0 only)

-- ── Distance threshold: GT ───────────────────────────────────────────────
SELECT '-- dist_gt';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' AND distance > 1.0 LIMIT 10);
-- expect: 3  (the three sqrt(2) distances)

-- ── Distance threshold: GE ───────────────────────────────────────────────
SELECT '-- dist_ge';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' AND distance >= 1.0 LIMIT 10);
-- expect: 4  (1.0 + three sqrt(2) distances)

-- ── Combined: LT + GT ───────────────────────────────────────────────────
SELECT '-- dist_band';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[1,0,0,0]' AND distance > 0.5 AND distance < 1.2 LIMIT 10);
-- expect: 1  (rowid 5 at distance=1.0)

-- ── Operator-based kNN (idxNum 151-156) ──────────────────────────────────
-- L2 operator
SELECT '-- op_l2';
SELECT rowid FROM v WHERE vec_distance_l2(v, '[1,0,0,0]') LIMIT 1;
-- expect: 1

-- Cosine operator (override metric)
SELECT '-- op_cosine';
SELECT rowid FROM v WHERE vec_distance_cosine(v, '[1,0,0,0]') LIMIT 1;
-- expect: 1

-- IP operator
SELECT '-- op_ip';
SELECT rowid FROM v WHERE vec_distance_ip(v, '[1,0,0,0]') LIMIT 1;
-- expect: 1 or 5 (both have max inner product with query)

-- L1 operator
SELECT '-- op_l1';
SELECT rowid FROM v WHERE vec_distance_l1(v, '[1,0,0,0]') LIMIT 1;
-- expect: 1

-- ── Operator with distance threshold ─────────────────────────────────────
SELECT '-- op_l2_dist_le';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE vec_distance_l2(v, '[1,0,0,0]') AND distance <= 1.0 LIMIT 10);
-- expect: 2

-- ── Operator with ef_search ──────────────────────────────────────────────
SELECT '-- op_l2_ef';
SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE vec_distance_l2(v, '[0.5,0.5,0,0]') AND ef_search=50 LIMIT 5);
-- expect: 5

-- ── Bit type with hamming via MATCH (text-format input) ──────────────────
CREATE VIRTUAL TABLE vb USING vec0(dims=8, metric=hamming, type=bit);
INSERT INTO vb(vector) VALUES('[1,1,1,1,1,1,1,1]');
INSERT INTO vb(vector) VALUES('[0,0,0,0,0,0,0,0]');
INSERT INTO vb(vector) VALUES('[1,1,1,1,0,0,0,0]');

SELECT '-- bit_match_hamming';
SELECT rowid FROM vb WHERE vb MATCH '[1,1,1,1,1,1,1,1]' LIMIT 1;
-- expect: 1

.print PASS
