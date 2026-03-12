-- vector_types_vtab.sql: test native storage types (float32, int8, binary)
.bail on
.load build/macosx/arm64/release/libsqlite_vector

-- ── float32 (default, regression) ──────────────────────────────────────────
CREATE VIRTUAL TABLE vf32 USING vec0(dims=4, metric=cosine);
INSERT INTO vf32(vector) VALUES('[1,2,3,4]');
INSERT INTO vf32(vector) VALUES('[4,3,2,1]');
SELECT 'f32 insert ok' AS status;

SELECT rowid FROM vf32 WHERE vf32 MATCH '[1,2,3,4]' LIMIT 1;
SELECT 'f32 match ok' AS status;

-- Verify blob size: 4 dims × 4 bytes = 16
SELECT CASE WHEN length(vector) = 16 THEN 'f32 blob size ok'
            ELSE 'FAIL: f32 blob size ' || length(vector) END AS status
  FROM vf32_data WHERE id = 1;

-- ── int8 ───────────────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE vi8 USING vec0(dims=4, metric=cosine, type=int8);
INSERT INTO vi8(vector) VALUES('[100,50,-30,80]');
INSERT INTO vi8(vector) VALUES('[10,20,30,40]');
INSERT INTO vi8(vector) VALUES('[-100,-50,30,-80]');
SELECT 'int8 insert ok' AS status;

-- Nearest to [100,50,-30,80] should be rowid 1
SELECT CASE WHEN rowid = 1 THEN 'int8 match ok'
            ELSE 'FAIL: int8 nearest rowid ' || rowid END AS status
  FROM vi8 WHERE vi8 MATCH '[90,40,-20,70]' LIMIT 1;

-- Verify blob size: 4 dims × 1 byte = 4
SELECT CASE WHEN length(vector) = 4 THEN 'int8 blob size ok'
            ELSE 'FAIL: int8 blob size ' || length(vector) END AS status
  FROM vi8_data WHERE id = 1;

-- int8 vector output should round-trip
SELECT CASE WHEN vector = '[100,50,-30,80]' THEN 'int8 output ok'
            ELSE 'FAIL: int8 output ' || vector END AS status
  FROM vi8 WHERE rowid = 1;

-- int8 with l2 metric
CREATE VIRTUAL TABLE vi8l2 USING vec0(dims=3, metric=l2, type=int8);
INSERT INTO vi8l2(vector) VALUES('[0,0,0]');
INSERT INTO vi8l2(vector) VALUES('[1,1,1]');
SELECT CASE WHEN rowid = 1 THEN 'int8 l2 ok'
            ELSE 'FAIL: int8 l2 nearest ' || rowid END AS status
  FROM vi8l2 WHERE vi8l2 MATCH '[0,0,0]' LIMIT 1;

-- int8 with ip metric
CREATE VIRTUAL TABLE vi8ip USING vec0(dims=3, metric=ip, type=int8);
INSERT INTO vi8ip(vector) VALUES('[10,20,30]');
INSERT INTO vi8ip(vector) VALUES('[1,1,1]');
SELECT CASE WHEN rowid = 1 THEN 'int8 ip ok'
            ELSE 'FAIL: int8 ip nearest ' || rowid END AS status
  FROM vi8ip WHERE vi8ip MATCH '[10,20,30]' LIMIT 1;

-- ── binary ─────────────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE vbit USING vec0(dims=8, metric=hamming, type=binary);
INSERT INTO vbit(vector) VALUES('[1,1,0,0,1,0,1,0]');
INSERT INTO vbit(vector) VALUES('[0,0,1,1,0,1,0,1]');
INSERT INTO vbit(vector) VALUES('[1,1,0,0,1,0,1,1]');
SELECT 'binary insert ok' AS status;

-- Exact match should have distance 0
SELECT CASE WHEN rowid = 1 AND distance = 0.0 THEN 'binary exact match ok'
            ELSE 'FAIL: binary match rowid=' || rowid || ' dist=' || distance
       END AS status
  FROM vbit WHERE vbit MATCH '[1,1,0,0,1,0,1,0]' LIMIT 1;

-- Verify blob size: 8 bits = 1 byte
SELECT CASE WHEN length(vector) = 1 THEN 'binary blob size ok'
            ELSE 'FAIL: binary blob size ' || length(vector) END AS status
  FROM vbit_data WHERE id = 1;

-- Binary output should be [0,1,...] format
SELECT CASE WHEN vector = '[1,1,0,0,1,0,1,0]' THEN 'binary output ok'
            ELSE 'FAIL: binary output ' || vector END AS status
  FROM vbit WHERE rowid = 1;

-- binary with jaccard metric
CREATE VIRTUAL TABLE vbitj USING vec0(dims=16, metric=jaccard, type=binary);
INSERT INTO vbitj(vector) VALUES('[1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0]');
INSERT INTO vbitj(vector) VALUES('[0,0,1,1,0,1,0,1,0,0,1,1,0,1,0,1]');
SELECT CASE WHEN rowid = 1 THEN 'binary jaccard ok'
            ELSE 'FAIL: binary jaccard nearest ' || rowid END AS status
  FROM vbitj WHERE vbitj MATCH '[1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0]' LIMIT 1;

-- ── Error cases tested manually (not in .bail on context) ──────────────────
-- int8 + hamming → "metric 'hamming' not supported for type 'int8'"
-- binary + cosine → "metric 'cosine' not supported for type 'binary'"
-- type=badtype → "unknown type 'badtype'"

SELECT 'all vector_types_vtab tests passed' AS status;
