-- test/operators.sql
-- Verifies xFindFunction operator aliases:
--   vec_distance_l2(col, query)      → idxNum 151 (L2 kNN)
--   vec_distance_cosine(col, query)  → idxNum 152 (cosine kNN)
--   vec_distance_ip(col, query)      → idxNum 153 (inner-product kNN)
--   vec_distance_l1(col, query)      → idxNum 154 (L1 kNN)
--
-- Also verifies that the function call overrides the table's declared metric.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

-- A row that violates CHECK(val) causes an immediate abort when we pass 0 (false).
CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- ── Dataset: orthonormal unit vectors in R3 ─────────────────────────────
-- rowid 1 = [1,0,0]  rowid 2 = [0,1,0]  rowid 3 = [0,0,1]
CREATE VIRTUAL TABLE vl2  USING vec0(dims=3, metric=l2,     ef_search=10);
CREATE VIRTUAL TABLE vcos USING vec0(dims=3, metric=cosine, ef_search=10);
CREATE VIRTUAL TABLE vip  USING vec0(dims=3, metric=ip,     ef_search=10);
CREATE VIRTUAL TABLE vl1  USING vec0(dims=3, metric=l1,     ef_search=10);

INSERT INTO vl2(vector)  VALUES(vec('[1.0,0.0,0.0]'));
INSERT INTO vl2(vector)  VALUES(vec('[0.0,1.0,0.0]'));
INSERT INTO vl2(vector)  VALUES(vec('[0.0,0.0,1.0]'));

INSERT INTO vcos(vector) VALUES(vec('[1.0,0.0,0.0]'));
INSERT INTO vcos(vector) VALUES(vec('[0.0,1.0,0.0]'));
INSERT INTO vcos(vector) VALUES(vec('[0.0,0.0,1.0]'));

INSERT INTO vip(vector)  VALUES(vec('[1.0,0.0,0.0]'));
INSERT INTO vip(vector)  VALUES(vec('[0.0,1.0,0.0]'));
INSERT INTO vip(vector)  VALUES(vec('[0.0,0.0,1.0]'));

INSERT INTO vl1(vector)  VALUES(vec('[1.0,0.0,0.0]'));
INSERT INTO vl1(vector)  VALUES(vec('[0.0,1.0,0.0]'));
INSERT INTO vl1(vector)  VALUES(vec('[0.0,0.0,1.0]'));

-- ── vec_distance_l2: L2 kNN (idxNum=151) ────────────────────────────────
-- Query [1,0,0]: exact match is rowid 1 (distance=0), others at sqrt(2).
CREATE TEMP TABLE op_l2 AS
  SELECT rowid, distance
  FROM vl2
  WHERE vec_distance_l2(vl2, '[1.0,0.0,0.0]')
  LIMIT 3;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_l2) = 3);
INSERT INTO assert VALUES((SELECT rowid    FROM op_l2 ORDER BY distance LIMIT 1) = 1);
INSERT INTO assert VALUES((SELECT distance FROM op_l2 ORDER BY distance LIMIT 1) = 0.0);
INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_l2 WHERE distance IS NULL) = 0);

-- ── vec_distance_cosine: cosine kNN (idxNum=152) ─────────────────────────
-- Query [1,0,0]: rowid 1 is cosine-nearest (similarity=1, distance≈0).
CREATE TEMP TABLE op_cos AS
  SELECT rowid, distance
  FROM vcos
  WHERE vec_distance_cosine(vcos, '[1.0,0.0,0.0]')
  LIMIT 3;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_cos) = 3);
INSERT INTO assert VALUES((SELECT rowid FROM op_cos ORDER BY distance LIMIT 1) = 1);
INSERT INTO assert VALUES((SELECT distance FROM op_cos ORDER BY distance LIMIT 1) < 1e-5);
INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_cos WHERE distance IS NULL) = 0);

-- ── vec_distance_ip: inner-product kNN (idxNum=153) ──────────────────────
-- IP distance is negated dot-product (or 1-dot) so [1,0,0]·[1,0,0]=1 → smallest.
CREATE TEMP TABLE op_ip AS
  SELECT rowid, distance
  FROM vip
  WHERE vec_distance_ip(vip, '[1.0,0.0,0.0]')
  LIMIT 3;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_ip) = 3);
INSERT INTO assert VALUES((SELECT rowid FROM op_ip ORDER BY distance LIMIT 1) = 1);
INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_ip WHERE distance IS NULL) = 0);

-- ── vec_distance_l1: L1 kNN (idxNum=154) ─────────────────────────────────
-- L1([1,0,0],[1,0,0])=0  L1([1,0,0],[0,1,0])=2  L1([1,0,0],[0,0,1])=2
CREATE TEMP TABLE op_l1 AS
  SELECT rowid, distance
  FROM vl1
  WHERE vec_distance_l1(vl1, '[1.0,0.0,0.0]')
  LIMIT 3;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_l1) = 3);
INSERT INTO assert VALUES((SELECT rowid    FROM op_l1 ORDER BY distance LIMIT 1) = 1);
INSERT INTO assert VALUES((SELECT distance FROM op_l1 ORDER BY distance LIMIT 1) = 0.0);
INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_l1 WHERE distance IS NULL) = 0);

-- ── LIMIT respected ──────────────────────────────────────────────────────
CREATE TEMP TABLE op_lim AS
  SELECT rowid FROM vl2 WHERE vec_distance_l2(vl2, '[1.0,0.0,0.0]') LIMIT 1;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM op_lim) = 1);
INSERT INTO assert VALUES((SELECT rowid FROM op_lim) = 1);

-- ── Metric override: table=l2, function=cosine overrides metric ───────────
-- With unit vectors both L2 and cosine rank [1,0,0] nearest to itself.
CREATE TEMP TABLE op_override AS
  SELECT rowid, distance
  FROM vl2
  WHERE vec_distance_cosine(vl2, '[1.0,0.0,0.0]')
  LIMIT 1;

INSERT INTO assert VALUES((SELECT rowid FROM op_override) = 1);
INSERT INTO assert VALUES((SELECT distance FROM op_override) < 1e-5);

SELECT 'operators tests passed';
