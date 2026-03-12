-- test/special_values.sql
-- Edge cases around special float values: zero vectors, negative components,
-- very large/small magnitudes, and uniform vectors.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- ── Zero vector ─────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE v_l2 USING vec0(dims=3, metric=l2);

INSERT INTO v_l2(vector) VALUES(vec('[0.0,0.0,0.0]'));
INSERT INTO v_l2(vector) VALUES(vec('[1.0,0.0,0.0]'));
INSERT INTO v_l2(vector) VALUES(vec('[0.0,1.0,0.0]'));

-- Self-distance of zero vector should be 0
INSERT INTO assert VALUES(
  (SELECT distance FROM v_l2 WHERE v_l2 MATCH '[0.0,0.0,0.0]' LIMIT 1) = 0.0
);
-- kNN returns all 3 results
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM (
    SELECT rowid FROM v_l2 WHERE v_l2 MATCH '[0.0,0.0,0.0]' LIMIT 3
  ))  = 3
);

-- ── Negative vector components ──────────────────────────────────────────
INSERT INTO v_l2(vector) VALUES(vec('[-1.0,-2.0,-3.0]'));

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v_l2_data) = 4
);
-- Distance between negative and positive vectors is computable
INSERT INTO assert VALUES(
  vec_distance_l2('[-1.0,-2.0,-3.0]','[1.0,2.0,3.0]') > 0.0
);

-- ── Uniform vectors ─────────────────────────────────────────────────────
-- All-ones vectors: cosine distance to itself should be 0
INSERT INTO assert VALUES(
  vec_distance_cosine('[1.0,1.0,1.0]','[1.0,1.0,1.0]') < 0.0001
);

-- ── Large magnitude ─────────────────────────────────────────────────────
-- L2 distance with large values shouldn't crash
INSERT INTO assert VALUES(
  vec_distance_l2('[1e20,0,0]','[0,0,0]') > 0.0
);

-- ── Small magnitude ─────────────────────────────────────────────────────
INSERT INTO assert VALUES(
  vec_distance_l2('[1e-20,0,0]','[0,0,0]') >= 0.0
);

-- ── Mixed sign KNN retrieval ────────────────────────────────────────────
CREATE VIRTUAL TABLE v_cos USING vec0(dims=2, metric=cosine);
INSERT INTO v_cos(vector) VALUES(vec('[1.0,0.0]'));
INSERT INTO v_cos(vector) VALUES(vec('[-1.0,0.0]'));   -- opposite direction
INSERT INTO v_cos(vector) VALUES(vec('[0.707,0.707]'));

-- Querying [1,0]: nearest should be rowid 1, farthest rowid 2
INSERT INTO assert VALUES(
  (SELECT rowid FROM v_cos WHERE v_cos MATCH '[1.0,0.0]' LIMIT 1) = 1
);

-- All 3 should be returned
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM (
    SELECT rowid FROM v_cos WHERE v_cos MATCH '[1.0,0.0]' LIMIT 3
  )) = 3
);

-- ── Scalar distance functions accept negative components ────────────────
INSERT INTO assert VALUES(
  vec_distance_l2('[-5.0,3.0]','[2.0,-1.0]') > 7.0
);
INSERT INTO assert VALUES(
  vec_distance_l2('[-5.0,3.0]','[2.0,-1.0]') < 9.0
);

SELECT 'special_values: all tests passed';
