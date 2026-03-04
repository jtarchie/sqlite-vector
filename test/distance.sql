-- test/distance.sql
-- Known-answer tests for vec_distance_* SQL scalar functions.
-- Run with: sqlite3 :memory: < test/distance.sql
-- Uses .bail on so the first failing assertion stops the run.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- ── vec_distance_l2 ────────────────────────────────────────────────────────
-- 3-4-5 right triangle: sqrt((3-0)^2 + (0-4)^2) = sqrt(25) = 5.0
INSERT INTO assert VALUES(ABS(vec_distance_l2('[3,0]', '[0,4]') - 5.0) < 1e-5);

-- identical vectors → distance 0
INSERT INTO assert VALUES(vec_distance_l2('[1,2,3]', '[1,2,3]') = 0.0);

-- ── vec_distance_cosine ────────────────────────────────────────────────────
-- same direction → cosine similarity 1 → distance 0
INSERT INTO assert VALUES(ABS(vec_distance_cosine('[1,0]', '[1,0]')) < 1e-5);

-- orthogonal unit vectors → cosine similarity 0 → distance 1
INSERT INTO assert VALUES(ABS(vec_distance_cosine('[1,0]', '[0,1]') - 1.0) < 1e-5);

-- opposite direction → cosine similarity -1 → distance 2
INSERT INTO assert VALUES(ABS(vec_distance_cosine('[1,0]', '[-1,0]') - 2.0) < 1e-5);

-- ── vec_distance_ip ────────────────────────────────────────────────────────
-- negated dot product: -([1,2]·[3,4]) = -(3+8) = -11
INSERT INTO assert VALUES(ABS(vec_distance_ip('[1,2]', '[3,4]') - (-11.0)) < 1e-5);

-- zero vector: dot = 0, negated = 0
INSERT INTO assert VALUES(ABS(vec_distance_ip('[0,0,0]', '[1,2,3]')) < 1e-5);

-- ── vec_distance_l1 ────────────────────────────────────────────────────────
-- |1-4| + |2-6| = 3 + 4 = 7
INSERT INTO assert VALUES(ABS(vec_distance_l1('[1,2]', '[4,6]') - 7.0) < 1e-5);

-- identical → 0
INSERT INTO assert VALUES(vec_distance_l1('[5,5,5]', '[5,5,5]') = 0.0);

-- ── NULL passthrough ───────────────────────────────────────────────────────
INSERT INTO assert VALUES(vec_distance_l2(NULL, '[1,2]') IS NULL);
INSERT INTO assert VALUES(vec_distance_cosine('[1,0]', NULL) IS NULL);
INSERT INTO assert VALUES(vec_distance_ip(NULL, NULL) IS NULL);

SELECT 'distance tests passed';
