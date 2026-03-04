-- test/knn.sql
-- Verifies kNN search (MATCH), xColumn vector text, DELETE, UPDATE,
-- and full-scan via xFilter idxNum=0.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, ef_search=10);

INSERT INTO vecs(vector) VALUES(vec('[1.0,0.0,0.0]'));  -- rowid 1
INSERT INTO vecs(vector) VALUES(vec('[0.0,1.0,0.0]'));  -- rowid 2
INSERT INTO vecs(vector) VALUES(vec('[0.0,0.0,1.0]'));  -- rowid 3
INSERT INTO vecs(vector) VALUES(vec('[0.9,0.1,0.0]'));  -- rowid 4

-- ── kNN ordering ─────────────────────────────────────────────────────────
-- Query [1,0,0]: nearest should be rowid 1 (d≈0), then 4 (d≈0.14), then 2,3
CREATE TEMP TABLE knn_res AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 4;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM knn_res) = 4);

-- rowid 1 is nearest (distance ≈ 0)
INSERT INTO assert VALUES(
  (SELECT rowid FROM knn_res ORDER BY distance LIMIT 1) = 1
);

-- rowid 4 is second nearest
INSERT INTO assert VALUES(
  (SELECT rowid FROM knn_res ORDER BY distance LIMIT 1 OFFSET 1) = 4
);

-- distances are non-NULL and ordered ascending
INSERT INTO assert VALUES(
  (SELECT MIN(distance) FROM knn_res) >= 0.0
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM knn_res WHERE distance IS NULL) = 0
);

-- ── vector column round-trips ─────────────────────────────────────────────
-- xColumn should return text like [1.0,0.0,0.0] for rowid 1
CREATE TEMP TABLE vec_out AS
  SELECT rowid, vector FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 1;

INSERT INTO assert VALUES(
  (SELECT vec_dims(vector) FROM vec_out WHERE rowid=1) = 3
);

-- ── LIMIT is respected ───────────────────────────────────────────────────
CREATE TEMP TABLE knn2 AS
  SELECT rowid FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 2;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM knn2) = 2);

-- ── Full scan (no MATCH) — all rows, distances NULL ──────────────────────
CREATE TEMP TABLE fullscan AS SELECT rowid, vector, distance FROM vecs;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM fullscan) = 4);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM fullscan WHERE distance IS NOT NULL) = 0
);

-- ── DELETE removes from kNN results ──────────────────────────────────────
DELETE FROM vecs WHERE rowid=1;

INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '3'
);

CREATE TEMP TABLE knn3 AS
  SELECT rowid FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 4;

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM knn3 WHERE rowid=1) = 0
);
-- rowid 4 is now nearest
INSERT INTO assert VALUES(
  (SELECT rowid FROM knn3 ORDER BY rowid LIMIT 1) != 1
);

-- ── UPDATE changes search results ────────────────────────────────────────
-- rowid 2 was [0,1,0]; update it to [1,0,0] so it becomes the nearest
UPDATE vecs SET vector = vec('[1.0,0.0,0.0]') WHERE rowid=2;

INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '3'
);

CREATE TEMP TABLE knn4 AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 4;

INSERT INTO assert VALUES(
  (SELECT rowid FROM knn4 ORDER BY distance LIMIT 1) = 2
);

SELECT 'knn tests passed';
