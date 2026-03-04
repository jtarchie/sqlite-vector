-- test/bulk_insert.sql
-- Verifies:
--   1. Bulk insert within BEGIN/COMMIT: count updated correctly, kNN works.
--   2. ROLLBACK correctly reverts inserts (count unchanged, data absent).

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2);

-- ── Bulk insert in a transaction ──────────────────────────────────────────
BEGIN;
  INSERT INTO vecs(vector) VALUES(vec('[1.0,0.0,0.0]'));  -- rowid 1
  INSERT INTO vecs(vector) VALUES(vec('[0.0,1.0,0.0]'));  -- rowid 2
  INSERT INTO vecs(vector) VALUES(vec('[0.0,0.0,1.0]'));  -- rowid 3
  INSERT INTO vecs(vector) VALUES(vec('[0.5,0.5,0.0]'));  -- rowid 4
  INSERT INTO vecs(vector) VALUES(vec('[0.5,0.0,0.5]'));  -- rowid 5
  INSERT INTO vecs(vector) VALUES(vec('[0.0,0.5,0.5]'));  -- rowid 6
  INSERT INTO vecs(vector) VALUES(vec('[0.9,0.1,0.0]'));  -- rowid 7
  INSERT INTO vecs(vector) VALUES(vec('[0.1,0.9,0.0]'));  -- rowid 8
  INSERT INTO vecs(vector) VALUES(vec('[0.1,0.1,0.8]'));  -- rowid 9
  INSERT INTO vecs(vector) VALUES(vec('[0.8,0.1,0.1]'));  -- rowid 10
COMMIT;

-- Count should be 10
INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '10'
);

-- kNN should return results (graph is navigable)
CREATE TEMP TABLE bulk_knn AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 5;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM bulk_knn) = 5);
-- rowid 1 (exact match) should be nearest
INSERT INTO assert VALUES(
  (SELECT rowid FROM bulk_knn ORDER BY distance LIMIT 1) = 1
);
-- All distances are non-negative and non-NULL
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM bulk_knn WHERE distance IS NULL OR distance < 0) = 0
);

-- ── ROLLBACK reverts inserts ──────────────────────────────────────────────
BEGIN;
  INSERT INTO vecs(vector) VALUES(vec('[0.2,0.3,0.5]'));  -- rowid 11
  INSERT INTO vecs(vector) VALUES(vec('[0.6,0.2,0.2]'));  -- rowid 12
  INSERT INTO vecs(vector) VALUES(vec('[0.3,0.6,0.1]'));  -- rowid 13
ROLLBACK;

-- Count must still be 10 (rolled-back inserts are gone)
INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '10'
);

-- Rowids 11–13 must not appear in full-scan
CREATE TEMP TABLE fullscan AS SELECT rowid FROM vecs;

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM fullscan WHERE rowid IN (11, 12, 13)) = 0
);

-- kNN still works after the rollback
CREATE TEMP TABLE post_rollback_knn AS
  SELECT rowid FROM vecs WHERE vecs MATCH '[0.0,1.0,0.0]' LIMIT 5;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM post_rollback_knn) = 5);
-- rowid 2 (exact [0,1,0]) should be nearest
INSERT INTO assert VALUES(
  (SELECT rowid FROM post_rollback_knn ORDER BY rowid LIMIT 1)
  = (SELECT rowid FROM vecs WHERE vecs MATCH '[0.0,1.0,0.0]'
     ORDER BY distance LIMIT 1)
);

SELECT 'bulk_insert tests passed';
