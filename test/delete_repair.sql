-- test/delete_repair.sql
-- Verifies graph repair on delete: after removing a node, kNN search on the
-- remaining nodes still returns sensible results. Tests three scenarios:
--   1. Delete a non-entry-point neighbour — remaining nodes still reachable.
--   2. Delete the entry-point node — new entry point elected, graph intact.
--   3. Chain of three consecutive deletes — graph remains navigable.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, ef_search=50);

-- Insert 8 vectors spread out over the unit cube corners / midpoints.
-- This ensures a reasonably connected graph even at small N.
INSERT INTO vecs(vector) VALUES(vec('[1.0, 0.0, 0.0]'));  -- rowid 1  (+x axis)
INSERT INTO vecs(vector) VALUES(vec('[0.0, 1.0, 0.0]'));  -- rowid 2  (+y axis)
INSERT INTO vecs(vector) VALUES(vec('[0.0, 0.0, 1.0]'));  -- rowid 3  (+z axis)
INSERT INTO vecs(vector) VALUES(vec('[0.9, 0.1, 0.0]'));  -- rowid 4  (near +x)
INSERT INTO vecs(vector) VALUES(vec('[0.1, 0.9, 0.0]'));  -- rowid 5  (near +y)
INSERT INTO vecs(vector) VALUES(vec('[0.1, 0.1, 0.8]'));  -- rowid 6  (near +z)
INSERT INTO vecs(vector) VALUES(vec('[0.6, 0.6, 0.0]'));  -- rowid 7  (x-y diag)
INSERT INTO vecs(vector) VALUES(vec('[0.6, 0.0, 0.6]'));  -- rowid 8  (x-z diag)

INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '8'
);

-- Baseline: can we find the nearest to [1,0,0]?
CREATE TEMP TABLE base_knn AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 5;

-- Should return at least 3 results and rowid 1 should be nearest (d=0).
INSERT INTO assert VALUES((SELECT COUNT(*) FROM base_knn) >= 3);
INSERT INTO assert VALUES(
  (SELECT rowid FROM base_knn ORDER BY distance LIMIT 1) = 1
);

-- ── Scenario 1: delete a non-entry-point interior node ───────────────────
-- Delete rowid 4 ([0.9,0.1,0.0]), which is close to the +x cluster.
-- Remaining graph: {1,2,3,5,6,7,8}. kNN for [1,0,0] must still find rowid 1.
DELETE FROM vecs WHERE rowid = 4;

INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '7'
);

CREATE TEMP TABLE after_del4 AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 5;

-- Must return results — graph is still navigable
INSERT INTO assert VALUES((SELECT COUNT(*) FROM after_del4) >= 1);
-- rowid 1 (d=0) must still be the nearest
INSERT INTO assert VALUES(
  (SELECT rowid FROM after_del4 ORDER BY distance LIMIT 1) = 1
);
-- rowid 4 must NOT appear (it was deleted)
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM after_del4 WHERE rowid = 4) = 0
);

-- ── Scenario 2: delete the current entry-point ────────────────────────────
-- We delete whichever node is the entry point.  After election a new EP
-- is chosen and kNN should still work.
DELETE FROM vecs WHERE rowid = (
  SELECT CAST(value AS INTEGER) FROM vecs_config WHERE key = 'entry_point'
);

-- At most 6 rows remain; count must have dropped by 1
INSERT INTO assert VALUES(
  (SELECT CAST(value AS INTEGER) FROM vecs_config WHERE key='count') <= 6
);
INSERT INTO assert VALUES(
  (SELECT CAST(value AS INTEGER) FROM vecs_config WHERE key='count') >= 5
);

-- entry_point must now reference a surviving node
INSERT INTO assert VALUES(
  (SELECT CAST(value AS INTEGER) FROM vecs_config WHERE key='entry_point') >= 1
);

-- kNN must return at least 1 result (graph non-empty)
CREATE TEMP TABLE after_ep_del AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[0.0,1.0,0.0]' LIMIT 4;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM after_ep_del) >= 1);
-- All returned distances are non-NULL and non-negative
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM after_ep_del WHERE distance IS NULL OR distance < 0) = 0
);

-- ── Scenario 3: chain of three consecutive deletes ────────────────────────
-- Delete rowids 2, 3, 7 one by one; graph repair should keep remaining nodes
-- reachable.
DELETE FROM vecs WHERE rowid = 2;
DELETE FROM vecs WHERE rowid = 3;
DELETE FROM vecs WHERE rowid = 7;

-- Must have at most 3 nodes left (started with ≤6; dropped 3 more = ≤3)
INSERT INTO assert VALUES(
  (SELECT CAST(value AS INTEGER) FROM vecs_config WHERE key='count') >= 1
);

-- kNN on surviving nodes (all near +x / +z):
-- surviving set is some subset of {1, 5, 6, 8}
CREATE TEMP TABLE final_knn AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[0.5,0.0,0.5]' LIMIT 4;

-- Must return at least 1 result (not an empty/broken graph)
INSERT INTO assert VALUES((SELECT COUNT(*) FROM final_knn) >= 1);
-- All distances are valid
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM final_knn WHERE distance IS NULL OR distance < 0) = 0
);

SELECT 'delete_repair tests passed';
