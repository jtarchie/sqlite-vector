-- test/custom_rowid.sql
-- Verifies rowid behaviour: auto-assigned sequential IDs, stability
-- through update/delete, and correct linkage in KNN results.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE vectors USING vec0(dims=3, metric=cosine);

-- ── Test 1: Auto-assigned rowids are sequential ─────────────────────────
INSERT INTO vectors(vector) VALUES(vec('[1.0, 0.0, 0.0]'));
INSERT INTO vectors(vector) VALUES(vec('[0.0, 1.0, 0.0]'));
INSERT INTO vectors(vector) VALUES(vec('[0.0, 0.0, 1.0]'));

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM vectors_data) = 3
);
INSERT INTO assert VALUES(
  (SELECT MIN(id) FROM vectors_data) = 1
);
INSERT INTO assert VALUES(
  (SELECT MAX(id) FROM vectors_data) = 3
);

-- ── Test 2: KNN results carry stable rowids ─────────────────────────────
-- Query should return rowids that match the shadow table
CREATE TEMP TABLE knn AS
  SELECT rowid AS rid, distance FROM vectors
  WHERE vectors MATCH '[1.0, 0.0, 0.0]'
  LIMIT 3;

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM knn) = 3
);
-- Nearest to [1,0,0] should be rowid 1
INSERT INTO assert VALUES(
  (SELECT rid FROM knn ORDER BY distance LIMIT 1) = 1
);

-- ── Test 3: Rowid linkage with a related table ──────────────────────────
CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO products VALUES(1, 'Alpha');
INSERT INTO products VALUES(2, 'Beta');
INSERT INTO products VALUES(3, 'Gamma');

-- Join KNN results to products via rowid
INSERT INTO assert VALUES(
  (SELECT p.name FROM products p
   JOIN (SELECT rowid AS rid, distance FROM vectors
         WHERE vectors MATCH '[1.0, 0.0, 0.0]' LIMIT 1) k
   ON p.id = k.rid) = 'Alpha'
);

-- ── Test 4: Delete removes correct row, others keep rowids ──────────────
DELETE FROM vectors WHERE rowid = 2;

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM vectors_data) = 2
);
-- Rowid 1 and 3 still present
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM vectors_data WHERE id IN (1,3)) = 2
);

-- ── Test 5: Update preserves rowid ──────────────────────────────────────
UPDATE vectors SET vector = vec('[0.5, 0.5, 0.0]') WHERE rowid = 1;

INSERT INTO assert VALUES(
  (SELECT id FROM vectors_data WHERE id = 1) = 1
);
-- Updated vector should still be nearest to itself
INSERT INTO assert VALUES(
  (SELECT rowid FROM vectors
   WHERE vectors MATCH '[0.5, 0.5, 0.0]' LIMIT 1) = 1
);

-- ── Test 6: Insert after delete continues sequence ──────────────────────
INSERT INTO vectors(vector) VALUES(vec('[0.1, 0.1, 0.8]'));

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM vectors_data) = 3
);
-- New row gets next available id (4, since 1-3 were used previously)
INSERT INTO assert VALUES(
  (SELECT MAX(id) FROM vectors_data) >= 4
);

SELECT 'custom rowid tests passed' AS result;
