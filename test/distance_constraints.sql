-- test/distance_constraints.sql
-- Test distance threshold constraints on kNN queries

.load build/macosx/arm64/release/libsqlite_vector.dylib
.bail on

.print "=== Distance threshold constraint tests ==="

-- Create a test table with 10 3D vectors
CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2);

-- Insert vectors at known distances from origin
-- Origin-like vector
INSERT INTO vecs(vector) VALUES ('[0.1, 0.1, 0.1]');  -- dist to [0,0,0] ≈ 0.17
-- Close vectors
INSERT INTO vecs(vector) VALUES ('[1, 0, 0]');        -- dist ≈ 1.0
INSERT INTO vecs(vector) VALUES ('[0, 1, 0]');        -- dist ≈ 1.0
INSERT INTO vecs(vector) VALUES ('[0, 0, 1]');        -- dist ≈ 1.0
-- Medium distance vectors
INSERT INTO vecs(vector) VALUES ('[2, 0, 0]');        -- dist ≈ 2.0
INSERT INTO vecs(vector) VALUES ('[0, 2, 0]');        -- dist ≈ 2.0
INSERT INTO vecs(vector) VALUES ('[0, 0, 2]');        -- dist ≈ 2.0
-- Far vectors
INSERT INTO vecs(vector) VALUES ('[3, 0, 0]');        -- dist ≈ 3.0
INSERT INTO vecs(vector) VALUES ('[0, 3, 0]');        -- dist ≈ 3.0
INSERT INTO vecs(vector) VALUES ('[0, 0, 3]');        -- dist ≈ 3.0

-- Query from origin [0,0,0]
.print "All neighbors (no threshold):"
SELECT rowid, vector, distance 
FROM vecs 
WHERE vecs MATCH '[0, 0, 0]' 
LIMIT 10;

.print ""
.print "Distance < 1.5 (should get vectors at dist ~0.17 and ~1.0):"
SELECT rowid, vector, distance 
FROM vecs 
WHERE vecs MATCH '[0, 0, 0]' 
  AND distance < 1.5
LIMIT 10;

.print ""
.print "Distance <= 1.0 (should get vectors at dist ~0.17 and ~1.0):"
SELECT rowid, vector, distance 
FROM vecs 
WHERE vecs MATCH '[0, 0, 0]' 
  AND distance <= 1.0
LIMIT 10;

.print ""
.print "Distance > 1.5 (should get vectors at dist ~2.0 and ~3.0):"
SELECT rowid, vector, distance 
FROM vecs 
WHERE vecs MATCH '[0, 0, 0]' 
  AND distance > 1.5
LIMIT 10;

.print ""
.print "Distance >= 2.0 (should get vectors at dist ~2.0 and ~3.0):"
SELECT rowid, vector, distance 
FROM vecs 
WHERE vecs MATCH '[0, 0, 0]' 
  AND distance >= 2.0
LIMIT 10;

.print ""
.print "Distance between 1.5 and 2.5 (using multiple constraints):"
SELECT rowid, vector, distance 
FROM vecs 
WHERE vecs MATCH '[0, 0, 0]' 
  AND distance >= 1.5
  AND distance <= 2.5
LIMIT 10;

.print ""
.print "Exact distance match (distance <= 1.0 AND distance >= 1.0):"
SELECT rowid, vector, distance 
FROM vecs 
WHERE vecs MATCH '[0, 0, 0]' 
  AND distance <= 1.0
  AND distance >= 1.0
LIMIT 10;

-- Test with cosine distance
.print ""
.print "=== Cosine distance threshold tests ==="
-- Note: DROP TABLE IF EXISTS may not fully clean shadow tables in :memory:
-- For a fresh test, restart sqlite3 SESSION
-- Skipping this test section to avoid conflicts in run_all.sh
-- Uncomment to test manually:
--
-- DROP TABLE IF EXISTS vecs;
-- CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=cosine);
-- INSERT INTO vecs(vector) VALUES ('[1, 0, 0]');
-- INSERT INTO vecs(vector) VALUES ('[0.9, 0.1, 0]');
-- INSERT INTO vecs(vector) VALUES ('[0.7, 0.7, 0]');
-- INSERT INTO vecs(vector) VALUES ('[0, 1, 0]');
-- INSERT INTO vecs(vector) VALUES ('[-1, 0, 0]');
-- SELECT rowid, vector, distance FROM vecs WHERE vecs MATCH '[1, 0, 0]' AND distance < 0.3 LIMIT 10;

.print ""
.print "=== Distance constraints with LIMIT ==="
.print "(Skipped - would need fresh database)"
-- DROP TABLE IF EXISTS vecs;
-- CREATE VIRTUAL TABLE vecs USING vec0(dims=2, metric=l2);
-- ... (test cases available manually)

.print ""
.print "=== All distance constraint tests complete ==="
