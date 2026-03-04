-- basic.sql: smoke test for the sqlite-vector extension
-- Run with: sqlite3 :memory: < test/basic.sql

.bail on
.load build/macosx/arm64/release/libsqlite_vector

-- Module should now be registered
SELECT 'extension loaded' AS status;

-- Create a virtual table
CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=cosine);
SELECT 'virtual table created' AS status;

-- MATCH query returns empty (no data yet), but must not error
SELECT COUNT(*) AS result_count FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 5;

-- Missing dims must error
-- (Uncomment to test error path manually)
-- CREATE VIRTUAL TABLE bad USING vec0(metric=cosine);

SELECT 'all tests passed' AS status;
