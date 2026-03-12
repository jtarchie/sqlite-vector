-- test/empty_table.sql
-- Verifies that operations on an empty vec0 table do not crash or produce
-- incorrect results.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=l2);

-- ── KNN on empty table returns zero rows ────────────────────────────────
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM (
    SELECT rowid FROM v WHERE v MATCH '[1.0,0.0,0.0]' LIMIT 5
  )) = 0
);

-- ── Full scan on empty table ────────────────────────────────────────────
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v) = 0
);

-- ── Rowid lookup on empty table ─────────────────────────────────────────
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v WHERE rowid=1) = 0
);

-- ── Delete on empty table is a no-op ────────────────────────────────────
DELETE FROM v WHERE rowid=1;
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v) = 0
);

-- ── Config still valid ──────────────────────────────────────────────────
INSERT INTO assert VALUES(
  (SELECT value FROM v_config WHERE key='count') = '0'
);
INSERT INTO assert VALUES(
  (SELECT value FROM v_config WHERE key='dims') = '3'
);

-- ── Insert one, delete all, query again ─────────────────────────────────
INSERT INTO v(vector) VALUES(vec('[1.0,2.0,3.0]'));
DELETE FROM v WHERE rowid=1;

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v) = 0
);
INSERT INTO assert VALUES(
  (SELECT value FROM v_config WHERE key='count') = '0'
);

-- KNN on re-emptied table
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM (
    SELECT rowid FROM v WHERE v MATCH '[1.0,0.0,0.0]' LIMIT 5
  )) = 0
);

-- ── Insert after emptying ───────────────────────────────────────────────
INSERT INTO v(vector) VALUES(vec('[1.0,0.0,0.0]'));
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v) = 1
);

-- KNN works normally again
INSERT INTO assert VALUES(
  (SELECT rowid FROM v WHERE v MATCH '[1.0,0.0,0.0]' LIMIT 1) IS NOT NULL
);

SELECT 'empty_table: all tests passed';
