-- test/atomic_update.sql
-- Verifies xBegin/xCommit/xRollback deferred config writes and that a
-- ROLLBACK correctly restores in-memory state.
--
-- Scenarios tested:
--   1. BEGIN..COMMIT: _config count deferred during txn, flushed at COMMIT.
--   2. BEGIN..ROLLBACK: count and graph revert; kNN still finds original rows.
--   3. DELETE inside committed txn reduces count and removes the entry.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, ef_search=50);

-- Seed 5 known vectors.
INSERT INTO vecs(vector) VALUES(vec('[1.0, 0.0, 0.0]'));   -- rowid 1
INSERT INTO vecs(vector) VALUES(vec('[0.0, 1.0, 0.0]'));   -- rowid 2
INSERT INTO vecs(vector) VALUES(vec('[0.0, 0.0, 1.0]'));   -- rowid 3
INSERT INTO vecs(vector) VALUES(vec('[0.9, 0.1, 0.0]'));   -- rowid 4
INSERT INTO vecs(vector) VALUES(vec('[0.0, 0.5, 0.5]'));   -- rowid 5

INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '5'
);

-- ── Scenario 1: BEGIN..COMMIT flushes deferred config ────────────────────────
BEGIN;
INSERT INTO vecs(vector) VALUES(vec('[0.3, 0.3, 0.3]'));   -- rowid 6
INSERT INTO vecs(vector) VALUES(vec('[0.7, 0.7, 0.0]'));   -- rowid 7

-- During txn, _config still reflects pre-txn value (5).
INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '5'
);
COMMIT;

-- After commit, _config must show 7.
INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '7'
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM vecs_data) = 7
);

-- kNN works and finds the closest vector to [1,0,0].
CREATE TEMP TABLE after_commit AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 3;
INSERT INTO assert VALUES((SELECT COUNT(*) FROM after_commit) >= 1);
INSERT INTO assert VALUES(
  (SELECT rowid FROM after_commit ORDER BY distance LIMIT 1) = 1
);

-- ── Scenario 2: BEGIN..ROLLBACK reverts count; kNN still works ───────────────
BEGIN;
INSERT INTO vecs(vector) VALUES(vec('[0.2, 0.8, 0.0]'));   -- would be rowid 8
INSERT INTO vecs(vector) VALUES(vec('[0.8, 0.2, 0.0]'));   -- would be rowid 9
DELETE FROM vecs WHERE rowid = 1;

-- Inside txn, _config still shows pre-txn value (7 - deferred).
INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '7'
);
ROLLBACK;

-- After rollback, count must revert to 7 and rowid 1 must be back.
INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '7'
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM vecs_data) = 7
);

CREATE TEMP TABLE after_rollback AS
  SELECT rowid, distance FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 3;
INSERT INTO assert VALUES((SELECT COUNT(*) FROM after_rollback) >= 1);
INSERT INTO assert VALUES(
  (SELECT rowid FROM after_rollback ORDER BY distance LIMIT 1) = 1
);

-- ── Scenario 3: DELETE in committed txn reduces count correctly ───────────────
BEGIN;
DELETE FROM vecs WHERE rowid = 7;
COMMIT;

INSERT INTO assert VALUES(
  (SELECT value FROM vecs_config WHERE key='count') = '6'
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM vecs_data) = 6
);
-- entry_point must not be the deleted rowid
INSERT INTO assert VALUES(
  (SELECT CAST(value AS INTEGER) FROM vecs_config WHERE key='entry_point') != 7
);
INSERT INTO assert VALUES(
  (SELECT CAST(value AS INTEGER) FROM vecs_config WHERE key='entry_point') >= 1
);

SELECT 'atomic_update tests passed';
