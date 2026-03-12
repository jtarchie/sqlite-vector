-- test/shadow_lifecycle.sql
-- Tests shadow table state transitions across INSERT, DELETE, UPDATE, DROP,
-- and re-CREATE. Extends shadow.sql which covers schema/config/DROP.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- ── Create table ──────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=l2, m=8);

-- ── After INSERT: data, graph, layers populated; config count updated ─────
INSERT INTO v(vector) VALUES('[1,0,0,0]');
INSERT INTO v(vector) VALUES('[0,1,0,0]');
INSERT INTO v(vector) VALUES('[0,0,1,0]');

-- _data has 3 rows
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_data) = 3);

-- _layers has 3 entries
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_layers) = 3);

-- _graph has edges (at least layer 0)
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_graph) > 0);

-- _config count = 3
INSERT INTO assert VALUES((SELECT value FROM v_config WHERE key='count') = '3');

-- entry_point is a valid data id
INSERT INTO assert VALUES(
  (SELECT value FROM v_config WHERE key='entry_point') IN
    (SELECT id FROM v_data)
);

-- ── After DELETE: data, graph, layers cleaned up; config count decremented ─
DELETE FROM v WHERE rowid = 2;

-- _data has 2 rows
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_data) = 2);

-- _layers has 2 entries
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_layers) = 2);

-- _config count = 2
INSERT INTO assert VALUES((SELECT value FROM v_config WHERE key='count') = '2');

-- Deleted node's edges are gone
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v_graph WHERE node_id = 2 OR neighbor_id = 2) = 0
);

-- No orphan edges (all node_ids and neighbor_ids reference existing data)
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM v_graph
   WHERE node_id NOT IN (SELECT id FROM v_data)
      OR neighbor_id NOT IN (SELECT id FROM v_data)) = 0
);

-- ── After UPDATE: data changed, graph intact ──────────────────────────────
UPDATE v SET vector = '[0.5,0.5,0,0]' WHERE rowid = 1;

-- Count unchanged
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_data) = 2);

-- _config count unchanged
INSERT INTO assert VALUES((SELECT value FROM v_config WHERE key='count') = '2');

-- ── After DROP: all shadow tables gone ────────────────────────────────────
DROP TABLE v;

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'v_%') = 0
);

-- ── Re-CREATE after DROP: clean state ────────────────────────────────────
CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=l2);

INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_data) = 0);
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_graph) = 0);
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_layers) = 0);
INSERT INTO assert VALUES((SELECT value FROM v_config WHERE key='count') = '0');
INSERT INTO assert VALUES((SELECT value FROM v_config WHERE key='entry_point') = '-1');

-- Insert works on re-created table
INSERT INTO v(vector) VALUES('[1,1,1,1]');
INSERT INTO assert VALUES((SELECT COUNT(*) FROM v_data) = 1);

DROP TABLE v;

SELECT 'shadow_lifecycle: all tests passed';
