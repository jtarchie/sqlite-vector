-- test/shadow.sql
-- Verifies that CREATE VIRTUAL TABLE creates the four shadow tables,
-- that config values are persisted correctly, and that DROP VIRTUAL TABLE
-- removes all shadow tables.
--
-- Run as: sqlite3 /tmp/sv_shadow_test.db < test/shadow.sql

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- ── xCreate: shadow tables must exist ─────────────────────────────────────
CREATE VIRTUAL TABLE items USING vec0(dims=4, metric=cosine, m=8, ef_construction=50, ef_search=5);

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_config') = 1
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_data') = 1
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_graph') = 1
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_layers') = 1
);

-- ── Config values persisted correctly ─────────────────────────────────────
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='dims') = '4'
);
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='metric') = 'cosine'
);
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='m') = '8'
);
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='ef_construction') = '50'
);
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='ef_search') = '5'
);
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='entry_point') = '-1'
);
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='count') = '0'
);

-- ── xDestroy: shadow tables must be gone ──────────────────────────────────
DROP TABLE items;

INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_config') = 0
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_data') = 0
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_graph') = 0
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='items_layers') = 0
);

SELECT 'shadow tests passed';
