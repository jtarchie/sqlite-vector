-- test/shadow_connect.sql
-- Verifies xConnect: opens an existing database with shadow tables and
-- reads config back without needing argv key=value pairs.
-- Run AFTER test/shadow_setup.sql has created the database.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- If xConnect works, the virtual table should be usable and dims should be 4
-- (read from _config, not from CREATE VIRTUAL TABLE argv).
INSERT INTO assert VALUES(vec_dims('[1,2,3,4]') = 4);

-- Query the virtual table; returns empty but must not error.
SELECT COUNT(*) FROM items WHERE items MATCH '[0.1,0.2,0.3,0.4]';

-- Config must still be readable via the shadow table.
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='dims') = '4'
);
INSERT INTO assert VALUES(
  (SELECT value FROM items_config WHERE key='metric') = 'cosine'
);

SELECT 'shadow xConnect tests passed';
