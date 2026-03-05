-- test/rename.sql
-- Verifies ALTER TABLE ... RENAME TO for vec0 virtual tables.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, ef_search=16);
INSERT INTO vecs(vector) VALUES(vec('[1.0,0.0,0.0]'));
INSERT INTO vecs(vector) VALUES(vec('[0.0,1.0,0.0]'));
INSERT INTO vecs(vector) VALUES(vec('[0.0,0.0,1.0]'));

ALTER TABLE vecs RENAME TO vecs2;

-- old shadow tables are gone
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs_config') = 0
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs_data') = 0
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs_graph') = 0
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs_layers') = 0
);

-- new shadow tables exist
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs2_config') = 1
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs2_data') = 1
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs2_graph') = 1
);
INSERT INTO assert VALUES(
  (SELECT COUNT(*) FROM sqlite_master WHERE name='vecs2_layers') = 1
);

-- kNN still works on renamed table
CREATE TEMP TABLE knn_after_rename AS
  SELECT rowid, distance FROM vecs2 WHERE vecs2 MATCH '[1.0,0.0,0.0]' LIMIT 3;

INSERT INTO assert VALUES((SELECT COUNT(*) FROM knn_after_rename) >= 1);
INSERT INTO assert VALUES(
  (SELECT rowid FROM knn_after_rename ORDER BY distance LIMIT 1) = 1
);

-- updates after rename still flow into renamed shadow tables
INSERT INTO vecs2(vector) VALUES(vec('[0.9,0.1,0.0]'));
INSERT INTO assert VALUES(
  (SELECT value FROM vecs2_config WHERE key='count') = '4'
);

SELECT 'rename tests passed';
