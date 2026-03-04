-- test/shadow_setup.sql
-- Creates an on-disk database with a vec0 virtual table for xConnect testing.
-- Run as: sqlite3 /tmp/sv_connect_test.db < test/shadow_setup.sql

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE VIRTUAL TABLE items USING vec0(dims=4, metric=cosine, m=8, ef_construction=50, ef_search=5);

SELECT 'shadow setup done';
