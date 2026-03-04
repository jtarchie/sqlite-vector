-- test/vec_parse.sql
-- Tests for vec(), vec_dims(), vec_norm() scalar functions.
-- Run: sqlite3 :memory: < test/vec_parse.sql
-- Uses a temp table with CHECK(val) as an assertion mechanism:
--   INSERT succeeds  → assertion passed (val = 1)
--   INSERT fails     → CHECK constraint error (val = 0), aborted by .bail on

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- ── vec() ──────────────────────────────────────────────────────────────────

-- basic normalisation: round-trips and strips redundant trailing zeros
INSERT INTO assert VALUES (vec('[1.0,2.0,3.0]') = '[1,2,3]');
DELETE FROM assert;

-- handles spaces around values and commas
INSERT INTO assert VALUES (vec('[ 1.0 , 2.0 , 3.0 ]') = '[1,2,3]');
DELETE FROM assert;

-- single-element vector
INSERT INTO assert VALUES (vec('[42.0]') = '[42]');
DELETE FROM assert;

-- negative and fractional values
INSERT INTO assert VALUES (vec('[-1.5,0.5]') = '[-1.5,0.5]');
DELETE FROM assert;

-- ── vec_dims() ─────────────────────────────────────────────────────────────

INSERT INTO assert VALUES (vec_dims('[1,2,3]') = 3);
DELETE FROM assert;

INSERT INTO assert VALUES (vec_dims('[0.0]') = 1);
DELETE FROM assert;

INSERT INTO assert VALUES (vec_dims('[1,2,3,4,5,6,7,8]') = 8);
DELETE FROM assert;

-- ── vec_norm() ─────────────────────────────────────────────────────────────

-- 3-4-5 right triangle: norm([3,4]) = 5
INSERT INTO assert VALUES (abs(vec_norm('[3.0,4.0]') - 5.0) < 1e-5);
DELETE FROM assert;

-- unit vector along x: norm = 1
INSERT INTO assert VALUES (abs(vec_norm('[1.0,0.0,0.0]') - 1.0) < 1e-5);
DELETE FROM assert;

-- zero vector: norm = 0
INSERT INTO assert VALUES (vec_norm('[0.0,0.0,0.0]') = 0.0);
DELETE FROM assert;

-- ── NULL inputs ────────────────────────────────────────────────────────────

INSERT INTO assert VALUES (vec(NULL) IS NULL);
DELETE FROM assert;

INSERT INTO assert VALUES (vec_dims(NULL) IS NULL);
DELETE FROM assert;

INSERT INTO assert VALUES (vec_norm(NULL) IS NULL);
DELETE FROM assert;

SELECT 'vec_parse: all tests passed';
