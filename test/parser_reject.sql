-- test/parser_reject.sql
-- Verifies parser acceptance of valid inputs. Error rejection is tested
-- in ffi_test.lua where error codes can be checked without aborting.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=l2);
INSERT INTO v(vector) VALUES(vec('[1.0,2.0,3.0]'));

-- Known-good baseline: 1 row
INSERT INTO assert VALUES(
  (SELECT value FROM v_config WHERE key='count') = '1'
);

-- ── Scalar distance function accepts valid input ────────────────────────
INSERT INTO assert VALUES(
  vec_distance_l2('[1.0,2.0,3.0]','[4.0,5.0,6.0]') >= 0.0
);

-- ── vec_dims works on valid inputs ──────────────────────────────────────
INSERT INTO assert VALUES(
  vec_dims(vec('[1.0,2.0,3.0]')) = 3
);
INSERT INTO assert VALUES(
  vec_dims(vec('[1.0]')) = 1
);

-- ── vec round-trips ─────────────────────────────────────────────────────
INSERT INTO assert VALUES(
  vec('[1.0,2.0,3.0]') = '[1,2,3]'
);
INSERT INTO assert VALUES(
  vec('[0.5,-0.5,1e10]') IS NOT NULL
);

-- ── Whitespace in vectors ───────────────────────────────────────────────
INSERT INTO assert VALUES(
  vec('[ 1.0 , 2.0 , 3.0 ]') = '[1,2,3]'
);

-- ── vec_int8 constructor ────────────────────────────────────────────────
INSERT INTO assert VALUES(
  vec_int8('[1,2,3]') IS NOT NULL
);

-- ── vec_bit constructor (requires BLOB input) ──────────────────────────
INSERT INTO assert VALUES(
  vec_bit(X'FF00') IS NOT NULL
);

SELECT 'parser_reject: all tests passed';
