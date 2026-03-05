-- test/vector_types.sql
-- Test typed vector constructors and vec_type() function

.load build/macosx/arm64/release/libsqlite_vector.dylib
.bail on

.print "=== vec_type() tests ==="

-- Text vector (no subtype)
SELECT vec_type('[1,2,3]') AS type;
-- Expected: text

-- Float32 vector
SELECT vec_type(vec_f32('[1,2,3]')) AS type;
-- Expected: float32

-- Int8 vector
SELECT vec_type(vec_int8('[1,2,3]')) AS type;
-- Expected: int8

-- Bit vector
SELECT vec_type(vec_bit(X'FF00AA55')) AS type;
-- Expected: bit

-- NULL handling
SELECT vec_type(NULL) AS type;
-- Expected: NULL

.print ""
.print "=== vec_f32() tests ==="

-- Convert text to float32
SELECT typeof(vec_f32('[1,2,3]')) AS type, length(vec_f32('[1,2,3]')) AS bytes;
-- Expected: blob, 12 (3 floats × 4 bytes)

-- Convert blob to float32 (just tags it)
SELECT vec_type(vec_f32(X'0000803F')) AS type;
-- Expected: float32

-- NULL handling
SELECT vec_f32(NULL) AS result;
-- Expected: NULL

.print ""
.print "=== vec_int8() tests ==="

-- Convert text to int8
SELECT typeof(vec_int8('[1,2,3]')) AS type, length(vec_int8('[1,2,3]')) AS bytes;
-- Expected: blob, 3

-- Test clamping to int8 range [-128, 127]
SELECT hex(vec_int8('[127, -128, 0]')) AS clamped;
-- Expected: 7F8000

SELECT hex(vec_int8('[200, -200, 50]')) AS clamped;
-- Expected: 7F8032 (200→127, -200→-128, 50→50)

-- Convert blob to int8 (just tags it)
SELECT vec_type(vec_int8(X'010203')) AS type;
-- Expected: int8

-- NULL handling
SELECT vec_int8(NULL) AS result;
-- Expected: NULL

.print ""
.print "=== vec_bit() tests ==="

-- Tag a bit-packed blob
SELECT typeof(vec_bit(X'FF00AA55')) AS type;
-- Expected: blob

SELECT vec_type(vec_bit(X'FF00')) AS type;
-- Expected: bit

-- Error: non-blob input (uncomment to test manually)
-- SELECT vec_bit('[1,2,3]');

.print ""
.print "=== Combined typed vector tests ==="

-- Note: SQLite subtypes are ephemeral and not preserved in storage.
-- They exist only during expression evaluation within a single statement.
-- When stored in a column, the subtype is lost and only the BLOB remains.

CREATE TABLE typed_vecs(
  id INTEGER PRIMARY KEY,
  vec_text TEXT,
  vec_f32 BLOB,
  vec_int8 BLOB,
  vec_bit BLOB
);

INSERT INTO typed_vecs VALUES (
  1,
  '[1,2,3]',
  vec_f32('[1,2,3]'),
  vec_int8('[1,2,3]'),
  vec_bit(X'FF00')
);

-- Subtypes are lost after storage (expected behavior)
SELECT 
  vec_type(vec_text) AS text_type,
  vec_type(vec_f32) AS f32_type_stored,
  vec_type(vec_int8) AS int8_type_stored,
  vec_type(vec_bit) AS bit_type_stored
FROM typed_vecs WHERE id = 1;
-- Expected: text, unknown, unknown, unknown (subtypes not preserved in storage)

-- Subtypes work when applied in-query
SELECT 
  vec_type(vec_f32(vec_f32)) AS f32_reapplied,
  vec_type(vec_int8(vec_int8)) AS int8_reapplied,
  vec_type(vec_bit(vec_bit)) AS bit_reapplied
FROM typed_vecs WHERE id = 1;
-- Expected: float32, int8, bit (subtypes reapplied during query)

.print ""
.print "=== Subtype ephemeral behavior ==="

-- Subtypes work in single expression
SELECT vec_type(vec_f32('[1,2,3]')) AS inline_type;
-- Expected: float32

-- Can chain operations while preserving subtypes
SELECT vec_type(vec_f32(vec_int8('[1,2,3]'))) AS chained;
-- Expected: float32 (last applied subtype wins)

.print ""
.print "=== All vector type tests complete ==="
