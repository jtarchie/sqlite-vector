-- test/vec_ops.sql
-- Test element-wise vector operations and utility functions

.load build/macosx/arm64/release/libsqlite_vector.dylib

.print "=== vec_add tests ==="

-- Basic addition
SELECT vec_add('[1,2,3]', '[4,5,6]') AS result;
-- Expected: [5,7,9]

-- Addition with negative values
SELECT vec_add('[1,-2,3]', '[-1,2,-3]') AS result;
-- Expected: [0,0,0]

-- Addition with floats
SELECT vec_add('[1.5,2.5]', '[0.5,1.5]') AS result;
-- Expected: [2,4]

-- Error tests (uncomment to test manually):
-- Dimension mismatch should fail:
--   SELECT vec_add('[1,2,3]', '[4,5]');

-- NULL handling
SELECT vec_add(NULL, '[1,2,3]') AS result;
-- Expected: NULL

SELECT vec_add('[1,2,3]', NULL) AS result;
-- Expected: NULL

.print "=== vec_sub tests ==="

-- Basic subtraction
SELECT vec_sub('[5,7,9]', '[1,2,3]') AS result;
-- Expected: [4,5,6]

-- Subtraction with negative values
SELECT vec_sub('[1,2,3]', '[2,3,4]') AS result;
-- Expected: [-1,-1,-1]

-- Subtraction with floats
SELECT vec_sub('[3.5,5.0]', '[1.5,2.0]') AS result;
-- Expected: [2,3]

-- Error tests (uncomment to test manually):
-- Dimension mismatch should fail:
--   SELECT vec_sub('[1,2,3]', '[4,5]');

-- NULL handling
SELECT vec_sub(NULL, '[1,2,3]') AS result;
-- Expected: NULL

.print "=== vec_normalize tests ==="

-- Normalize a simple vector
SELECT vec_normalize('[3,4]') AS result;
-- Expected: [0.6,0.8] (3/5, 4/5)

-- Normalize a unit vector (should stay the same)
SELECT vec_normalize('[1,0,0]') AS result;
-- Expected: [1,0,0]

-- Normalize a 3D vector
SELECT vec_normalize('[1,1,1]') AS result;
-- Expected: approximately [0.577,0.577,0.577] (1/sqrt(3) each)

-- Verify normalized vector has unit length by checking L2 distance from origin
-- ||v|| = sqrt(sum(v[i]^2)) = 1 for unit vector
-- We can verify using: vec_distance_l2(normalized_vec, '[0,0,...]') should be close to 1
SELECT CAST(vec_distance_l2(vec_normalize('[3,4]'), '[0,0]') AS TEXT) AS norm;
-- Expected: close to 1.0

-- Error tests (uncomment to test manually):
-- Zero vector should fail:
--   SELECT vec_normalize('[0,0,0]');

-- NULL handling
SELECT vec_normalize(NULL) AS result;
-- Expected: NULL

.print "=== vec_slice tests ==="

-- Basic slice from middle
SELECT vec_slice('[1,2,3,4,5]', 1, 4) AS result;
-- Expected: [2,3,4] (elements at indices 1, 2, 3)

-- Slice entire vector
SELECT vec_slice('[1,2,3]', 0, 3) AS result;
-- Expected: [1,2,3]

-- Slice first element
SELECT vec_slice('[10,20,30]', 0, 1) AS result;
-- Expected: [10]

-- Slice last element
SELECT vec_slice('[10,20,30]', 2, 3) AS result;
-- Expected: [30]

-- Slice first two elements
SELECT vec_slice('[1,2,3,4]', 0, 2) AS result;
-- Expected: [1,2]

-- Matryoshka embedding use case: select first 64 dims from 128-dim vector
-- (Simulating with smaller numbers)
SELECT vec_dims(vec_slice('[1,2,3,4,5,6,7,8]', 0, 4)) AS dims;
-- Expected: 4

-- Error tests (uncomment to test manually):
-- Empty slice should fail:
--   SELECT vec_slice('[1,2,3]', 2, 2);
-- Invalid indices should fail:
--   SELECT vec_slice('[1,2,3]', -1, 2);
--   SELECT vec_slice('[1,2,3]', 0, 5);
--   SELECT vec_slice('[1,2,3]', 2, 1);

-- NULL handling
SELECT vec_slice(NULL, 0, 2) AS result;
-- Expected: NULL

.print "=== Combined operations tests ==="

-- Add two vectors then normalize
SELECT vec_normalize(vec_add('[1,0,0]', '[0,1,0]')) AS result;
-- Expected: approximately [0.707,0.707,0] (normalized [1,1,0])

-- Subtract and then take slice
SELECT vec_slice(vec_sub('[5,6,7,8]', '[1,2,3,4]'), 0, 2) AS result;
-- Expected: [4,4]

-- Distance between normalized vectors
SELECT vec_distance_l2(
    vec_normalize('[1,1,0]'),
    vec_normalize('[1,0,0]')
) AS distance;
-- Expected: should be consistent

-- Verify that vec_add and vec_sub are inverses
SELECT vec_sub(vec_add('[1,2,3]', '[4,5,6]'), '[4,5,6]') AS result;
-- Expected: [1,2,3]

.print "=== All vec_ops tests complete ==="
