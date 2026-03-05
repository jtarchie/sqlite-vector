-- test/hamming_jaccard.sql
-- Documents current vec_distance_hamming/jaccard behavior.
-- Current implementation operates on float32 byte representation.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- identical values have zero distance
INSERT INTO assert VALUES (vec_distance_hamming('[0]', '[0]') = 0.0);
INSERT INTO assert VALUES (vec_distance_jaccard('[0]', '[0]') = 1.0);

-- +0.0 vs -0.0 differs by sign bit only (one bit per float element)
INSERT INTO assert VALUES (vec_distance_hamming('[0]', '[-0]') = 1.0);
INSERT INTO assert VALUES (vec_distance_hamming('[0,0]', '[-0,-0]') = 2.0);

-- jaccard distance on those bit patterns is maximal (disjoint set bits)
INSERT INTO assert VALUES (vec_distance_jaccard('[0]', '[-0]') = 1.0);
INSERT INTO assert VALUES (vec_distance_jaccard('[0,0]', '[-0,-0]') = 1.0);

-- NULL propagation
INSERT INTO assert VALUES (vec_distance_hamming(NULL, '[0]') IS NULL);
INSERT INTO assert VALUES (vec_distance_jaccard('[0]', NULL) IS NULL);

SELECT 'hamming_jaccard: all tests passed';
