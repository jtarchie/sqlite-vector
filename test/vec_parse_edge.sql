-- test/vec_parse_edge.sql
-- Edge-case behavior tests for vec() parser/formatter.

.bail on
.load build/macosx/arm64/release/libsqlite_vector

CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- trailing comma currently normalizes to a single-element vector
INSERT INTO assert VALUES (vec('[1,]') = '[1]');

-- repeated comma currently skips empty element and keeps parsed values
INSERT INTO assert VALUES (vec('[1,,3]') = '[1,3]');

-- trailing characters after closing bracket are currently ignored
INSERT INTO assert VALUES (vec('[1,2,3]9') = '[1,2,3]');

-- special float values are accepted and preserved
INSERT INTO assert VALUES (vec('[inf]') = '[inf]');
INSERT INTO assert VALUES (vec('[nan]') = '[nan]');
INSERT INTO assert VALUES (vec('[-inf]') = '[-inf]');

-- large and tiny exponent values format as float32
INSERT INTO assert VALUES (vec('[1.5e38]') = '[1.5e+38]');
INSERT INTO assert VALUES (vec('[1e-46]') = '[0]');

SELECT 'vec_parse_edge: all tests passed';
