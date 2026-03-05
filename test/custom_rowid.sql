-- test/custom_rowid.sql
-- Verifies that users can specify explicit rowids when inserting vectors
-- to link them to other tables/content.
--
-- This is useful when you want to:
-- 1. Link vectors to IDs from another table
-- 2. Maintain consistent rowids across database migrations
-- 3. Reference vectors from external identifiers
.bail on.load build / macosx / arm64 / release / libsqlite_vector CREATE TEMP TABLE assert(val INTEGER NOT NULL CHECK(val));

-- Create a vector table
CREATE VIRTUAL TABLE vectors USING vec0(dims = 3, metric = cosine);

-- Create a related table to link to (e.g., products)
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT
);

-- ── Test 1: Auto-assigned rowids (default behavior) ─────────────────────────
INSERT INTO
  vectors(vector)
VALUES
(vec('[1.0, 0.0, 0.0]'));

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        vectors_data
    ) = 1
  );

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        MAX(id)
      FROM
        vectors_data
    ) = 1
  );

-- ── Test 2: Explicit rowid assignment ──────────────────────────────────────
-- Insert with explicit rowid=100 to match a related products.id
INSERT INTO
  vectors(rowid, vector)
VALUES
(100, vec('[0.0, 1.0, 0.0]'));

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        vectors_data
    ) = 2
  );

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        id
      FROM
        vectors_data
      WHERE
        id = 100
    ) = 100
  );

-- ── Test 3: Multiple explicit rowids ──────────────────────────────────────
INSERT INTO
  vectors(rowid, vector)
VALUES
(50, vec('[0.1, 0.9, 0.0]'));

INSERT INTO
  vectors(rowid, vector)
VALUES
(75, vec('[0.2, 0.8, 0.0]'));

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        vectors_data
    ) = 4
  );

-- ── Test 4: Mix auto and explicit rowids ──────────────────────────────────
INSERT INTO
  vectors(vector)
VALUES
(vec('[0.3, 0.7, 0.0]'));

-- auto
INSERT INTO
  vectors(rowid, vector)
VALUES
(200, vec('[0.4, 0.6, 0.0]'));

-- explicit
INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        vectors_data
    ) = 6
  );

-- ── Test 5: Linking vectors to related table via rowid ───────────────────
INSERT INTO
  products(id, name, description)
VALUES
(100, 'Product A', 'A great product');

INSERT INTO
  products(id, name, description)
VALUES
(50, 'Product B', 'Another great product');

-- Now we can join vectors to products using rowid
CREATE TEMP TABLE joined AS
SELECT
  p.id,
  p.name,
  v.vector
FROM
  products p
  JOIN vectors_data v ON p.id = v.id;

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        joined
    ) = 2
  );

-- ── Test 6: KNN query preserves rowid linkage ─────────────────────────────
-- Search for similar vectors and get the linked product info
CREATE TEMP TABLE knn_results AS
SELECT
  v.id,
  p.name,
  v.vector,
  v.distance
FROM
  (
    SELECT
      *
    FROM
      vectors
    WHERE
      vectors MATCH '[1.0, 0.0, 0.0]'
    LIMIT
      3
  ) v
  LEFT JOIN products p ON v.id = p.id;

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        knn_results
    ) > 0
  );

-- Verify the linked product name is available
INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        knn_results
      WHERE
        name IS NOT NULL
    ) > 0
  );

-- ── Test 7: Delete with explicit rowid preserves linkage integrity ───────
DELETE FROM
  vectors
WHERE
  rowid = 50;

-- Verify product 50 is still there but vector is gone
INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        products
      WHERE
        id = 50
    ) = 1
  );

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        vectors_data
      WHERE
        id = 50
    ) = 0
  );

-- ── Test 8: Update with explicit rowid (should maintain rowid) ───────────
UPDATE
  vectors
SET
  vector = vec('[0.5, 0.5, 0.0]')
WHERE
  rowid = 100;

-- Verify the vector is updated but rowid unchanged
INSERT INTO
  assert
VALUES
(
    (
      SELECT
        id
      FROM
        vectors_data
      WHERE
        id = 100
    ) = 100
  );

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        products
      WHERE
        id = 100
    ) = 1
  );

-- Verify we can still join
CREATE TEMP TABLE check_link AS
SELECT
  p.id,
  p.name
FROM
  products p
  JOIN vectors_data v ON p.id = v.id
WHERE
  p.id = 100;

INSERT INTO
  assert
VALUES
(
    (
      SELECT
        COUNT(*)
      FROM
        check_link
    ) = 1
  );

SELECT
  'custom rowid tests passed' AS result;