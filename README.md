# sqlite-vector

Vector similarity search for SQLite

Store your vectors with the rest of your data and run approximate nearest
neighbor searches with HNSW indexing. Supports ACID transactions, multiple
distance metrics, and works seamlessly with SQLite's embedded architecture.

## Highlights

- **HNSW indexing** - Fast approximate nearest neighbor search
- **Multiple distance metrics** - L2, cosine, inner product, L1, Hamming,
  Jaccard
- **ACID compliant** - Full transaction support with rollback safety
- **Embedded database** - No server setup, runs in-process
- **Vector operations** - Add, subtract, normalize, slice vectors
- **Hardware accelerated** - SIMD optimizations via SimSIMD
- **SQL-native** - No in-memory index, all data in SQLite tables

## Why sqlite-vector?

Unlike PostgreSQL's pgvector which requires a database server, sqlite-vector
brings vector search to SQLite's embedded, serverless architecture. Perfect for:

- **Mobile and edge applications** - Vector search on device
- **Desktop applications** - Single-file databases with vector capabilities
- **Prototyping** - Quick setup without server infrastructure
- **Embedded AI** - Local vector embeddings with your application data

## Installation

### Building from Source

**Prerequisites:**

- [xmake](https://xmake.io/) build system
- C11 compiler (gcc, clang, or MSVC)
- SQLite 3.x headers (included in `include/`)

**Linux / macOS:**

```bash
git clone https://github.com/yourusername/sqlite-vector.git
cd sqlite-vector
xmake
```

The extension will be built to
`build/[platform]/[arch]/release/libsqlite_vector.[so|dylib|dll]`

**Windows:**

```bash
git clone https://github.com/yourusername/sqlite-vector.git
cd sqlite-vector
xmake f -p mingw
xmake
```

### Loading the Extension

**SQLite CLI:**

```bash
sqlite3 mydb.db
```

```sql
.load build/macosx/arm64/release/libsqlite_vector
```

**C API:**

```c
#include <sqlite3.h>

sqlite3 *db;
sqlite3_open("mydb.db", &db);

char *err_msg = NULL;
sqlite3_load_extension(db, "./libsqlite_vector.so", NULL, &err_msg);
```

**Python (sqlite3):**

```python
import sqlite3

conn = sqlite3.connect("mydb.db")
conn.enable_load_extension(True)
conn.load_extension("./libsqlite_vector.so")
```

## Getting Started

Create a virtual table to store vectors:

```sql
CREATE VIRTUAL TABLE items USING vec0(
  dims=3,            -- 3-dimensional vectors
  metric=cosine      -- cosine distance
);
```

Insert vectors:

```sql
INSERT INTO items(vector) VALUES(vec('[1.0, 0.0, 0.0]'));
INSERT INTO items(vector) VALUES(vec('[0.0, 1.0, 0.0]'));
INSERT INTO items(vector) VALUES(vec('[0.0, 0.0, 1.0]'));
INSERT INTO items(vector) VALUES(vec('[0.9, 0.1, 0.0]'));
```

Find the nearest neighbors:

```sql
SELECT rowid, distance 
FROM items 
WHERE items MATCH '[1.0, 0.0, 0.0]' 
LIMIT 5;
```

Output:

```
rowid | distance
------|----------
1     | 0.0
4     | 0.005012...
2     | 1.0
3     | 1.0
```

## Simple Wikipedia Semantic Search

This repository now includes Ruby scripts to build an embedded semantic search
database from the Simple English Wikipedia dump using Ollama model
`embeddinggemma` and sqlite-vector.

### Prerequisites

1. Build the extension:

```bash
xmake
```

2. Install Ruby dependencies:

```bash
bundle install
```

3. Start Ollama and ensure the embedding model is available:

```bash
ollama pull embeddinggemma
ollama serve
```

### 1) Ingest Wikipedia and Build Vectors

Run the ingest script (default indexes a subset first: 10,000 articles):

```bash
bundle exec ruby scripts/wiki_ingest.rb
```

Useful flags:

```bash
bundle exec ruby scripts/wiki_ingest.rb \
  --max-articles 20000 \
  --db-path data/wiki_simple.db \
  --chunk-max-chars 1200 \
  --batch-size 100
```

For full-dump indexing:

```bash
bundle exec ruby scripts/wiki_ingest.rb --max-articles 0
```

### 2) Query by Semantic Similarity

Search and show article title plus matched content segment:

```bash
bundle exec ruby scripts/wiki_query.rb "how does photosynthesis work"
```

Tune top-k and HNSW query width:

```bash
bundle exec ruby scripts/wiki_query.rb "history of london" --k 8 --ef-search 120
```

### Script Outputs

- `scripts/wiki_ingest.rb`:
  - Downloads `simplewiki-latest-pages-articles.xml.bz2`
  - Parses articles
  - Chunks article content by paragraphs
  - Embeds chunks with Ollama `embeddinggemma`
  - Stores metadata in relational tables and vectors in a `vec0` table
- `scripts/wiki_query.rb`:
  - Embeds query text
  - Runs `MATCH` KNN search in sqlite-vector
  - Joins to article/chunk metadata and prints ranked results

## Creating Tables

Create a virtual table with the `vec0` module:

```sql
CREATE VIRTUAL TABLE table_name USING vec0(
  dims=N,                      -- REQUIRED: vector dimensions (1-8192)
  metric='metric_name',        -- Distance metric (default: 'cosine')
  m=16,                        -- HNSW M parameter (default: 16)
  ef_construction=200,         -- Build-time search width (default: 200)
  ef_search=10                 -- Query-time search width (default: 10)
);
```

### Parameters

| Parameter         | Required | Default  | Description                    |
| ----------------- | -------- | -------- | ------------------------------ |
| `dims`            | Yes      | -        | Vector dimensionality (1-8192) |
| `metric`          | No       | `cosine` | Distance metric to use         |
| `m`               | No       | `16`     | Max connections per HNSW layer |
| `ef_construction` | No       | `200`    | Beam width during index build  |
| `ef_search`       | No       | `10`     | Beam width during search       |

### Supported Metrics

| Metric    | Aliases                | Description                             | Use Case               |
| --------- | ---------------------- | --------------------------------------- | ---------------------- |
| `l2`      | `euclidean`            | Euclidean distance                      | Absolute similarity    |
| `cosine`  | `cos`                  | Cosine distance (1 - cosine similarity) | Normalized similarity  |
| `ip`      | `dot`, `inner_product` | Negative inner product                  | Dot product similarity |
| `l1`      | `manhattan`, `taxicab` | Manhattan distance                      | Grid-based distance    |
| `hamming` | -                      | Hamming distance                        | Binary vectors         |
| `jaccard` | -                      | Jaccard distance                        | Set similarity         |

### Examples

**L2 distance (Euclidean):**

```sql
CREATE VIRTUAL TABLE embeddings USING vec0(
  dims=1536,
  metric=l2
);
```

**Cosine distance (normalized):**

```sql
CREATE VIRTUAL TABLE documents USING vec0(
  dims=768,
  metric=cosine,
  ef_search=50           -- Higher recall at query time
);
```

**High-quality index:**

```sql
CREATE VIRTUAL TABLE images USING vec0(
  dims=512,
  metric=l2,
  m=32,                  -- More connections = better recall
  ef_construction=400    -- Slower build, better quality
);
```

## Storing Vectors

### Vector Format

Vectors are represented as text in JSON array format:

```sql
'[1.0, 2.0, 3.0, 4.0]'
```

Use the `vec()` function to parse and validate:

```sql
INSERT INTO items(vector) VALUES(vec('[1.0, 2.0, 3.0]'));
```

### Typed Vector Constructors

**Float32 vectors (default):**

```sql
INSERT INTO items(vector) VALUES(vec_f32('[1.0, 2.0, 3.0]'));
SELECT vec_type(vector) FROM items;  -- Returns 'float32'
```

**Int8 vectors (quantized):**

```sql
INSERT INTO items(vector) VALUES(vec_int8('[1, 2, 3]'));
SELECT vec_type(vector) FROM items;  -- Returns 'int8'
```

**Bit vectors (binary):**

```sql
INSERT INTO items(vector) VALUES(vec_bit(X'FF00'));
SELECT vec_type(vector) FROM items;  -- Returns 'bit'
```

### Bulk Insert

Wrap bulk inserts in transactions for better performance:

```sql
BEGIN;
  INSERT INTO items(vector) VALUES(vec('[1,0,0]'));
  INSERT INTO items(vector) VALUES(vec('[0,1,0]'));
  -- ... many more inserts ...
  INSERT INTO items(vector) VALUES(vec('[0,0,1]'));
COMMIT;
```

### Linking to Other Content with Explicit rowid

You can specify an explicit `rowid` when inserting vectors to link them to
related data:

```sql
-- Create a products table
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT,
  embedding_id INTEGER
);

-- Create vector table
CREATE VIRTUAL TABLE embeddings USING vec0(dims=768);

-- Insert a product
INSERT INTO products(id, name) VALUES(123, 'Laptop');

-- Later, insert the embedding with matching rowid
INSERT INTO embeddings(rowid, vector) VALUES(123, vec('[...]'));

-- Now you can join vectors to products directly
SELECT p.name, e.vector 
FROM products p
JOIN embeddings USING(rowid)
WHERE p.id = 123;
```

**Use cases for explicit rowids:**

- **Link to external IDs** - Match vector IDs to product IDs, user IDs, document
  IDs
- **Cross-database references** - Maintain consistent rowids across multiple
  databases
- **Relational integrity** - Ensure vectors correspond to specific entities
- **Data migration** - Preserve rowid mappings during schema changes

**Auto vs Explicit:** If you don't specify `rowid`, SQLite auto-assigns
sequential values:

```sql
INSERT INTO embeddings(vector) VALUES(vec('[...]'));  -- rowid auto-assigned

INSERT INTO embeddings(rowid, vector) VALUES(999, vec('[...]'));  -- explicit rowid
```

### Validation

Dimension mismatches are rejected:

```sql
CREATE VIRTUAL TABLE items USING vec0(dims=3);
INSERT INTO items(vector) VALUES(vec('[1,2,3,4]'));  -- Error: dimension mismatch
```

## Querying

### KNN Search with MATCH

The primary way to find nearest neighbors:

```sql
SELECT rowid, distance 
FROM items 
WHERE items MATCH '[1.0, 0.0, 0.0]' 
LIMIT 10;
```

The `distance` column is automatically available in MATCH queries.

### Full Table Scan

Query without a MATCH clause to scan all vectors:

```sql
SELECT rowid, vector FROM items;
```

### Point Lookup

Retrieve a specific vector by rowid:

```sql
SELECT vector FROM items WHERE rowid = 5;
```

### Runtime ef_search Override

Tune recall at query time:

```sql
SELECT rowid, distance 
FROM items 
WHERE items MATCH '[1.0, 0.0, 0.0]' AND ef_search = 200
LIMIT 10;
```

Higher `ef_search` values increase recall but take longer.

### Result Limit

The effective limit is bounded by `ef_search × 2`. If you need more results,
increase `ef_search`:

```sql
-- To get 100 results reliably, use ef_search >= 50
SELECT rowid, distance 
FROM items 
WHERE items MATCH query AND ef_search = 50
LIMIT 100;
```

## Distance Functions

Distance functions can be used standalone or for metric override in KNN queries.

### All Distance Functions

| Function                     | Description            | Range   |
| ---------------------------- | ---------------------- | ------- |
| `vec_distance_l2(a, b)`      | Euclidean distance     | [0, ∞)  |
| `vec_distance_cosine(a, b)`  | Cosine distance        | [0, 2]  |
| `vec_distance_ip(a, b)`      | Negative inner product | (-∞, ∞) |
| `vec_distance_l1(a, b)`      | Manhattan distance     | [0, ∞)  |
| `vec_distance_hamming(a, b)` | Hamming distance       | [0, n]  |
| `vec_distance_jaccard(a, b)` | Jaccard distance       | [0, 1]  |

### Standalone Usage

Calculate distances without a MATCH query:

```sql
SELECT vec_distance_l2('[3, 0]', '[0, 4]');        -- 5.0
SELECT vec_distance_cosine('[1, 0]', '[0, 1]');    -- 1.0
SELECT vec_distance_ip('[1, 2, 3]', '[4, 5, 6]');  -- -32.0
```

### Metric Override in KNN

Use a different metric than the table's default:

```sql
-- Table uses cosine, but query with L2
CREATE VIRTUAL TABLE items USING vec0(dims=3, metric=cosine);

SELECT rowid 
FROM items 
WHERE vec_distance_l2(items, '[1, 0, 0]') 
LIMIT 10;
```

### When to Use Each Metric

**L2 (Euclidean):**

- Absolute distance matters
- Vectors are not normalized
- Image embeddings, spatial data

**Cosine:**

- Direction matters more than magnitude
- Text embeddings (BERT, GPT, etc.)
- Already normalized or need scale invariance

**Inner Product:**

- Pre-normalized vectors
- Approximate cosine when vectors are unit length
- Faster than cosine for normalized data

**L1 (Manhattan):**

- Sparse vectors
- Outlier robustness
- Grid-based or discrete spaces

**Hamming:**

- Binary vectors
- Bit-level similarity
- Locality-sensitive hashing

**Jaccard:**

- Set similarity
- Binary feature vectors
- Document intersection

## Vector Operations

### Element-wise Operations

**Addition:**

```sql
SELECT vec_add('[1, 2, 3]', '[4, 5, 6]');  -- [5, 7, 9]
```

**Subtraction:**

```sql
SELECT vec_sub('[5, 7, 9]', '[1, 2, 3]');  -- [4, 5, 6]
```

### Normalization

Normalize to unit length (L2 norm = 1):

```sql
SELECT vec_normalize('[3, 4]');  -- [0.6, 0.8]
```

Useful for cosine similarity when you want to use inner product:

```sql
INSERT INTO items(vector) VALUES(vec_normalize('[3, 4]'));
```

### Slicing

Extract a subvector (0-indexed, inclusive start, exclusive end):

```sql
SELECT vec_slice('[1, 2, 3, 4, 5]', 1, 4);  -- [2, 3, 4]
```

### Utility Functions

**Dimensionality:**

```sql
SELECT vec_dims('[1, 2, 3]');  -- 3
```

**L2 Norm:**

```sql
SELECT vec_norm('[3, 4]');  -- 5.0
```

**Type Introspection:**

```sql
SELECT vec_type(vec_f32('[1,2,3]'));  -- 'float32'
SELECT vec_type(vec_int8('[1,2,3]')); -- 'int8'
SELECT vec_type(vec_bit(X'FF00'));    -- 'bit'
```

### NULL Handling

Vector operations return NULL if any input is NULL:

```sql
SELECT vec_add('[1, 2]', NULL);      -- NULL
SELECT vec_distance_l2(NULL, '[1]'); -- NULL
```

## HNSW Indexing

sqlite-vector uses Hierarchical Navigable Small World (HNSW) graphs for
approximate nearest neighbor search. The index is stored in SQLite shadow
tables.

### How HNSW Works

HNSW creates a hierarchical graph structure:

- Multiple layers with decreasing density
- Each node connects to nearby neighbors
- Search starts at the top layer and descends
- Fast approximate search with high recall

### Parameters

**`m` - Connections per Layer**

Number of bi-directional links per node (default: 16).

- Higher values: Better recall, slower builds, more memory
- Lower values: Faster builds, less memory, lower recall
- Typical range: 8-64

```sql
CREATE VIRTUAL TABLE items USING vec0(dims=128, m=32);
```

**`ef_construction` - Build-Time Width**

Beam width during index construction (default: 200).

- Higher values: Better index quality, slower builds
- Lower values: Faster builds, lower recall
- Typical range: 100-800

```sql
CREATE VIRTUAL TABLE items USING vec0(dims=128, ef_construction=400);
```

**`ef_search` - Query-Time Width**

Beam width during search (default: 10).

- Higher values: Better recall, slower queries
- Lower values: Faster queries, lower recall
- Typical range: 10-500

```sql
-- Set at table creation
CREATE VIRTUAL TABLE items USING vec0(dims=128, ef_search=50);

-- Or override per query
SELECT rowid FROM items 
WHERE items MATCH '[...]' AND ef_search=200 
LIMIT 10;
```

### Tuning Guidelines

**For high recall (>95%):**

```sql
CREATE VIRTUAL TABLE items USING vec0(
  dims=768,
  m=32,
  ef_construction=400,
  ef_search=100
);
```

**For fast queries:**

```sql
CREATE VIRTUAL TABLE items USING vec0(
  dims=768,
  m=16,
  ef_construction=200,
  ef_search=10
);
```

**Balanced:**

```sql
CREATE VIRTUAL TABLE items USING vec0(
  dims=768,
  m=24,
  ef_construction=300,
  ef_search=50
);
```

### Build vs Query Tradeoffs

- Build with high `ef_construction` once for quality
- Tune `ef_search` per query for recall/speed balance
- Override `ef_search` at runtime to experiment

```sql
-- Production table with high-quality index
CREATE VIRTUAL TABLE items USING vec0(
  dims=1536,
  m=32,
  ef_construction=400,
  ef_search=10  -- Default for fast queries
);

-- Low-latency query
SELECT rowid FROM items 
WHERE items MATCH query AND ef_search=10 
LIMIT 5;

-- High-recall query
SELECT rowid FROM items 
WHERE items MATCH query AND ef_search=200 
LIMIT 5;
```

## Filtering

### Distance Constraints

Filter results by maximum distance:

```sql
SELECT rowid, distance 
FROM items 
WHERE items MATCH '[1.0, 0.0, 0.0]' AND distance < 0.5
LIMIT 10;
```

Only vectors within the distance threshold are returned.

### Limitations

Unlike pgvector's WHERE clause filtering, sqlite-vector currently only supports
distance-based filtering within MATCH queries. Arbitrary column filters are not
yet supported:

```sql
-- Not supported yet:
-- SELECT rowid FROM items 
-- WHERE items MATCH '[1,0,0]' AND category = 'books'
-- LIMIT 10;
```

**Workaround** - Post-filter in application code or use subqueries:

```sql
-- Get candidates with vector search
CREATE TEMP TABLE candidates AS
  SELECT rowid FROM items 
  WHERE items MATCH '[1,0,0]' 
  LIMIT 100;

-- Filter by other columns
SELECT i.* FROM items i
JOIN candidates c ON i.rowid = c.rowid
WHERE i.category = 'books'
LIMIT 10;
```

## Updates and Deletes

### Updating Vectors

Update a vector by rowid:

```sql
UPDATE items 
SET vector = vec('[4.0, 5.0, 6.0]') 
WHERE rowid = 1;
```

The HNSW graph is automatically updated.

### Deleting Vectors

Delete vectors normally:

```sql
DELETE FROM items WHERE rowid = 5;
```

**Graph Repair:** sqlite-vector automatically repairs the HNSW graph when nodes
are deleted:

- Disconnected neighbors are re-wired
- Entry point is re-elected if deleted
- Graph connectivity is maintained

### Transaction Safety

All operations are ACID compliant:

```sql
BEGIN;
  INSERT INTO items(vector) VALUES(vec('[1,0,0]'));
  INSERT INTO items(vector) VALUES(vec('[0,1,0]'));
  DELETE FROM items WHERE rowid = 3;
  UPDATE items SET vector = vec('[0,0,1]') WHERE rowid = 2;
  
  -- Something went wrong, roll back everything
  ROLLBACK;
```

Changes are atomic and isolated within transactions.

## Performance

### Index Build Optimization

**Use high ef_construction:**

```sql
CREATE VIRTUAL TABLE items USING vec0(
  dims=768,
  ef_construction=400  -- Better quality, worth the build time
);
```

**Bulk load in transactions:**

```sql
BEGIN;
  -- Insert many vectors...
COMMIT;
```

### Query Optimization

**Start with low ef_search:**

```sql
-- Fast, reasonable recall
SELECT rowid FROM items 
WHERE items MATCH query AND ef_search=10 
LIMIT 10;
```

**Increase if recall is insufficient:**

```sql
-- Better recall, slower
SELECT rowid FROM items 
WHERE items MATCH query AND ef_search=100 
LIMIT 10;
```

**Effective limit:**

Remember that effective limit ≤ ef_search × 2. For 100 results:

```sql
SELECT rowid FROM items 
WHERE items MATCH query AND ef_search=50 
LIMIT 100;
```

### SQLite Tuning

Optimize SQLite for vector workloads:

```sql
-- Use WAL mode for better concurrency
PRAGMA journal_mode=WAL;

-- Increase page size (before creating tables)
PRAGMA page_size=8192;

-- Increase cache size
PRAGMA cache_size=-2000000;  -- 2GB

-- Disable synchronous writes (if durability is not critical)
PRAGMA synchronous=NORMAL;
```

### Memory vs Accuracy Tradeoffs

| Parameter                | Memory Impact     | Accuracy Impact | Query Speed      |
| ------------------------ | ----------------- | --------------- | ---------------- |
| Higher `m`               | +50% per doubling | +2-5% recall    | -10-20%          |
| Higher `ef_construction` | Minimal           | +5-10% recall   | No change        |
| Higher `ef_search`       | Minimal           | +10-30% recall  | -2× per doubling |

### Benchmarking

Run the included benchmarks:

```bash
# Recall@k benchmark
lua test/recall_bench.lua

# SIFT dataset benchmark
lua bench/sift_bench.lua

# Parameter sweep
lua bench/param_sweep.lua
```

Results are saved to `bench/results/`.

## Monitoring

### Shadow Tables

sqlite-vector stores data in shadow tables with a `_` prefix:

| Shadow Table       | Purpose                                   |
| ------------------ | ----------------------------------------- |
| `tablename_config` | Configuration (dims, metric, HNSW params) |
| `tablename_data`   | Vector storage (rowid → vector BLOB)      |
| `tablename_graph`  | HNSW graph edges                          |
| `tablename_layers` | Layer assignments per node                |

### Inspect Configuration

```sql
SELECT * FROM items_config;
```

Output:

```
entry_point | dims | metric | m  | ef_construction | ef_search
----------- | ---- | ------ | -- | --------------- | ---------
1           | 128  | cosine | 16 | 200             | 10
```

### Count Verification

Check that data and index are in sync:

```sql
-- Logical count (via virtual table)
SELECT COUNT(*) FROM items;

-- Physical count (shadow table)
SELECT COUNT(*) FROM items_data;

-- Should match
```

### Graph Statistics

Analyze graph connectivity:

```sql
-- Average degree (connections per node)
SELECT AVG(cnt) FROM (
  SELECT node, COUNT(*) as cnt 
  FROM items_graph 
  GROUP BY node
);

-- Max degree
SELECT MAX(cnt) FROM (
  SELECT node, COUNT(*) as cnt 
  FROM items_graph 
  GROUP BY node
);
```

## Reference

### SQL Functions

#### Vector Construction

| Function         | Returns | Description                        |
| ---------------- | ------- | ---------------------------------- |
| `vec(text)`      | BLOB    | Parse JSON array to float32 vector |
| `vec_f32(text)`  | BLOB    | Create float32 vector              |
| `vec_int8(text)` | BLOB    | Create int8 vector                 |
| `vec_bit(blob)`  | BLOB    | Create bit vector                  |

#### Distance Functions

| Function                     | Returns | Description            |
| ---------------------------- | ------- | ---------------------- |
| `vec_distance_l2(a, b)`      | REAL    | Euclidean distance     |
| `vec_distance_cosine(a, b)`  | REAL    | Cosine distance        |
| `vec_distance_ip(a, b)`      | REAL    | Negative inner product |
| `vec_distance_l1(a, b)`      | REAL    | Manhattan distance     |
| `vec_distance_hamming(a, b)` | REAL    | Hamming distance       |
| `vec_distance_jaccard(a, b)` | REAL    | Jaccard distance       |

#### Vector Operations

| Function                   | Returns | Description                    |
| -------------------------- | ------- | ------------------------------ |
| `vec_add(a, b)`            | TEXT    | Element-wise addition          |
| `vec_sub(a, b)`            | TEXT    | Element-wise subtraction       |
| `vec_normalize(v)`         | TEXT    | L2 normalize to unit vector    |
| `vec_slice(v, start, end)` | TEXT    | Extract subvector [start, end) |

#### Utility Functions

| Function                 | Returns | Description                            |
| ------------------------ | ------- | -------------------------------------- |
| `vec_dims(v)`            | INTEGER | Vector dimensionality                  |
| `vec_norm(v)`            | REAL    | L2 norm                                |
| `vec_type(v)`            | TEXT    | Type: 'float32', 'int8', 'bit', 'text' |
| `vec_format(blob, dims)` | TEXT    | Format raw BLOB as JSON array          |

### Virtual Table Parameters

| Parameter         | Type    | Range     | Default      | Description                |
| ----------------- | ------- | --------- | ------------ | -------------------------- |
| `dims`            | INTEGER | 1-8192    | **required** | Vector dimensions          |
| `metric`          | TEXT    | see below | `cosine`     | Distance metric            |
| `m`               | INTEGER | 2-100     | `16`         | HNSW connections per layer |
| `ef_construction` | INTEGER | 1-1000    | `200`        | Build-time beam width      |
| `ef_search`       | INTEGER | 1-1000    | `10`         | Query-time beam width      |

### Metric Names

| Primary   | Aliases                | Formula             |
| --------- | ---------------------- | ------------------- |
| `l2`      | `euclidean`            | sqrt(Σ(aᵢ - bᵢ)²)   |
| `cosine`  | `cos`                  | 1 - (a·b)/(‖a‖‖b‖)  |
| `ip`      | `dot`, `inner_product` | -(a·b)              |
| `l1`      | `manhattan`, `taxicab` | Σ\|aᵢ - bᵢ\|        |
| `hamming` | -                      | Σ(aᵢ ≠ bᵢ)          |
| `jaccard` | -                      | 1 - \|a∩b\|/\|a∪b\| |

### Limits

| Limit           | Value                   |
| --------------- | ----------------------- |
| Max dimensions  | 8192                    |
| Max LIMIT       | 4096                    |
| Max vector size | 32,768 bytes (8192 × 4) |

## Troubleshooting

### Empty Result Set

**Problem:** MATCH returns no results when data exists.

**Solution:** Increase `ef_search`:

```sql
SELECT rowid FROM items 
WHERE items MATCH query AND ef_search=100 
LIMIT 10;
```

### Dimension Mismatch

**Problem:** `dimension mismatch` error on insert.

**Solution:** Ensure vectors match table dimensions:

```sql
CREATE VIRTUAL TABLE items USING vec0(dims=3);
INSERT INTO items(vector) VALUES(vec('[1, 2, 3]'));  -- ✓ OK
INSERT INTO items(vector) VALUES(vec('[1, 2, 3, 4]'));  -- ✗ Error
```

### Extension Loading Error

**Problem:** `error loading extension`

**Solution:** Check library path and permissions:

```sql
-- Use absolute path
.load /full/path/to/libsqlite_vector.so

-- Or set LD_LIBRARY_PATH (Linux/macOS)
-- export LD_LIBRARY_PATH=/path/to/extension:$LD_LIBRARY_PATH
```

On macOS, ensure the extension isn't quarantined:

```bash
xattr -d com.apple.quarantine libsqlite_vector.dylib
```

### Build Errors

**Linux:** Missing math library:

```bash
# Install build essentials
sudo apt-get install build-essential

# Ensure -lm is linked (xmake.lua handles this)
```

**macOS:** Compiler not found:

```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**Windows:** MinGW required:

```bash
# Install mingw-w64
# Set platform in xmake
xmake f -p mingw
xmake
```

### Poor Recall

**Problem:** KNN returns irrelevant results.

**Solutions:**

1. Increase `ef_construction` when building:
   ```sql
   CREATE VIRTUAL TABLE items USING vec0(dims=768, ef_construction=400);
   ```

2. Increase `ef_search` at query time:
   ```sql
   SELECT rowid FROM items 
   WHERE items MATCH query AND ef_search=200 
   LIMIT 10;
   ```

3. Increase `m` for better graph connectivity:
   ```sql
   CREATE VIRTUAL TABLE items USING vec0(dims=768, m=32);
   ```

### Slow Queries

**Problem:** Queries take too long.

**Solutions:**

1. Lower `ef_search`:
   ```sql
   SELECT rowid FROM items 
   WHERE items MATCH query AND ef_search=10 
   LIMIT 10;
   ```

2. Enable WAL mode:
   ```sql
   PRAGMA journal_mode=WAL;
   ```

3. Increase cache size:
   ```sql
   PRAGMA cache_size=-2000000;  -- 2GB
   ```

### Memory Usage

**Problem:** High memory consumption.

**Solutions:**

1. Lower `m` parameter (fewer connections)
2. Use smaller dimensions
3. Use int8 or bit vectors instead of float32
4. Adjust SQLite cache size: `PRAGMA cache_size=N`

## Comparison to pgvector

sqlite-vector and pgvector share similar goals but target different database
systems.

### Feature Comparison

| Feature                    | sqlite-vector        | pgvector        |
| -------------------------- | -------------------- | --------------- |
| **Database**               | SQLite               | PostgreSQL      |
| **Architecture**           | Embedded, in-process | Client-server   |
| **HNSW index**             | ✓                    | ✓               |
| **IVFFlat index**          | ✗                    | ✓               |
| **Exact search**           | ✓                    | ✓               |
| **L2, cosine, IP, L1**     | ✓                    | ✓               |
| **Hamming, Jaccard**       | ✓                    | ✓               |
| **Sparse vectors**         | ✗                    | ✓               |
| **Half precision**         | ✗                    | ✓ (native type) |
| **Transactions**           | ✓ (ACID)             | ✓ (ACID)        |
| **Graph repair on delete** | ✓                    | Lazy vacuum     |
| **WHERE filtering**        | Limited              | ✓               |
| **Expression indexes**     | ✗                    | ✓               |
| **Parallel queries**       | ✗                    | ✓               |

### API Differences

**Table Creation:**

```sql
-- pgvector
CREATE TABLE items (embedding vector(3));
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);

-- sqlite-vector
CREATE VIRTUAL TABLE items USING vec0(dims=3, metric=l2);
```

**KNN Search:**

```sql
-- pgvector
SELECT * FROM items ORDER BY embedding <-> '[1,2,3]' LIMIT 5;

-- sqlite-vector
SELECT rowid FROM items WHERE items MATCH '[1,2,3]' LIMIT 5;
```

**Distance Functions:**

```sql
-- pgvector (operators)
embedding <-> '[1,2,3]'   -- L2
embedding <#> '[1,2,3]'   -- inner product
embedding <=> '[1,2,3]'   -- cosine

-- sqlite-vector (functions)
vec_distance_l2(items, '[1,2,3]')
vec_distance_ip(items, '[1,2,3]')
vec_distance_cosine(items, '[1,2,3]')
```

**Runtime Parameters:**

```sql
-- pgvector (session variables)
SET hnsw.ef_search = 100;
SELECT * FROM items ORDER BY embedding <-> '[1,2,3]' LIMIT 5;

-- sqlite-vector (query predicates)
SELECT rowid FROM items 
WHERE items MATCH '[1,2,3]' AND ef_search=100 
LIMIT 5;
```

### When to Choose Each

**Choose sqlite-vector when:**

- Building mobile or embedded applications
- Need single-file database portability
- Want simple deployment (no server)
- Prototyping or development
- Working with moderate dataset sizes (<1M vectors)
- Using SQLite ecosystem

**Choose pgvector when:**

- Running production server applications
- Need advanced PostgreSQL features
- Require concurrent write scalability
- Working with very large datasets (>10M vectors)
- Need WHERE clause filtering with vector search
- Want parallel query execution

Both are excellent choices—pick based on your database system and deployment
needs.

## Implementation Notes

### Architecture

- **Virtual table module:** Implements SQLite's `sqlite3_module` interface
- **Shadow tables:** Config, data, graph, and layers stored as regular SQLite
  tables
- **No in-memory index:** Entire HNSW graph persists in SQLite (SQL-native)
- **Prepared statement cache:** Reused statements for graph operations
- **Transaction hooks:** Config updates deferred until commit

### SimSIMD Integration

Distance computations are hardware-accelerated using
[SimSIMD](https://github.com/ashvardanian/SimSIMD):

- **Runtime CPU dispatch:** Selects optimal SIMD kernels (AVX-512, AVX2, NEON,
  SVE)
- **Cross-platform:** x86-64, ARM64, RISC-V
- **Fallback:** Portable C implementation
- **Zero overhead:** No runtime penalty for SIMD detection

### HNSW Details

Implementation follows the original HNSW paper with modifications:

- **SQL-native storage:** No in-memory index, all data in SQLite
- **Layered graph:** Exponentially decreasing density per layer
- **Bidirectional links:** Edges maintained in both directions
- **Graph repair:** Deletes trigger re-wiring of disconnected neighbors
- **Entry point election:** New entry point selected if deleted
- **Empty table support:** entry_point = -1 for empty tables

### Testing

Comprehensive test suite in `test/`:

- **Basic tests:** Smoke tests for core functionality
- **KNN tests:** Search correctness, ordering, updates, deletes
- **Distance tests:** Verify all metrics against known results
- **Vector ops tests:** Add, subtract, normalize, slice
- **Edge cases:** Parser edge cases, dimension mismatches
- **Persistence:** Database close/reopen behavior
- **Benchmarks:** Recall@k, latency, parameter sweeps

Run all tests:

```bash
cd test
./run_all.sh
```

## Contributing

Contributions are welcome! Please:

1. **Run tests** before submitting:
   ```bash
   cd test && ./run_all.sh
   ```

2. **Add tests** for new features

3. **Follow existing code style:**
   - C11 standard
   - 4-space indentation
   - Descriptive variable names

4. **Update documentation** for user-facing changes

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/sqlite-vector.git
cd sqlite-vector

# Build debug version
xmake f -m debug
xmake

# Run tests
cd test
./run_all.sh

# Run benchmarks
cd ../bench
lua recall_bench.lua
```

### Project Structure

```
src/
  extension.c    - Entry point, function registration
  vtab.c         - Virtual table implementation
  hnsw.c         - HNSW algorithm
  distance.c     - Distance functions, SimSIMD integration
  vec_ops.c      - Vector operations
  vec_parse.c    - Text parsing and formatting
test/
  *.sql          - SQL test cases
  *.lua          - Lua test runners
bench/
  recall_bench.lua    - Recall@k benchmarks
  sift_bench.lua      - Large-scale benchmarks
  param_sweep.lua     - Parameter exploration
```

## Acknowledgments

- **SimSIMD** by [@ashvardanian](https://github.com/ashvardanian) for
  hardware-accelerated distance computations
- **HNSW algorithm** by Malkov & Yashunin
  ([paper](https://arxiv.org/abs/1603.09320))
- **SQLite** by D. Richard Hipp and contributors
- **pgvector** by [@ankane](https://github.com/ankane) for inspiration

## License

[MIT License](LICENSE) - see LICENSE file for details

## History

- **v0.1.0** (2026-03-05) - Initial release
  - HNSW indexing
  - Multiple distance metrics
  - Vector operations
  - SimSIMD integration
  - Full test coverage

---

Questions? Issues?
[Open an issue](https://github.com/yourusername/sqlite-vector/issues) or start a
[discussion](https://github.com/yourusername/sqlite-vector/discussions).
