## Plan: SQLite Vector Extension (`sqlite-vector`)

A loadable SQLite extension implementing pgvector-style nearest-neighbor search via HNSW, built with xmake, using usearch for the index backend. Vectors are exposed through a virtual table module named `vec0` (matching the sqlite-vec convention), with a `MATCH`-style SQL interface.

**SQL interface target:**
```sql
CREATE VIRTUAL TABLE items USING vec0(dims=1536, metric=cosine);
INSERT INTO items VALUES (vec('[0.1, 0.2, ...]'));
SELECT rowid, distance FROM items WHERE items MATCH vec('[0.1, ...]') LIMIT 10;
```

---

**Steps**

**1. Repository & build scaffold**

Create the repo layout:
```
sqlite-vector/
  xmake.lua               -- build definition
  include/
    sqlite3ext.h          -- bundled SQLite extension header
  third_party/
    usearch/              -- git submodule: unum-cloud/usearch
  src/
    extension.cpp         -- entry point + module registration
    vtab.cpp/.h           -- sqlite3_module implementation
    vec_parse.cpp/.h      -- text '[...]' → float* parsing + formatting
    distance.cpp/.h       -- scalar distance functions (wraps usearch metrics)
    hnsw.cpp/.h           -- usearch index lifecycle wrapper
  test/
    basic.sql
```

xmake.lua: `set_kind("shared")`, `set_languages("cxx17")`, `add_files("src/*.cpp")`, include paths for `third_party/usearch/include` and `third_party/usearch/c`, macOS `-undefined dynamic_lookup` ldflags, Linux `-fPIC`. Output filename `sqlite_vector` (so SQLite looks for `sqlite3_sqlite_vector_init`).

**2. Vector parsing & type layer** — src/vec_parse.cpp

Since SQLite has no custom column types, vectors are stored as `TEXT` (pgvector's `[1.0,2.0,3.0]` format) in user-facing schema columns but kept as raw `float32` BLOB in shadow storage. Implement:
- `vec_parse(const char*, float**, int* dims)` — tokenize, validate, alloc
- `vec_format(const float*, int dims, char* out)` — float[] → `[x,y,z]` text
- `vec('[1,2,3]')` scalar SQL function → returns its argument validated (identity, for symmetry with pgvector)
- `vec_dims(text|blob)` scalar function
- `vec_norm(text)` scalar function

**3. Shadow table schema** — created in `xCreate`, dropped in `xDestroy`

For a virtual table named `items`:
- `items_config` — `(key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID` — stores `dims`, `metric`, `version`
- `items_data` — `(id INTEGER PRIMARY KEY, vector BLOB)` — raw float32 vectors per rowid
- `items_index` — `(id INTEGER PRIMARY KEY, data BLOB)` — single row (id=1): serialized usearch HNSW graph BLOB

`xShadowName` returns 1 for `"config"`, `"data"`, `"index"`.

`sqlite3_declare_vtab` schema:
```sql
CREATE TABLE x(vector TEXT, distance REAL HIDDEN, k INTEGER HIDDEN)
```
`vector` = the user-facing column; `distance` and `k` are hidden output/input columns.

**4. Virtual table module (`sqlite3_module`)** — src/vtab.cpp

Implement all required callbacks with `iVersion = 3`:

| Callback | Responsibility |
|---|---|
| `xCreate` | Create 3 shadow tables, write config, declare vtab schema, init empty usearch index, serialize to `_index` |
| `xConnect` | Load config from `_config`, deserialize HNSW graph from `_index` BLOB into RAM |
| `xDisconnect` | Free in-memory usearch index |
| `xDestroy` | Drop all 3 shadow tables, free index |
| `xBestIndex` | Detect `SQLITE_INDEX_CONSTRAINT_FUNCTION` from `MATCH`/`vec_distance_*` + `SQLITE_INDEX_CONSTRAINT_LIMIT`; set `idxNum` and `estimatedCost` |
| `xOpen` / `xClose` | Alloc/free cursor (holds result rowids + distances array) |
| `xFilter` | Parse query vector from `argv[0]`, run `usearch_search`, store results in cursor |
| `xNext` / `xEof` | Iterate cursor result array |
| `xColumn` | Return `vector` text (fetch from `_data`) or `distance` float |
| `xRowid` | Return current result rowid |
| `xUpdate` | INSERT: parse+validate dims, write to `_data`, call `usearch_add`; DELETE: write to `_data`, call `usearch_remove`; UPDATE: delete+reinsert |
| `xCommit` | Serialize usearch index → `_index` BLOB |
| `xRollback` | Reload usearch index from `_index` BLOB (discard RAM changes) |
| `xFindFunction` | Overload `<->`, `<=>`, `<#>`, `<+>` as `SQLITE_INDEX_CONSTRAINT_FUNCTION` |
| `xShadowName` | Guard shadow tables from direct writes in defensive mode |

**5. HNSW wrapper** — src/hnsw.cpp

Thin C++ class wrapping the usearch C API:
- `HnswIndex(dims, metric, m=16, ef_construction=128)` constructor → `usearch_init`
- `add(int64_t rowid, const float* vec)` → `usearch_add`
- `remove(int64_t rowid)` → `usearch_remove`
- `search(const float* query, size_t k, int64_t* keys_out, float* dists_out)` → `usearch_search`
- `serialize(void** buf, size_t* len)` → `usearch_save_buffer`
- `deserialize(const void* buf, size_t len)` → `usearch_load_buffer`
- All usearch errors caught and surfaced via return code + message

Metric mapping: `cosine` → `usearch_metric_cos_k`, `l2` → `usearch_metric_l2sq_k`, `inner_product` → `usearch_metric_ip_k`, `l1` → `usearch_metric_l1_k`, `hamming` → `usearch_metric_hamming_k`, `jaccard` → `usearch_metric_jaccard_k`.

**6. Distance scalar functions** — src/distance.cpp

Register as `sqlite3_create_function_v2` with `SQLITE_DETERMINISTIC`:
- `vec_distance_l2(a, b)` — Euclidean L2
- `vec_distance_cosine(a, b)` — cosine distance (1 − cosine similarity)
- `vec_distance_ip(a, b)` — inner product (dot; note: returned *negated* for ORDER BY compatibility, matching pgvector `<#>`)
- `vec_distance_l1(a, b)` — Manhattan
- `vec_distance_hamming(a, b)` — for bit vectors (packed as BLOB)
- `vec_distance_jaccard(a, b)` — for bit vectors

Also register the operator-style aliases `<->`, `<=>`, `<#>`, `<+>` via `xFindFunction`.

**7. Extension entry point** — src/extension.cpp

```c
int sqlite3_sqlitevector_init(sqlite3*, char**, const sqlite3_api_routines*)
```
- `SQLITE_EXTENSION_INIT2`
- `sqlite3_create_module_v2(db, "vec0", &vectorModule, ...)`
- Register all scalar functions from step 6
- Register `vec()`, `vec_dims()`, `vec_norm()` from step 2

**8. MATCH operator wiring**

`MATCH` in `WHERE items MATCH vec('[...]')` is syntactic sugar for the `match(items, '[...]')` function. In `xFindFunction`, intercept `"match"` with `nArg=2` and map it to the kNN search path (`idxNum=1`). Also intercept `"<->"`, `"<=>"`, `"<#>"`, `"<+>"` for explicit distance-operator queries. Both call the same underlying `usearch_search` code path in `xFilter`.

**9. `CREATE VIRTUAL TABLE` options**

Parse `argv` in `xCreate`/`xConnect` for:
- `dims=N` (required)
- `metric=cosine|l2|ip|l1|hamming|jaccard` (default: `l2`)
- `m=16` (HNSW connectivity)
- `ef_construction=128`

Write all to `_config` shadow table.

---

**Verification**

1. Build: `xmake -v` → produces `build/libsqlite_vector.dylib` (macOS) or `.so` (Linux)
2. Load & smoke test:
   ```sql
   .load ./build/libsqlite_vector
   CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=cosine);
   INSERT INTO vecs VALUES (vec('[1.0, 0.0, 0.0]'));
   INSERT INTO vecs VALUES (vec('[0.0, 1.0, 0.0]'));
   SELECT rowid, distance FROM vecs WHERE vecs MATCH vec('[1.0, 0.1, 0.0]') LIMIT 2;
   ```
3. Persistence test: close and reopen DB, repeat query — index reloads from `_index` BLOB.
4. Rollback test: `BEGIN; INSERT ...; ROLLBACK;` → vector absent from results.
5. Scalar functions: `SELECT vec_distance_l2('[1,0]', '[0,1]');` → `1.4142...`
6. All metrics: create 6 virtual tables with each metric, insert, query.

---

**Decisions**
- **usearch over hnswlib**: usearch provides a stable C99 API, no C++ exception leakage, and has a reference SQLite integration.
- **MATCH interface only** (not hidden-column or function-constraint styles) for the initial cut — cleaner SQL, matches FTS5 muscle memory.
- **Text encoding `[1.0,2.0,3.0]`** for user-facing values; stored as raw BLOB in `_data` shadow table for efficiency.
- **Serialize index on every `xCommit`**: correct-by-default, can be optimized later with dirty-flag or WAL integration.
- **`vec0` module name**: matches the sqlite-vec naming convention, avoiding conflicts with any future official SQLite vector module.