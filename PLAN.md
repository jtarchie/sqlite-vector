## Plan: SQLite Vector Extension (`sqlite-vector`)

A loadable SQLite extension implementing pgvector-style nearest-neighbor search
via HNSW, built with xmake, with **no external dependencies**. The HNSW graph is
stored entirely in SQLite shadow tables — the index _is_ the database. Vectors
are exposed through a virtual table module named `vec0`, with a `MATCH`-style
SQL interface.

**Key architectural principle:** No external index library is loaded into RAM.
The HNSW graph adjacency list is stored in a shadow table backed by SQLite's
B-tree. A query traversal touches O(ef_search × M) rows — a few hundred rows for
typical settings — which SQLite's page cache handles efficiently. Memory usage
at query time is O(ef_search), not O(N). Durability and rollback are free via
SQLite's WAL.

**SQL interface target:**

```sql
CREATE VIRTUAL TABLE items USING vec0(dims=1536, metric=cosine);
INSERT INTO items VALUES (vec('[0.1, 0.2, ...]'));
SELECT rowid, distance FROM items WHERE items MATCH vec('[0.1, ...]') LIMIT 10;
```

---

## Built so far

- **Build**: `xmake.lua` — shared lib; macOS `-undefined dynamic_lookup`;
  SimSIMD include path. Bundled `include/sqlite3{,ext}.h` (SQLite 3.51.2).
  SimSIMD as `third_party/simsimd` git submodule (header-only, Apache-2.0).
- **`src/extension.c`** — `sqlite3_sqlitevector_init` entry point; registers
  `vec0` module, `vec_parse` scalars, and distance scalars.
- **`src/vec_parse.{h,c}`** — `vec_parse()` / `vec_format()`; SQL scalars
  `vec()`, `vec_dims()`, `vec_norm()`.
- **`src/distance.{h,c}`** — Six SimSIMD-backed kernels (`l2`, `cosine`, `ip`,
  `l1`, `hamming`, `jaccard`); `distance_for_metric()` resolver; SQL scalars
  `vec_distance_*`.
- **`src/hnsw.{h,c}`** — Full SQL-native HNSW: `hnsw_search`, `hnsw_insert`,
  `hnsw_delete`. All graph state in shadow tables; C heap holds only candidate
  heaps (O(ef) nodes). Binary min/max-heaps + sorted visited set. Random-level
  sampling, bidirectional edge pruning, entry-point election on delete (v1: no
  graph repair).
- **`src/vtab.{h,c}`** — Complete `sqlite3_module` (iVersion=3):
  `xCreate`/`xConnect`/`xDisconnect`/`xDestroy` (4 shadow tables + config
  persistence); `xBestIndex` (MATCH → kNN, rowid-eq → point lookup, full-scan
  fallback); `xFilter` calling `hnsw_search`; `xColumn` fetching vector BLOBs as
  `[x,y,z]` text; `xUpdate` for INSERT (→ `hnsw_insert`) and DELETE (→
  `hnsw_delete`); `xShadowName`.
- **Tests** (all passing via `test/run_all.sh`): `basic.sql`, `vec_parse.sql`,
  `distance.sql`, `shadow.sql`, `insert.sql`, `ffi_test.lua`, `shadow_connect`
  (xConnect persistence via `test/wrappers/run_shadow_connect.sh`).

**Not yet built:** `xFindFunction` (operator aliases `<->`, `<=>`, `<#>`,
`<+>`); UPDATE (delete + reinsert); full-scan `xFilter` (idxNum=0 returns
empty); accuracy/recall benchmarks.

---

## Steps

### 1. Repository & build scaffold

```
sqlite-vector/
  xmake.lua               -- build definition
  include/
    sqlite3ext.h          -- bundled SQLite extension header
  third_party/
    simsimd/              -- git submodule: ashvardanian/SimSIMD (header-only)
  src/
    extension.c           -- entry point + module registration
    vtab.c / vtab.h       -- sqlite3_module implementation
    vec_parse.c / vec_parse.h  -- text '[...]' ↔ float* parsing
    distance.c / distance.h    -- distance functions via SimSIMD
    hnsw.c / hnsw.h            -- HNSW traversal via shadow table SQL
  test/
    basic.sql
```

[xmake.lua](xmake.lua): `set_kind("shared")`, `set_languages("c11")`,
`add_files("src/*.c")`, `add_includedirs("third_party/simsimd/include")`, macOS
`-undefined dynamic_lookup` ldflags, Linux `-fPIC`. Output filename
`sqlite_vector` (entry point: `sqlite3_sqlite_vector_init`).

### 2. Vector parsing & type layer — [src/vec_parse.c](src/vec_parse.c)

Since SQLite has no custom column types, vectors are passed as `TEXT` in
pgvector's `[1.0,2.0,3.0]` format and stored as raw `float32` BLOBs in shadow
tables. Implement:

- `vec_parse(const char*, float**, int* dims)` — tokenize, validate, alloc
- `vec_format(const float*, int dims, char* out)` — float[] → `[x,y,z]` text
- `vec('[1,2,3]')` scalar SQL function — validates and round-trips the text
  (identity, for pgvector symmetry)
- `vec_dims(text|blob)` scalar SQL function — returns dimension count
- `vec_norm(text)` scalar SQL function — returns L2 norm

### 3. Shadow table schema — created in `xCreate`, dropped in `xDestroy`

For a virtual table named `items`, four shadow tables are created:

```sql
-- Persisted configuration
CREATE TABLE items_config (
    key   TEXT PRIMARY KEY,
    value TEXT
) WITHOUT ROWID;
-- keys: dims, metric, m, ef_construction, entry_point, max_layer, count

-- Raw vector storage; id == user-visible rowid
CREATE TABLE items_data (
    id     INTEGER PRIMARY KEY,
    vector BLOB NOT NULL   -- raw float32, len = dims * 4 bytes
);

-- HNSW graph: directed adjacency list
-- SQLite's B-tree makes (layer, node_id) lookups O(log N)
CREATE TABLE items_graph (
    layer       INTEGER NOT NULL,
    node_id     INTEGER NOT NULL,
    neighbor_id INTEGER NOT NULL,
    distance    REAL    NOT NULL,
    PRIMARY KEY (layer, node_id, neighbor_id)
) WITHOUT ROWID;

-- Per-node maximum layer (for entry-point and level assignment)
CREATE TABLE items_layers (
    node_id   INTEGER PRIMARY KEY,
    max_layer INTEGER NOT NULL
) WITHOUT ROWID;
```

`xShadowName` returns 1 for `"config"`, `"data"`, `"graph"`, `"layers"`.

`sqlite3_declare_vtab` schema:

```sql
CREATE TABLE x(vector TEXT, distance REAL HIDDEN)
```

`vector` — user-facing column; `distance` — hidden output populated during kNN
scan.

### 4. Virtual table module (`sqlite3_module`) — [src/vtab.c](src/vtab.c)

Implement all required callbacks with `iVersion = 3`:

| Callback           | Responsibility                                                                                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `xCreate`          | Create 4 shadow tables, write config defaults, call `sqlite3_declare_vtab`                                                                                           |
| `xConnect`         | Read config from `_config` into vtab struct; no data loaded into RAM                                                                                                 |
| `xDisconnect`      | Free vtab struct (no in-memory index to release)                                                                                                                     |
| `xDestroy`         | `DROP TABLE` all 4 shadow tables, free vtab struct                                                                                                                   |
| `xBestIndex`       | Detect `SQLITE_INDEX_CONSTRAINT_FUNCTION` for `MATCH` + `SQLITE_INDEX_CONSTRAINT_LIMIT`; `idxNum=1`, low `estimatedCost`; full-scan fallback `idxNum=0`              |
| `xOpen` / `xClose` | Alloc/free cursor; cursor holds a heap array of `(rowid, distance)` results populated by `xFilter`                                                                   |
| `xFilter`          | Parse query vector from `argv[0]`, read `LIMIT` from `argv[1]`, call HNSW search (step 5), store result array in cursor                                              |
| `xNext` / `xEof`   | Advance/check cursor position in result array                                                                                                                        |
| `xColumn`          | col 0 (`vector`): fetch BLOB from `_data`, format as text; col 1 (`distance`): return stored distance                                                                |
| `xRowid`           | Return current result rowid                                                                                                                                          |
| `xUpdate`          | INSERT: validate dims, write to `_data`, call HNSW insert; DELETE: remove from `_data`, `_graph`, `_layers`, update entry point if needed; UPDATE: delete + reinsert |
| `xFindFunction`    | Intercept `"match"` (nArg=2) → kNN path; intercept `"<->"`, `"<=>"`, `"<#>"`, `"<+>"` as distance constraints returning `SQLITE_INDEX_CONSTRAINT_FUNCTION` (150)     |
| `xShadowName`      | Return 1 for `"config"`, `"data"`, `"graph"`, `"layers"`                                                                                                             |

No `xCommit` / `xRollback` needed — SQLite's own transaction machinery covers
the shadow tables automatically.

### 5. SQL-native HNSW traversal — [src/hnsw.c](src/hnsw.c)

The HNSW algorithm is implemented in C, issuing `sqlite3_prepare`/`sqlite3_step`
queries against the shadow tables. No index data ever lives outside SQLite's
page cache.

**Prepared statements cached on the vtab struct:**

```sql
-- Fetch neighbors for a node at a given layer
SELECT neighbor_id, distance FROM {name}_graph
WHERE layer = ? AND node_id = ?;

-- Insert/replace a directed edge
INSERT OR REPLACE INTO {name}_graph(layer, node_id, neighbor_id, distance)
VALUES (?, ?, ?, ?);

-- Delete all edges for a node
DELETE FROM {name}_graph WHERE node_id = ? OR neighbor_id = ?;

-- Fetch vector BLOB by rowid
SELECT vector FROM {name}_data WHERE id = ?;

-- Read / write a config value
SELECT value FROM {name}_config WHERE key = ?;
UPDATE {name}_config SET value = ? WHERE key = ?;
```

**`hnsw_search(vtab, query_vec, k, ef_search)` → result array:**

1. Read `entry_point` and `max_layer` from config.
2. Greedy descend layers `max_layer` → 1: fetch neighbors via `_graph`, compute
   distance, track current best.
3. At layer 0: maintain a max-heap of size `ef_search` candidates; expand each
   candidate's neighbors, prune heap. Return top-k sorted by distance.
4. All heap structures live in C heap; only neighbor-row fetches touch SQLite
   pages.

**`hnsw_insert(vtab, rowid, vec)` → void:**

1. Sample random level `l = floor(-ln(uniform()) * 1/ln(M))`.
2. Write to `_data` and `_layers`.
3. Descend from `max_layer` → `l+1` to find entry to lower layers.
4. At each layer ≤ l: find `ef_construction` nearest neighbors, select best `M`
   (or `M0 = 2M` at layer 0), write edges bidirectionally to `_graph` with
   heuristic neighbor pruning.
5. Update `entry_point` / `max_layer` in `_config` if new node's level exceeds
   current.

**`hnsw_delete(vtab, rowid)` → void:**

1. Remove rows from `_data` and `_layers`.
2. Delete all `_graph` edges where `node_id = rowid OR neighbor_id = rowid`.
3. If deleted node was the entry point, scan `_layers` for a new
   `MAX(max_layer)` candidate and reassign.
4. v1: no graph repair — deleted nodes leave edges sparser but traversal skips
   missing nodes gracefully.

### 6. Distance functions — [src/distance.c](src/distance.c)

Distance kernels are provided by
[SimSIMD](https://github.com/ashvardanian/SimSIMD) (Apache-2.0, header-only,
`third_party/simsimd/include/simsimd/simsimd.h`). SimSIMD dispatches to AVX-512,
AVX2, ARM Neon, or SVE at runtime, falling back to scalar. This is significant
at high dimensions (1536+) where `dist_fn` is called O(ef_search × M × layers) ≈
1000 times per query.

```c
#include "simsimd/simsimd.h"

simsimd_distance_t dist;
simsimd_cos_f32(a, b, dims, &dist);    // cosine
simsimd_l2sq_f32(a, b, dims, &dist);   // L2 squared
simsimd_dot_f32(a, b, dims, &dist);    // inner product
simsimd_l1_f32(a, b, dims, &dist);     // L1 / Manhattan
simsimd_hamming_b8(a, b, bytes, &dist); // hamming (bit vectors)
simsimd_jaccard_b8(a, b, bytes, &dist); // jaccard (bit vectors)
```

Note: `dist_ip` is stored negated (`-dot`) so ORDER BY ASC returns nearest
first, matching pgvector's `<#>` convention.

A `dist_fn` function pointer on the vtab struct selects the active metric at
query time.

Register as SQL scalar functions via `sqlite3_create_function_v2` with
`SQLITE_DETERMINISTIC`: `vec_distance_l2`, `vec_distance_cosine`,
`vec_distance_ip`, `vec_distance_l1`, `vec_distance_hamming`,
`vec_distance_jaccard`.

### 7. Extension entry point — [src/extension.c](src/extension.c)

```c
int sqlite3_sqlitevector_init(sqlite3 *db, char **pzErrMsg,
                               const sqlite3_api_routines *pApi)
{
    SQLITE_EXTENSION_INIT2(pApi);
    sqlite3_create_module_v2(db, "vec0", &vectorModule, NULL, NULL);
    // register scalar distance functions (step 6)
    // register vec(), vec_dims(), vec_norm() (step 2)
    return SQLITE_OK;
}
```

### 8. MATCH operator wiring

`WHERE items MATCH vec('[...]')` desugars to `match(items, '[...]')`. In
`xFindFunction`, intercept:

- `"match"` with `nArg=2` → kNN path (`idxNum=1`), return 150
  (`SQLITE_INDEX_CONSTRAINT_FUNCTION`)
- `"<->"`, `"<=>"`, `"<#>"`, `"<+>"` → same kNN path with per-call metric
  override

Both paths arrive in `xFilter` as `argv[0]` = query vector text, `argv[1]` =
limit. `xFilter` calls `hnsw_search`, populates cursor result array, and
`xNext`/`xColumn` iterate it.

### 9. `CREATE VIRTUAL TABLE` options

Parsed from `argv` in `xCreate` / `xConnect`, written to `_config`:

| Option                | Default    | Description                                      |
| --------------------- | ---------- | ------------------------------------------------ |
| `dims=N`              | (required) | Vector dimensionality                            |
| `metric=...`          | `l2`       | `cosine`, `l2`, `ip`, `l1`, `hamming`, `jaccard` |
| `m=16`                | 16         | HNSW max neighbors per node per layer            |
| `ef_construction=128` | 128        | Candidate set size during index build            |
| `ef_search=64`        | 64         | Candidate set size during query                  |

---

## Testing policy

Every feature is implemented alongside its test. **Tests must pass before
committing.**

- Test file for each source file lives in `test/` with a matching name (e.g.
  `src/vec_parse.c` → `test/vec_parse.sql` or `test/vec_parse_test.c`).
- SQL-level behaviour (insert, query, error cases) is tested with `.sql` scripts
  run via `sqlite3 :memory: < test/foo.sql`. Use `.bail on` at the top of every
  SQL test so any error aborts.
- Unit-level behaviour (distance math, parse edge cases) may use small LuaJIT
  with FFI to test.
- Run the full test suite after every change:
  `sqlite3 :memory: < test/basic.sql` (and any new test files).
- `test/basic.sql` is the integration regression — it grows with each commit and
  must always pass.

---

## Verification

1. **Build**: `xmake -v` → produces `build/libsqlite_vector.dylib` (macOS) or
   `.so` (Linux).
2. **Smoke test**:
   ```sql
   .load ./build/libsqlite_vector
   CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=cosine);
   INSERT INTO vecs VALUES (vec('[1.0, 0.0, 0.0]'));
   INSERT INTO vecs VALUES (vec('[0.0, 1.0, 0.0]'));
   SELECT rowid, distance FROM vecs WHERE vecs MATCH vec('[1.0, 0.1, 0.0]') LIMIT 2;
   ```
3. **Persistence**: close and reopen the DB, repeat the query — results are
   identical; no reload step needed since the graph lives in shadow tables.
4. **Rollback**: `BEGIN; INSERT ...; ROLLBACK;` → vector absent from results and
   from `vecs_data`.
5. **Scalar functions**: `SELECT vec_distance_l2('[1,0]', '[0,1]');` →
   `1.4142...`
6. **All metrics**: create 6 virtual tables (one per metric), insert vectors,
   query each.
7. **Shadow table inspection**: `SELECT * FROM vecs_graph LIMIT 20;` — readable
   adjacency list.

---

## Decisions

- **SQL-native HNSW graph**: stored in shadow tables; SQLite's page cache
  handles partial loading. Memory at query time is O(ef_search), not O(N).
  Rollback and durability are free via SQLite WAL — no `xCommit`/`xRollback`
  needed.
- **SimSIMD for distance kernels**: header-only submodule (Apache-2.0); provides
  SIMD-accelerated L2, cosine, IP, L1, Hamming, Jaccard with runtime CPU
  dispatch. The HNSW graph itself remains pure C + SQLite.
- **MATCH interface only**: Cleaner SQL, matches FTS5 muscle memory. Operator
  aliases (`<->` etc.) wired through `xFindFunction`.
- **Text encoding `[1.0,2.0,3.0]`** for user-facing values; stored as raw
  `float32` BLOB in `_data` for efficiency.
- **`vec0` module name**: matches the sqlite-vec naming convention, avoiding
  conflicts with any future official SQLite vector module.
