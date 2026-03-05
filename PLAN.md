## Plan: Expand Testing, Benchmarking & Features

The project currently has solid correctness tests for its core HNSW + kNN path,
but is missing edge-case error handling, error message assertions, fuzzing,
distance-threshold filtering, element-wise vector ops, metadata columns, typed
vector types, and any industry-scale benchmarking. We'll work across all three
tracks in parallel, keeping the existing Lua + SQL tooling.

---

### Track 3 — New Features

Each feature below needs: C implementation, function registration in
extension.c, and a corresponding SQL test file.

**3.1 Element-wise vector ops: `vec_add`, `vec_sub`, `vec_normalize` ✅ DONE**

Implemented in `src/vec_ops.c`:
- `vec_add(a TEXT, b TEXT) → TEXT`: element-wise sum, requires same dims
- `vec_sub(a TEXT, b TEXT) → TEXT`: element-wise difference
- `vec_normalize(v TEXT) → TEXT`: divide by L2 norm to create unit vector

Tested in `test/vec_ops.sql` with known-answer tests.

**3.2 `vec_slice(v, start, end) → TEXT` ✅ DONE**

Extracts a contiguous subvector (0-indexed, exclusive end). Useful for
Matryoshka embeddings. Implemented in `src/vec_ops.c`. Tested in `test/vec_ops.sql`.

**3.3 Distance threshold constraints (`AND distance < X`) ✅ DONE**

Modified `xBestIndex` in `src/vtab.c` to detect `SQLITE_INDEX_CONSTRAINT_LT/LE/GT/GE`
on the `distance` column when a MATCH constraint is also present. Pass
the threshold value(s) via `aConstraintUsage`. In `vec0_filter_knn`,
post-filter the HNSW result set against thresholds before returning rows.

Tested in `test/distance_constraints.sql` with range queries.

**3.4 Metadata columns in `vec0` (TODO)**

This is the largest outstanding feature. The declaration syntax becomes:

```sql
CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, name TEXT, score REAL);
```

Changes required:

- Parse extra `name TYPE` pairs in `xCreate`/`xConnect` in vtab.c
- Create a `{name}_meta` shadow table with `(id INTEGER PRIMARY KEY, ...)` for
  each metadata column
- Extend `xColumn` to read from the meta table for non-hidden, non-vector
  columns
- Extend `xUpdate` (INSERT/UPDATE) to write metadata alongside the vector
- Extend `xBestIndex` to recognize equality/range constraints on metadata
  columns and pass them as post-filters on the kNN result set (or as a
  pre-filter in the full-scan path)
- New `test/metadata.sql`

**3.5 Typed vector types ✅ DONE**

BLOB subtype tagging with constructors (mirroring sqlite-vec's subtype 223/224/225 approach):

- `vec_f32(text|blob) → BLOB` with subtype 223: raw float32 little-endian
- `vec_int8(text|blob) → BLOB` with subtype 225: raw int8 (clamped to [-128, 127])
- `vec_bit(blob) → BLOB` with subtype 224: validates bit-packed BLOB
- `vec_type(v) → TEXT`: returns `'float32'`, `'int8'`, `'bit'`, or `'text'`

Implemented in `src/vec_ops.c`. Tested in `test/vector_types.sql`.

Note: Subtypes are ephemeral in SQLite (not persisted to storage, but work in queries).

Optional enhancement (not blocking):
- Update `vec_distance_hamming` and `vec_distance_jaccard` in distance.c to
  require/validate subtype 224 (bitvector) for BLOB inputs

---

### Verification

- All existing run_all.sh tests must still pass after each change
- Add new tests to run_all.sh as they're created
- Benchmark scripts in `bench/` are run separately:
  `luajit bench/param_sweep.lua` and `luajit bench/sift_bench.lua`
- For the large-scale benchmark, define a minimum acceptable recall threshold
  (suggest ≥ 0.90 at default M=16, ef_search=64 on SIFT-1M)

---

**Decisions**

- Test tooling stays Lua + SQL; no Python/pytest introduced
- Benchmark scale: small scripts stay in test, large scripts go in a new
  top-level `bench/` directory
- Metadata columns use a new `_meta` shadow table rather than packing into
  `_data` (keeps the hot HNSW path clean)
- Typed vector types use SQLite subtype markers (not a new column type tag) for
  backward compatibility with existing `vec()` TEXT path
