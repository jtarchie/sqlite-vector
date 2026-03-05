## Plan: Expand Testing, Benchmarking & Features

The project currently has solid correctness tests for its core HNSW + kNN path, but is missing edge-case error handling, error message assertions, fuzzing, distance-threshold filtering, element-wise vector ops, metadata columns, typed vector types, and any industry-scale benchmarking. We'll work across all three tracks in parallel, keeping the existing Lua + SQL tooling.

---

### Track 1 — Close Testing Gaps

**1.1 Error message assertions**

Extend insert.sql to capture the exact SQLite error text on dimension mismatch (e.g. `"Expected 3 dimensions but got 4"`) using `.catch` or `;` error trapping in the Lua wrapper. Extend ffi_test.lua to assert the error string via the FFI `errmsg` call. Mirror what sqlite-vec does in its `test-insert-delete.py`.

**1.2 Parsing edge cases**

Add `test/vec_parse_edge.sql` covering:
- Empty: `vec('[]')`
- Malformed: `vec('[1,]')`, `vec('[,]')`, `vec('[1,,3]')`, `vec('[1a]')`, `vec('[1,2,3')`  (unterminated)
- Special floats: `vec('[inf]')`, `vec('[nan]')`, `vec('[-inf]')`  — decide policy (reject or allow) and assert consistently
- Trailing garbage: `vec('[1,2,3]9')`
- Overflow / underflow floats

**1.3 Input validation limits**

Add guards in vtab.c for: `dims = 0`, `dims > 8192` (match sqlite-vec's cap), and `LIMIT > 4096` on kNN. Add assertions in insert.sql / ffi_test.lua for each error.

**1.4 Hamming/Jaccard semantic guards**

The current distance.c implements `vec_distance_hamming` and `vec_distance_jaccard` by reinterpreting float32 memory as bytes, which is semantically wrong for float inputs. Either (a) enforce that these functions require a dedicated bitvector type (see Track 3, item 3.5), or (b) document and test the current byte-reinterpretation contract explicitly. Add `test/hamming_jaccard.sql` with known-answer tests on byte-aligned bit patterns.

**1.5 Empty-table edge cases**

Add a test (extend delete_repair.sql or new file) that:
- Deletes all rows, then runs a kNN MATCH — expects 0 results, no crash
- Inserts after full deletion — confirms correct count and navigable graph

**1.6 WAL mode persistence**

Add `test/wal_persist.lua` (parallel to persist_test.lua) that:
- Opens file-based DB with `PRAGMA journal_mode=WAL`
- Inserts vectors, checkpoints, closes, reopens
- Verifies config values and kNN results survive

**1.7 EXPLAIN QUERY PLAN output**

Extend ffi_test.lua to run `EXPLAIN QUERY PLAN` on each index path (idxNum 0, 1, 2, 151–156) and assert the `detail` column contains expected strings (e.g. `"MATCH"` for kNN, `"SCAN"` for full table scan).

**1.8 Fuzz / crash-safety sweep**

Add `test/fuzz.lua` that calls every registered SQL function with obviously bad inputs (`NULL`, empty string, `""`, integer, float, `'\x00'`, huge string) and asserts SQLite doesn't crash (only `SQLITE_ERROR` or `SQLITE_OK` are acceptable return codes).

**1.9 RENAME TABLE**

Once `xRename` is implemented (Track 3, item 3.6), add `test/rename.sql` asserting shadow tables are renamed and kNN still works post-rename.

---

### Track 2 — Benchmarking

**2.1 Enhance `recall_bench.lua` (CI-compatible, ≤100K vectors)**

Augment recall_bench.lua with:
- Query latency: wall-clock per query → report min/p50/p95/p99/max using `socket.gettime` or `os.clock`
- QPS (queries per second)
- Delete throughput (deletes/second for N random deletions)
- A brute-force baseline using `vec_distance_*` full-scan, so recall and speed are compared against ground truth in the same run
- Structured output (CSV or JSON) to `bench/results/` for easy graphing

**2.2 Parameter sweep script (CI-compatible)**

New `bench/param_sweep.lua` that loops over a grid of `(M, ef_construction, ef_search)` values and records recall@10 and QPS for each combination. Outputs a table to stdout and a CSV. This directly answers the "M × ef_construction trade-off" question that neither sqlite-vec nor pgvector benchmarks.

**2.3 Large-scale benchmark (separate, not in CI)**

New `bench/sift_bench.lua` targeting 1M × 128D:
- Downloads or generates synthetic SIFT-1M-shaped data (128D float32, 1M vectors, 10K queries)
- Measures insert throughput in batches of 10K
- Measures kNN latency at `k=10`, `k=100`
- Optionally runs `PRAGMA page_size` variations (4096, 8192, 16384, 65536)
- Compares HNSW kNN vs brute-force scan (`vec_distance_l2` full table scan) for recall and latency
- Compares `journal_mode=DELETE` vs `journal_mode=WAL` for insert throughput

**2.4 Memory tracking**

In `bench/sift_bench.lua`, sample `/proc/self/status` (Linux) or `ps -o rss` (macOS) at: before index build, after 100K inserts, after 500K inserts, after 1M inserts. Report RSS growth trend to identify memory leak risk.

---

### Track 3 — New Features

Each feature below needs: C implementation, function registration in extension.c, and a corresponding SQL test file.

**3.1 Element-wise vector ops: `vec_add`, `vec_sub`, `vec_normalize`**

Add to distance.c (or a new `src/vec_ops.c`):
- `vec_add(a TEXT, b TEXT) → TEXT`: element-wise sum, same dims required
- `vec_sub(a TEXT, b TEXT) → TEXT`: element-wise difference
- `vec_normalize(v TEXT) → TEXT`: divide each element by L2 norm (unit vector)

Register in extension.c. Test in `test/vec_ops.sql` with known-answer assertions.

**3.2 `vec_slice(v, start, end) → TEXT`**

Extracts a contiguous subvector (0-indexed, exclusive end). Useful for Matryoshka embeddings. Add to vec_parse.c. Test in `test/vec_ops.sql`.

**3.3 Distance threshold constraints (`AND distance < X`)**

Modify `xBestIndex` in vtab.c to detect `SQLITE_INDEX_CONSTRAINT_LT/LE/GT/GE` or `BETWEEN` on the `distance` column when a MATCH constraint is also present. Pass the threshold value(s) via `aConstraintUsage`. In the kNN path of `xFilter`, post-filter the HNSW result set against the threshold before returning rows. New `test/distance_constraints.sql`.

**3.4 Metadata columns in `vec0`**

This is the largest feature. The declaration syntax becomes:
```sql
CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, name TEXT, score REAL);
```

Changes required:
- Parse extra `name TYPE` pairs in `xCreate`/`xConnect` in vtab.c
- Create a `{name}_meta` shadow table with `(id INTEGER PRIMARY KEY, ...)` for each metadata column
- Extend `xColumn` to read from the meta table for non-hidden, non-vector columns
- Extend `xUpdate` (INSERT/UPDATE) to write metadata alongside the vector
- Extend `xBestIndex` to recognize equality/range constraints on metadata columns and pass them as post-filters on the kNN result set (or as a pre-filter in the full-scan path)
- New `test/metadata.sql`

**3.5 Typed vector types: `vec_int8`, `vec_bit` constructors and `vec_type()`**

Introduce BLOB subtype tagging (mirroring sqlite-vec's subtype 223/224/225 approach):
- `vec_f32(text|blob) → BLOB` with subtype 223: normalizes to raw float32 little-endian BLOB
- `vec_int8(text|blob) → BLOB` with subtype 225: normalizes to raw int8 BLOB
- `vec_bit(blob) → BLOB` with subtype 224: validates packed bit BLOB
- `vec_type(v) → TEXT`: returns `'float32'`, `'int8'`, or `'bit'`
- Update `vec_distance_hamming` and `vec_distance_jaccard` in distance.c to require subtype 224 (bitvector) input and raise `SQLITE_ERROR` otherwise
- New `test/vector_types.sql`

**3.6 RENAME TABLE (`xRename`)**

Implement `xRename` in vtab.c by executing `ALTER TABLE {old}_config RENAME TO {new}_config` etc. for all 4 shadow tables, and updating the in-memory `vtab` state. Test in `test/rename.sql`.

---

### Verification

- All existing run_all.sh tests must still pass after each change
- Add new tests to run_all.sh as they're created
- Benchmark scripts in `bench/` are run separately: `luajit bench/param_sweep.lua` and `luajit bench/sift_bench.lua`
- For the large-scale benchmark, define a minimum acceptable recall threshold (suggest ≥ 0.90 at default M=16, ef_search=64 on SIFT-1M)

---

**Decisions**
- Test tooling stays Lua + SQL; no Python/pytest introduced
- Benchmark scale: small scripts stay in test, large scripts go in a new top-level `bench/` directory
- Metadata columns use a new `_meta` shadow table rather than packing into `_data` (keeps the hot HNSW path clean)
- Typed vector types use SQLite subtype markers (not a new column type tag) for backward compatibility with existing `vec()` TEXT path