-- test/recall_bench.lua
-- Recall@k benchmark for sqlite-vector HNSW.
--
-- Inserts N random vectors, queries Q of them via HNSW (MATCH),
-- compares against brute-force (full-scan + vec_distance_*) to
-- compute recall@k.  Exits 0 if mean recall >= RECALL_THRESHOLD.
--
-- Usage:  luajit test/recall_bench.lua
--         N=5000 DIMS=256 K=20 luajit test/recall_bench.lua
local ffi = require "ffi"

-- ── SQLite C API declarations ───────────────────────────────────────────────
ffi.cdef [[
  typedef struct sqlite3      sqlite3;
  typedef struct sqlite3_stmt sqlite3_stmt;

  int    sqlite3_open_v2(const char *filename, sqlite3 **ppDb, int flags, const char *zVfs);
  int    sqlite3_close(sqlite3 *db);
  int    sqlite3_exec(sqlite3 *db, const char *sql, void *cb, void *arg, char **errmsg);
  void   sqlite3_free(void *p);
  int    sqlite3_enable_load_extension(sqlite3 *db, int onoff);
  int    sqlite3_load_extension(sqlite3 *db, const char *file, const char *proc, char **errmsg);

  int    sqlite3_prepare_v2(sqlite3 *db, const char *sql, int nBytes,
                            sqlite3_stmt **ppStmt, const char **pzTail);
  int    sqlite3_step(sqlite3_stmt *stmt);
  int    sqlite3_finalize(sqlite3_stmt *stmt);
  int    sqlite3_reset(sqlite3_stmt *stmt);
  int    sqlite3_bind_text(sqlite3_stmt *stmt, int idx, const char *val, int n, void *dtor);

  int         sqlite3_column_count(sqlite3_stmt *stmt);
  int         sqlite3_column_type(sqlite3_stmt *stmt, int col);
  double      sqlite3_column_double(sqlite3_stmt *stmt, int col);
  int64_t     sqlite3_column_int64(sqlite3_stmt *stmt, int col);
  const unsigned char *sqlite3_column_text(sqlite3_stmt *stmt, int col);

  const char *sqlite3_errmsg(sqlite3 *db);
]]

local SQLITE_OK = 0
local SQLITE_ROW = 100
local SQLITE_DONE = 101
local SQLITE_OPEN_READWRITE = 0x00000002
local SQLITE_OPEN_CREATE = 0x00000004
local SQLITE_OPEN_MEMORY = 0x00000080
local SQLITE_OPEN_URI = 0x00000040
local SQLITE_TRANSIENT = ffi.cast("void*", -1)

local sq = ffi.load("/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib")

-- ── Configuration ───────────────────────────────────────────────────────────
-- Override via environment variables: N=5000 luajit test/recall_bench.lua

local N = tonumber(os.getenv("N")) or 1000
local DIMS = tonumber(os.getenv("DIMS")) or 64
local Q = tonumber(os.getenv("Q")) or 50
local K = tonumber(os.getenv("K")) or 10
local METRIC = os.getenv("METRIC") or "l2"
local M = tonumber(os.getenv("M")) or 16
local EF_CONSTRUCTION = tonumber(os.getenv("EF_CONSTRUCTION")) or 200
local EF_SEARCH = tonumber(os.getenv("EF_SEARCH")) or 64
local RECALL_THRESHOLD = tonumber(os.getenv("RECALL_THRESHOLD")) or 0.80
local SEED = tonumber(os.getenv("SEED")) or 42

-- ── Helpers ─────────────────────────────────────────────────────────────────

local DYLIB = "build/macosx/arm64/release/libsqlite_vector.dylib"

local function open_db()
    local flags = SQLITE_OPEN_READWRITE + SQLITE_OPEN_CREATE + SQLITE_OPEN_MEMORY + SQLITE_OPEN_URI
    local dbp = ffi.new("sqlite3*[1]")
    local rc = sq.sqlite3_open_v2(":memory:", dbp, flags, nil)
    assert(rc == SQLITE_OK, "sqlite3_open_v2 failed: " .. rc)
    local db = dbp[0]
    sq.sqlite3_enable_load_extension(db, 1)
    local errmsg = ffi.new("char*[1]")
    rc = sq.sqlite3_load_extension(db, DYLIB, nil, errmsg)
    if rc ~= SQLITE_OK then
        local msg = ffi.string(errmsg[0])
        sq.sqlite3_free(errmsg[0])
        error("load_extension failed: " .. msg)
    end
    return db
end

local function exec(db, sql)
    local errmsg = ffi.new("char*[1]")
    local rc = sq.sqlite3_exec(db, sql, nil, nil, errmsg)
    if rc ~= SQLITE_OK then
        local msg = ffi.string(errmsg[0])
        sq.sqlite3_free(errmsg[0])
        error("exec failed (" .. rc .. "): " .. msg .. "\nSQL: " .. sql)
    end
end

local function prepare(db, sql)
    local stmtp = ffi.new("sqlite3_stmt*[1]")
    local rc = sq.sqlite3_prepare_v2(db, sql, -1, stmtp, nil)
    assert(rc == SQLITE_OK, "prepare failed: " .. ffi.string(sq.sqlite3_errmsg(db)) .. "\nSQL: " .. sql)
    return stmtp[0]
end

-- ── Vector generation ───────────────────────────────────────────────────────

math.randomseed(SEED)

-- Generate a random float vector as "[f1,f2,...,fn]" text.
local function random_vec(dims)
    local t = {}
    for i = 1, dims do
        t[i] = string.format("%.6f", math.random() * 2 - 1) -- [-1, 1]
    end
    return "[" .. table.concat(t, ",") .. "]"
end

-- ── Main ────────────────────────────────────────────────────────────────────

io.write(string.format("recall_bench: N=%d DIMS=%d Q=%d K=%d metric=%s m=%d ef_c=%d ef_s=%d\n", N, DIMS, Q, K, METRIC,
    M, EF_CONSTRUCTION, EF_SEARCH))

local db = open_db()

-- Create virtual table
exec(db, string.format(
    "CREATE VIRTUAL TABLE bench USING vec0(dims=%d, metric=%s, m=%d, " .. "ef_construction=%d, ef_search=%d)", DIMS,
    METRIC, M, EF_CONSTRUCTION, EF_SEARCH))

-- Generate all vectors up front (so queries use same vectors)
io.write("generating vectors... ")
io.flush()
local vectors = {}
for i = 1, N do
    vectors[i] = random_vec(DIMS)
end
io.write("done\n")

-- ── Insert phase ────────────────────────────────────────────────────────────
io.write("inserting... ")
io.flush()
local t0 = os.clock()

local ins_stmt = prepare(db, "INSERT INTO bench(vector) VALUES(?)")
exec(db, "BEGIN")
for i = 1, N do
    sq.sqlite3_bind_text(ins_stmt, 1, vectors[i], #vectors[i], SQLITE_TRANSIENT)
    local rc = sq.sqlite3_step(ins_stmt)
    assert(rc == SQLITE_DONE, "insert step failed: " .. rc)
    sq.sqlite3_reset(ins_stmt)
end
exec(db, "COMMIT")
sq.sqlite3_finalize(ins_stmt)

local t_insert = os.clock() - t0
io.write(string.format("%.3fs (%.0f vec/s)\n", t_insert, N / t_insert))

-- ── Pick query indices ──────────────────────────────────────────────────────
local query_indices = {}
local step = math.max(1, math.floor(N / Q))
for i = 1, N, step do
    query_indices[#query_indices + 1] = i
    if #query_indices >= Q then
        break
    end
end

-- ── HNSW query phase ────────────────────────────────────────────────────────
io.write("HNSW queries... ")
io.flush()
local t1 = os.clock()

local match_stmt = prepare(db, "SELECT rowid FROM bench WHERE bench MATCH ? LIMIT " .. K)

local hnsw_results = {} -- query_idx -> {rowid1, rowid2, ...}
for _, qi in ipairs(query_indices) do
    local qvec = vectors[qi]
    sq.sqlite3_bind_text(match_stmt, 1, qvec, #qvec, SQLITE_TRANSIENT)
    local rowids = {}
    while sq.sqlite3_step(match_stmt) == SQLITE_ROW do
        rowids[#rowids + 1] = tonumber(sq.sqlite3_column_int64(match_stmt, 0))
    end
    sq.sqlite3_reset(match_stmt)
    hnsw_results[qi] = rowids
end
sq.sqlite3_finalize(match_stmt)

local t_hnsw = os.clock() - t1
io.write(string.format("%.3fs (%.1f ms/query)\n", t_hnsw, t_hnsw / #query_indices * 1000))

-- ── Brute-force phase ───────────────────────────────────────────────────────
-- Use vtab full-scan + vec_distance_* scalar for ground truth.
io.write("brute-force queries... ")
io.flush()
local t2 = os.clock()

local dist_fn_name = "vec_distance_" .. METRIC
local bf_sql = string.format("SELECT rowid FROM bench ORDER BY %s(vector, ?) LIMIT %d", dist_fn_name, K)
local bf_stmt = prepare(db, bf_sql)

local bf_results = {} -- query_idx -> {rowid1, rowid2, ...}
for _, qi in ipairs(query_indices) do
    local qvec = vectors[qi]
    sq.sqlite3_bind_text(bf_stmt, 1, qvec, #qvec, SQLITE_TRANSIENT)
    local rowids = {}
    while sq.sqlite3_step(bf_stmt) == SQLITE_ROW do
        rowids[#rowids + 1] = tonumber(sq.sqlite3_column_int64(bf_stmt, 0))
    end
    sq.sqlite3_reset(bf_stmt)
    bf_results[qi] = rowids
end
sq.sqlite3_finalize(bf_stmt)

local t_bf = os.clock() - t2
io.write(string.format("%.3fs (%.1f ms/query)\n", t_bf, t_bf / #query_indices * 1000))

-- ── Recall computation ──────────────────────────────────────────────────────
local total_recall = 0
local min_recall = 1.0
local max_recall = 0.0

for _, qi in ipairs(query_indices) do
    -- Build set from brute-force results
    local bf_set = {}
    for _, rid in ipairs(bf_results[qi]) do
        bf_set[rid] = true
    end

    -- Count intersection
    local hits = 0
    for _, rid in ipairs(hnsw_results[qi]) do
        if bf_set[rid] then
            hits = hits + 1
        end
    end

    local recall = hits / K
    total_recall = total_recall + recall
    if recall < min_recall then
        min_recall = recall
    end
    if recall > max_recall then
        max_recall = recall
    end
end

local mean_recall = total_recall / #query_indices

-- ── Report ──────────────────────────────────────────────────────────────────
io.write("\n")
io.write(string.format("  %-24s %d\n", "vectors:", N))
io.write(string.format("  %-24s %d\n", "dimensions:", DIMS))
io.write(string.format("  %-24s %s\n", "metric:", METRIC))
io.write(string.format("  %-24s %d\n", "k:", K))
io.write(string.format("  %-24s %d\n", "queries:", #query_indices))
io.write(string.format("  %-24s %d\n", "m:", M))
io.write(string.format("  %-24s %d\n", "ef_construction:", EF_CONSTRUCTION))
io.write(string.format("  %-24s %d\n", "ef_search:", EF_SEARCH))
io.write(string.format("  %-24s %.3fs\n", "insert time:", t_insert))
io.write(string.format("  %-24s %.1f ms\n", "hnsw query (avg):", t_hnsw / #query_indices * 1000))
io.write(string.format("  %-24s %.1f ms\n", "brute-force query (avg):", t_bf / #query_indices * 1000))
io.write(string.format("  %-24s %.4f\n", "recall@k (mean):", mean_recall))
io.write(string.format("  %-24s %.4f\n", "recall@k (min):", min_recall))
io.write(string.format("  %-24s %.4f\n", "recall@k (max):", max_recall))
io.write(string.format("  %-24s %.2f\n", "threshold:", RECALL_THRESHOLD))
io.write("\n")

sq.sqlite3_close(db)

if mean_recall >= RECALL_THRESHOLD then
    io.write(string.format("PASS  recall@%d = %.4f >= %.2f\n", K, mean_recall, RECALL_THRESHOLD))
    os.exit(0)
else
    io.write(string.format("FAIL  recall@%d = %.4f < %.2f\n", K, mean_recall, RECALL_THRESHOLD))
    os.exit(1)
end
