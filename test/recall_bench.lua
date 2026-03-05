-- test/recall_bench.lua
-- Recall@k + latency benchmark for sqlite-vector HNSW.
--
-- Inserts N random vectors, runs Q HNSW queries and SQL brute-force baseline
-- queries (full-scan + vec_distance_*), computes recall@k, reports query
-- latency stats, and measures delete throughput.
--
-- Writes structured outputs to bench/results:
--   - bench/results/recall_bench.csv
--   - bench/results/recall_bench_latest.json
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

local SQLITE_ROW = sqlite.SQLITE_ROW
local SQLITE_DONE = sqlite.SQLITE_DONE
local SQLITE_TRANSIENT = sqlite.SQLITE_TRANSIENT

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
local DELETE_FRACTION = tonumber(os.getenv("DELETE_FRACTION")) or 0.10
local OUTPUT_DIR = os.getenv("OUTPUT_DIR") or "bench/results"
local DEBUG_RECALL = os.getenv("DEBUG_RECALL") == "1"

-- ── Helpers ─────────────────────────────────────────────────────────────────

local function open_db()
    return sqlite.open_db({
        memory = true
    })
end

local function exec(db, sql)
    sqlite.exec(db, sql)
end

local function prepare(db, sql)
    return sqlite.prepare(db, sql)
end

local function now_seconds()
    return os.clock()
end

local function sort_copy(values)
    local tmp = {}
    for i = 1, #values do
        tmp[i] = values[i]
    end
    table.sort(tmp)
    return tmp
end

local function percentile(sorted_values, p)
    if #sorted_values == 0 then
        return 0.0
    end
    local pos = p * (#sorted_values - 1) + 1
    local lo = math.floor(pos)
    local hi = math.ceil(pos)
    if lo == hi then
        return sorted_values[lo]
    end
    local frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac
end

local function latency_stats(latencies)
    if #latencies == 0 then
        return {
            min = 0,
            p50 = 0,
            p95 = 0,
            p99 = 0,
            max = 0
        }
    end
    local sorted = sort_copy(latencies)
    return {
        min = sorted[1],
        p50 = percentile(sorted, 0.50),
        p95 = percentile(sorted, 0.95),
        p99 = percentile(sorted, 0.99),
        max = sorted[#sorted]
    }
end

local function ensure_output_dir(path)
    os.execute("mkdir -p '" .. path .. "'")
end

local function json_escape(s)
    s = s:gsub("\\", "\\\\")
    s = s:gsub('"', '\\"')
    s = s:gsub("\n", "\\n")
    return s
end

local function write_json(path, obj)
    local f = assert(io.open(path, "w"))
    f:write("{\n")
    local i = 0
    local keys = {}
    for k, _ in pairs(obj) do
        keys[#keys + 1] = k
    end
    table.sort(keys)
    for _, k in ipairs(keys) do
        i = i + 1
        local v = obj[k]
        local tail = (i < #keys) and "," or ""
        if type(v) == "number" then
            f:write(string.format("  \"%s\": %.10g%s\n", json_escape(k), v, tail))
        else
            f:write(string.format("  \"%s\": \"%s\"%s\n", json_escape(k), json_escape(tostring(v)), tail))
        end
    end
    f:write("}\n")
    f:close()
end

local function append_csv(path, header, row)
    local exists = io.open(path, "r") ~= nil
    local f = assert(io.open(path, "a"))
    if not exists then
        f:write(header .. "\n")
    end
    f:write(row .. "\n")
    f:close()
end

-- ── Vector generation ───────────────────────────────────────────────────────

math.randomseed(SEED)

-- Generate a random float vector as "[f1,f2,...,fn]" text.
local function random_vec(dims)
    local vals = {}
    local txt = {}
    for i = 1, dims do
        local v = math.random() * 2 - 1
        vals[i] = v
        txt[i] = string.format("%.6f", v)
    end
    return vals, "[" .. table.concat(txt, ",") .. "]"
end

local function metric_sql_fn(metric)
    local map = {
        l2 = "vec_distance_l2",
        l1 = "vec_distance_l1",
        ip = "vec_distance_ip",
        cosine = "vec_distance_cosine",
        hamming = "vec_distance_hamming",
        jaccard = "vec_distance_jaccard"
    }
    local fn = map[metric]
    if not fn then
        error("unsupported metric for SQL brute-force baseline: " .. tostring(metric))
    end
    return fn
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
local vectors_num = {}
for i = 1, N do
    local vals, txt = random_vec(DIMS)
    vectors_num[i] = vals
    vectors[i] = txt
end
io.write("done\n")

-- ── Insert phase ────────────────────────────────────────────────────────────
io.write("inserting... ")
io.flush()
local t0 = now_seconds()

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

local t_insert = now_seconds() - t0
io.write(string.format("%.3fs (%.0f vec/s)\n", t_insert, N / t_insert))

if DEBUG_RECALL then
    sqlite.query(db, "SELECT key, value FROM bench_config ORDER BY key", function(stmt)
        local k = ffi.string(sq.sqlite3_column_text(stmt, 0))
        local v = ffi.string(sq.sqlite3_column_text(stmt, 1))
        io.write(string.format("debug config %s=%s\n", k, v))
    end)
end

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
local t1 = now_seconds()

local hnsw_results = {} -- query_idx -> {rowid1, rowid2, ...}
local hnsw_total_rows = 0
local hnsw_latencies = {}
for _, qi in ipairs(query_indices) do
    local qvec = vectors[qi]
    local sql = "SELECT rowid FROM bench WHERE bench MATCH '" .. qvec .. "' LIMIT " .. K
    local q0 = now_seconds()
    local stmt = prepare(db, sql)
    local rowids = {}
    while true do
        local rc = sq.sqlite3_step(stmt)
        if rc == SQLITE_ROW then
            rowids[#rowids + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
        elseif rc == SQLITE_DONE then
            break
        else
            local msg = ffi.string(sq.sqlite3_errmsg(db))
            sq.sqlite3_finalize(stmt)
            error(string.format("hnsw query failed (rc=%d): %s\nSQL: %s", rc, msg, sql))
        end
    end
    sq.sqlite3_finalize(stmt)
    hnsw_latencies[#hnsw_latencies + 1] = now_seconds() - q0
    hnsw_results[qi] = rowids
    hnsw_total_rows = hnsw_total_rows + #rowids
end

local t_hnsw = now_seconds() - t1
io.write(string.format("%.3fs (%.1f ms/query)\n", t_hnsw, t_hnsw / #query_indices * 1000))

-- ── Brute-force phase (SQL full-scan baseline) ─────────────────────────────
io.write("brute-force queries... ")
io.flush()
local t2 = now_seconds()

local bf_results = {} -- query_idx -> {rowid1, rowid2, ...}
local bf_total_rows = 0
local bf_latencies = {}
local metric_fn = metric_sql_fn(METRIC)
for _, qi in ipairs(query_indices) do
    local qvec = vectors[qi]
    local sql = string.format("SELECT rowid FROM bench ORDER BY %s(vector, '%s') LIMIT %d", metric_fn, qvec, K)
    local q0 = now_seconds()
    local stmt = prepare(db, sql)
    local rowids = {}
    while true do
        local rc = sq.sqlite3_step(stmt)
        if rc == SQLITE_ROW then
            rowids[#rowids + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
        elseif rc == SQLITE_DONE then
            break
        else
            local msg = ffi.string(sq.sqlite3_errmsg(db))
            sq.sqlite3_finalize(stmt)
            error(string.format("brute-force query failed (rc=%d): %s\nSQL: %s", rc, msg, sql))
        end
    end
    sq.sqlite3_finalize(stmt)
    bf_latencies[#bf_latencies + 1] = now_seconds() - q0
    bf_results[qi] = rowids
    bf_total_rows = bf_total_rows + #rowids
end

local t_bf = now_seconds() - t2
io.write(string.format("%.3fs (%.1f ms/query)\n", t_bf, t_bf / #query_indices * 1000))

-- ── Delete throughput phase ─────────────────────────────────────────────────
io.write("delete throughput... ")
io.flush()
local n_delete = math.max(1, math.floor(N * DELETE_FRACTION))
local d0 = now_seconds()
exec(db, "BEGIN")
for i = 1, n_delete do
    local rid = i
    exec(db, "DELETE FROM bench WHERE rowid=" .. tostring(rid))
end
exec(db, "COMMIT")
local t_delete = now_seconds() - d0
io.write(string.format("%.3fs (%.0f deletes/s)\n", t_delete, n_delete / t_delete))

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
local hnsw_stats = latency_stats(hnsw_latencies)
local bf_stats = latency_stats(bf_latencies)

if DEBUG_RECALL and #query_indices > 0 then
    local qi = query_indices[1]
    local h = hnsw_results[qi] or {}
    local b = bf_results[qi] or {}
    local bset = {}
    for _, rid in ipairs(b) do
        bset[rid] = true
    end
    local hits = 0
    for _, rid in ipairs(h) do
        if bset[rid] then
            hits = hits + 1
        end
    end
    io.write(string.format("debug first query idx=%d hnsw=[%s]\n", qi, table.concat(h, ",")))
    io.write(string.format("debug first query idx=%d brute=[%s]\n", qi, table.concat(b, ",")))
    io.write(string.format("debug first query overlap=%d/%d\n", hits, K))
end

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
io.write(string.format("  %-24s %.1f\n", "hnsw qps:", #query_indices / t_hnsw))
io.write(string.format("  %-24s %.3f/%.3f/%.3f/%.3f/%.3f ms\n", "hnsw p(min/50/95/99/max):", hnsw_stats.min * 1000,
    hnsw_stats.p50 * 1000, hnsw_stats.p95 * 1000, hnsw_stats.p99 * 1000, hnsw_stats.max * 1000))
io.write(string.format("  %-24s %.1f ms\n", "brute-force query (avg):", t_bf / #query_indices * 1000))
io.write(string.format("  %-24s %.1f\n", "brute-force qps:", #query_indices / t_bf))
io.write(string.format("  %-24s %.3f/%.3f/%.3f/%.3f/%.3f ms\n", "brute p(min/50/95/99/max):", bf_stats.min * 1000,
    bf_stats.p50 * 1000, bf_stats.p95 * 1000, bf_stats.p99 * 1000, bf_stats.max * 1000))
io.write(string.format("  %-24s %.1f\n", "hnsw rows/query:", hnsw_total_rows / #query_indices))
io.write(string.format("  %-24s %.1f\n", "brute rows/query:", bf_total_rows / #query_indices))
io.write(string.format("  %-24s %.3fs\n", "delete time:", t_delete))
io.write(string.format("  %-24s %.1f\n", "delete throughput:", n_delete / t_delete))
io.write(string.format("  %-24s %.4f\n", "recall@k (mean):", mean_recall))
io.write(string.format("  %-24s %.4f\n", "recall@k (min):", min_recall))
io.write(string.format("  %-24s %.4f\n", "recall@k (max):", max_recall))
io.write(string.format("  %-24s %.2f\n", "threshold:", RECALL_THRESHOLD))
io.write("\n")

-- ── Structured outputs ─────────────────────────────────────────────────────
ensure_output_dir(OUTPUT_DIR)
local csv_path = OUTPUT_DIR .. "/recall_bench.csv"
local json_path = OUTPUT_DIR .. "/recall_bench_latest.json"

local header = table.concat({"timestamp", "seed", "n", "dims", "q", "k", "metric", "m", "ef_construction", "ef_search",
                             "insert_s", "hnsw_s", "hnsw_qps", "hnsw_min_ms", "hnsw_p50_ms", "hnsw_p95_ms",
                             "hnsw_p99_ms", "hnsw_max_ms", "brute_s", "brute_qps", "brute_min_ms", "brute_p50_ms",
                             "brute_p95_ms", "brute_p99_ms", "brute_max_ms", "delete_s", "delete_n", "delete_per_s",
                             "mean_recall", "min_recall", "max_recall", "threshold"}, ",")

local row = string.format(
    "%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f",
    os.time(), SEED, N, DIMS, #query_indices, K, METRIC, M, EF_CONSTRUCTION, EF_SEARCH, t_insert, t_hnsw,
    #query_indices / t_hnsw, hnsw_stats.min * 1000, hnsw_stats.p50 * 1000, hnsw_stats.p95 * 1000, hnsw_stats.p99 * 1000,
    hnsw_stats.max * 1000, t_bf, #query_indices / t_bf, bf_stats.min * 1000, bf_stats.p50 * 1000, bf_stats.p95 * 1000,
    bf_stats.p99 * 1000, bf_stats.max * 1000, t_delete, n_delete, n_delete / t_delete, mean_recall, min_recall,
    max_recall, RECALL_THRESHOLD)
append_csv(csv_path, header, row)

write_json(json_path, {
    timestamp = os.time(),
    seed = SEED,
    n = N,
    dims = DIMS,
    q = #query_indices,
    k = K,
    metric = METRIC,
    m = M,
    ef_construction = EF_CONSTRUCTION,
    ef_search = EF_SEARCH,
    insert_s = t_insert,
    hnsw_s = t_hnsw,
    hnsw_qps = #query_indices / t_hnsw,
    hnsw_min_ms = hnsw_stats.min * 1000,
    hnsw_p50_ms = hnsw_stats.p50 * 1000,
    hnsw_p95_ms = hnsw_stats.p95 * 1000,
    hnsw_p99_ms = hnsw_stats.p99 * 1000,
    hnsw_max_ms = hnsw_stats.max * 1000,
    brute_s = t_bf,
    brute_qps = #query_indices / t_bf,
    brute_min_ms = bf_stats.min * 1000,
    brute_p50_ms = bf_stats.p50 * 1000,
    brute_p95_ms = bf_stats.p95 * 1000,
    brute_p99_ms = bf_stats.p99 * 1000,
    brute_max_ms = bf_stats.max * 1000,
    delete_s = t_delete,
    delete_n = n_delete,
    delete_per_s = n_delete / t_delete,
    mean_recall = mean_recall,
    min_recall = min_recall,
    max_recall = max_recall,
    threshold = RECALL_THRESHOLD
})

io.write("structured output:\n")
io.write("  " .. csv_path .. "\n")
io.write("  " .. json_path .. "\n\n")

sq.sqlite3_close(db)

if mean_recall >= RECALL_THRESHOLD then
    io.write(string.format("PASS  recall@%d = %.4f >= %.2f\n", K, mean_recall, RECALL_THRESHOLD))
    os.exit(0)
else
    io.write(string.format("FAIL  recall@%d = %.4f < %.2f\n", K, mean_recall, RECALL_THRESHOLD))
    os.exit(1)
end
