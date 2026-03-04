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
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

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

local function metric_distance(metric, a, b)
    if metric == "l2" then
        local sum = 0.0
        for i = 1, #a do
            local d = a[i] - b[i]
            sum = sum + d * d
        end
        return math.sqrt(sum)
    elseif metric == "l1" then
        local sum = 0.0
        for i = 1, #a do
            local d = a[i] - b[i]
            if d < 0 then
                d = -d
            end
            sum = sum + d
        end
        return sum
    elseif metric == "ip" then
        local dot = 0.0
        for i = 1, #a do
            dot = dot + a[i] * b[i]
        end
        return -dot
    elseif metric == "cosine" then
        local dot = 0.0
        local na = 0.0
        local nb = 0.0
        for i = 1, #a do
            dot = dot + a[i] * b[i]
            na = na + a[i] * a[i]
            nb = nb + b[i] * b[i]
        end
        if na == 0.0 or nb == 0.0 then
            return 1.0
        end
        return 1.0 - (dot / (math.sqrt(na) * math.sqrt(nb)))
    end

    error("unsupported metric for Lua brute-force baseline: " .. tostring(metric))
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
-- Use pure Lua distance evaluation over all generated vectors as ground truth.
io.write("brute-force queries... ")
io.flush()
local t2 = os.clock()

local bf_results = {} -- query_idx -> {rowid1, rowid2, ...}
for _, qi in ipairs(query_indices) do
    local qnum = vectors_num[qi]
    local scored = {}

    for candidate = 1, N do
        scored[#scored + 1] = {
            rowid = candidate,
            dist = metric_distance(METRIC, vectors_num[candidate], qnum)
        }
    end

    table.sort(scored, function(a, b)
        return a.dist < b.dist
    end)

    local rowids = {}
    for i = 1, math.min(K, #scored) do
        rowids[#rowids + 1] = scored[i].rowid
    end

    bf_results[qi] = rowids
end

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
