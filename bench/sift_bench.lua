-- bench/sift_bench.lua
-- Large-scale benchmark runner for sqlite-vector, designed for SIFT-like shapes.
--
-- Defaults target 1M x 128D, but can be overridden for quick local runs.
--
-- Usage examples:
--   luajit bench/sift_bench.lua
--   N=100000 Q=200 BRUTE_Q=30 PAGE_SIZES=4096 JOURNAL_MODES=DELETE,WAL luajit bench/sift_bench.lua
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

ffi.cdef [[
int getpid(void);
]]

local SQLITE_ROW = sqlite.SQLITE_ROW
local SQLITE_DONE = sqlite.SQLITE_DONE
local SQLITE_TRANSIENT = sqlite.SQLITE_TRANSIENT

local N = tonumber(os.getenv("N")) or 1000000
local DIMS = tonumber(os.getenv("DIMS")) or 128
local Q = tonumber(os.getenv("Q")) or 1000
local BRUTE_Q = tonumber(os.getenv("BRUTE_Q")) or 100
local K_LIST_RAW = os.getenv("K_LIST") or "10,100"
local METRIC = os.getenv("METRIC") or "l2"
local M = tonumber(os.getenv("M")) or 16
local EF_CONSTRUCTION = tonumber(os.getenv("EF_CONSTRUCTION")) or 200
local EF_SEARCH = tonumber(os.getenv("EF_SEARCH")) or 64
local BATCH = tonumber(os.getenv("BATCH_SIZE")) or 10000
local PAGE_SIZES_RAW = os.getenv("PAGE_SIZES") or "4096,8192,16384,65536"
local JOURNAL_MODES_RAW = os.getenv("JOURNAL_MODES") or "DELETE,WAL"
local SEED = tonumber(os.getenv("SEED")) or 42
local OUTPUT_DIR = os.getenv("OUTPUT_DIR") or "bench/results"
local OUTPUT_CSV = OUTPUT_DIR .. "/sift_bench.csv"

local function now_seconds()
    return os.clock()
end

local function parse_int_list(raw)
    local out = {}
    for token in string.gmatch(raw, "[^,]+") do
        out[#out + 1] = tonumber(token)
    end
    return out
end

local function parse_str_list(raw)
    local out = {}
    for token in string.gmatch(raw, "[^,]+") do
        out[#out + 1] = token
    end
    return out
end

local K_LIST = parse_int_list(K_LIST_RAW)
local PAGE_SIZES = parse_int_list(PAGE_SIZES_RAW)
local JOURNAL_MODES = parse_str_list(JOURNAL_MODES_RAW)

local function ensure_output_dir(path)
    os.execute("mkdir -p '" .. path .. "'")
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

local function deterministic_value(i, d, seed)
    local x = (i * 1103515245 + d * 12345 + seed * 2654435761) % 2147483647
    return (x / 2147483647.0) * 2.0 - 1.0
end

local function vec_text(i, dims, seed)
    local out = {}
    for d = 1, dims do
        out[d] = string.format("%.6f", deterministic_value(i, d, seed))
    end
    return "[" .. table.concat(out, ",") .. "]"
end

local function metric_sql_fn(metric)
    local map = {
        l2 = "vec_distance_l2",
        l1 = "vec_distance_l1",
        ip = "vec_distance_ip",
        cosine = "vec_distance_cosine"
    }
    local fn = map[metric]
    if not fn then
        error("unsupported metric for sift bench: " .. tostring(metric))
    end
    return fn
end

local function rss_kb()
    local f = io.open("/proc/self/status", "r")
    if f then
        for line in f:lines() do
            local v = line:match("^VmRSS:%s+(%d+)")
            if v then
                f:close()
                return tonumber(v)
            end
        end
        f:close()
    end

    local pid = tonumber(ffi.C.getpid())
    local p = io.popen("ps -o rss= -p " .. tostring(pid) .. " 2>/dev/null")
    if p then
        local s = p:read("*a") or ""
        p:close()
        s = s:gsub("%s+", "")
        return tonumber(s) or 0
    end
    return 0
end

local function query_ids(n, q)
    local ids = {}
    local step = math.max(1, math.floor(n / q))
    for i = 1, n, step do
        ids[#ids + 1] = i
        if #ids >= q then
            break
        end
    end
    return ids
end

local function remove_db(path)
    os.remove(path)
    os.remove(path .. "-wal")
    os.remove(path .. "-shm")
end

local function open_file_db(path)
    return sqlite.open_db({
        filename = path,
        memory = false
    })
end

local function exec(db, sql)
    sqlite.exec(db, sql)
end

local function prepare(db, sql)
    return sqlite.prepare(db, sql)
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
    table.sort(latencies)
    local function pct(p)
        local pos = p * (#latencies - 1) + 1
        local lo = math.floor(pos)
        local hi = math.ceil(pos)
        if lo == hi then
            return latencies[lo]
        end
        local frac = pos - lo
        return latencies[lo] + (latencies[hi] - latencies[lo]) * frac
    end
    return {
        min = latencies[1],
        p50 = pct(0.50),
        p95 = pct(0.95),
        p99 = pct(0.99),
        max = latencies[#latencies]
    }
end

local function run_config(journal_mode, page_size)
    local dbpath = string.format("/tmp/sqlite_vector_sift_%s_%d.db", journal_mode, page_size)
    remove_db(dbpath)

    local db = open_file_db(dbpath)
    exec(db, "PRAGMA page_size=" .. tostring(page_size))
    exec(db, "PRAGMA journal_mode=" .. journal_mode)
    exec(db,
        string.format(
            "CREATE VIRTUAL TABLE bench USING vec0(dims=%d, metric=%s, m=%d, ef_construction=%d, ef_search=%d)", DIMS,
            METRIC, M, EF_CONSTRUCTION, EF_SEARCH))

    local rss_samples = {}
    rss_samples[#rss_samples + 1] = {
        at = 0,
        rss_kb = rss_kb()
    }

    local ins = prepare(db, "INSERT INTO bench(vector) VALUES(?)")
    local t_insert = now_seconds()
    local batch_times = {}

    exec(db, "BEGIN")
    local i = 1
    while i <= N do
        local b0 = now_seconds()
        local upper = math.min(N, i + BATCH - 1)
        for rid = i, upper do
            local vtxt = vec_text(rid, DIMS, SEED)
            sq.sqlite3_bind_text(ins, 1, vtxt, #vtxt, SQLITE_TRANSIENT)
            local rc = sq.sqlite3_step(ins)
            assert(rc == SQLITE_DONE, "insert failed: " .. rc)
            sq.sqlite3_reset(ins)
        end
        batch_times[#batch_times + 1] = now_seconds() - b0

        if upper >= 100000 and not rss_samples[2] then
            rss_samples[2] = {
                at = 100000,
                rss_kb = rss_kb()
            }
        end
        if upper >= 500000 and not rss_samples[3] then
            rss_samples[3] = {
                at = 500000,
                rss_kb = rss_kb()
            }
        end
        if upper >= 1000000 and not rss_samples[4] then
            rss_samples[4] = {
                at = 1000000,
                rss_kb = rss_kb()
            }
        end

        i = upper + 1
    end
    exec(db, "COMMIT")
    sq.sqlite3_finalize(ins)
    local insert_s = now_seconds() - t_insert
    rss_samples[#rss_samples + 1] = {
        at = N,
        rss_kb = rss_kb()
    }

    local qids = query_ids(N, Q)
    local fn = metric_sql_fn(METRIC)

    local by_k = {}
    for _, k in ipairs(K_LIST) do
        local hnsw_lat = {}
        local hnsw_results = {}
        local t0 = now_seconds()
        for _, qi in ipairs(qids) do
            local qvec = vec_text(qi, DIMS, SEED)
            local stmt = prepare(db, "SELECT rowid FROM bench WHERE bench MATCH '" .. qvec .. "' LIMIT " .. k)
            local q0 = now_seconds()
            local rows = {}
            while true do
                local rc = sq.sqlite3_step(stmt)
                if rc == SQLITE_ROW then
                    rows[#rows + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
                elseif rc == SQLITE_DONE then
                    break
                else
                    local msg = ffi.string(sq.sqlite3_errmsg(db))
                    sq.sqlite3_finalize(stmt)
                    error("hnsw failed: " .. msg)
                end
            end
            sq.sqlite3_finalize(stmt)
            hnsw_lat[#hnsw_lat + 1] = now_seconds() - q0
            hnsw_results[qi] = rows
        end
        local hnsw_s = now_seconds() - t0

        local brute_n = math.min(BRUTE_Q, #qids)
        local brute_lat = {}
        local recall_sum = 0.0
        local t1 = now_seconds()
        for j = 1, brute_n do
            local qi = qids[j]
            local qvec = vec_text(qi, DIMS, SEED)
            local stmt = prepare(db, string.format("SELECT rowid FROM bench ORDER BY %s(vector, '%s') LIMIT %d", fn,
                qvec, k))
            local b0 = now_seconds()
            local gt = {}
            while true do
                local rc = sq.sqlite3_step(stmt)
                if rc == SQLITE_ROW then
                    gt[#gt + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
                elseif rc == SQLITE_DONE then
                    break
                else
                    local msg = ffi.string(sq.sqlite3_errmsg(db))
                    sq.sqlite3_finalize(stmt)
                    error("brute failed: " .. msg)
                end
            end
            sq.sqlite3_finalize(stmt)
            brute_lat[#brute_lat + 1] = now_seconds() - b0

            local set = {}
            for _, rid in ipairs(gt) do
                set[rid] = true
            end
            local hit = 0
            for _, rid in ipairs(hnsw_results[qi] or {}) do
                if set[rid] then
                    hit = hit + 1
                end
            end
            recall_sum = recall_sum + (hit / k)
        end
        local brute_s = now_seconds() - t1

        by_k[k] = {
            hnsw_s = hnsw_s,
            hnsw_qps = #qids / hnsw_s,
            hnsw_stats = latency_stats(hnsw_lat),
            brute_s = brute_s,
            brute_qps = brute_n / brute_s,
            brute_stats = latency_stats(brute_lat),
            recall = (math.min(BRUTE_Q, #qids) > 0) and (recall_sum / math.min(BRUTE_Q, #qids)) or 0,
            brute_n = brute_n,
            hnsw_n = #qids
        }
    end

    sq.sqlite3_close(db)
    remove_db(dbpath)

    local batch_avg = 0.0
    for _, t in ipairs(batch_times) do
        batch_avg = batch_avg + t
    end
    if #batch_times > 0 then
        batch_avg = batch_avg / #batch_times
    end

    return {
        page_size = page_size,
        journal_mode = journal_mode,
        insert_s = insert_s,
        insert_per_s = N / insert_s,
        batch_avg_s = batch_avg,
        rss_samples = rss_samples,
        by_k = by_k
    }
end

ensure_output_dir(OUTPUT_DIR)

local header = table.concat({"timestamp", "seed", "n", "dims", "metric", "m", "ef_construction", "ef_search",
                             "journal_mode", "page_size", "k", "insert_s", "insert_per_s", "batch_avg_s", "hnsw_q",
                             "hnsw_s", "hnsw_qps", "hnsw_min_ms", "hnsw_p50_ms", "hnsw_p95_ms", "hnsw_p99_ms",
                             "hnsw_max_ms", "brute_q", "brute_s", "brute_qps", "brute_min_ms", "brute_p50_ms",
                             "brute_p95_ms", "brute_p99_ms", "brute_max_ms", "recall", "rss_0_kb", "rss_100k_kb",
                             "rss_500k_kb", "rss_1m_kb", "rss_final_kb"}, ",")

print(string.format("sift_bench target: N=%d DIMS=%d Q=%d BRUTE_Q=%d metric=%s", N, DIMS, Q, BRUTE_Q, METRIC))
print(string.format("params: m=%d ef_c=%d ef_s=%d batch=%d", M, EF_CONSTRUCTION, EF_SEARCH, BATCH))

for _, journal_mode in ipairs(JOURNAL_MODES) do
    for _, page_size in ipairs(PAGE_SIZES) do
        print(string.format("running journal=%s page_size=%d", journal_mode, page_size))
        local r = run_config(journal_mode, page_size)

        local rss_map = {
            [0] = 0,
            [100000] = 0,
            [500000] = 0,
            [1000000] = 0,
            final = 0
        }
        for _, s in ipairs(r.rss_samples) do
            if s.at == 0 then
                rss_map[0] = s.rss_kb
            end
            if s.at == 100000 then
                rss_map[100000] = s.rss_kb
            end
            if s.at == 500000 then
                rss_map[500000] = s.rss_kb
            end
            if s.at == 1000000 then
                rss_map[1000000] = s.rss_kb
            end
            if s.at == N then
                rss_map.final = s.rss_kb
            end
        end

        for _, k in ipairs(K_LIST) do
            local x = r.by_k[k]
            print(string.format("  k=%d hnsw_qps=%.1f brute_qps=%.1f recall=%.4f", k, x.hnsw_qps, x.brute_qps, x.recall))

            local row = string.format(
                "%d,%d,%d,%d,%s,%d,%d,%d,%s,%d,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d",
                os.time(), SEED, N, DIMS, METRIC, M, EF_CONSTRUCTION, EF_SEARCH, r.journal_mode, r.page_size, k,
                r.insert_s, r.insert_per_s, r.batch_avg_s, x.hnsw_n, x.hnsw_s, x.hnsw_qps, x.hnsw_stats.min * 1000,
                x.hnsw_stats.p50 * 1000, x.hnsw_stats.p95 * 1000, x.hnsw_stats.p99 * 1000, x.hnsw_stats.max * 1000,
                x.brute_n, x.brute_s, x.brute_qps, x.brute_stats.min * 1000, x.brute_stats.p50 * 1000,
                x.brute_stats.p95 * 1000, x.brute_stats.p99 * 1000, x.brute_stats.max * 1000, x.recall, rss_map[0],
                rss_map[100000], rss_map[500000], rss_map[1000000], rss_map.final)
            append_csv(OUTPUT_CSV, header, row)
        end
    end
end

print("wrote: " .. OUTPUT_CSV)
