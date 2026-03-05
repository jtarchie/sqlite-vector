-- bench/param_sweep.lua
-- CI-friendly parameter sweep for HNSW quality/speed trade-offs.
--
-- Usage:
--   luajit bench/param_sweep.lua
--   N=3000 DIMS=64 Q=40 K=10 M_LIST=8,16 EFC_LIST=100,200 EFS_LIST=16,32,64 luajit bench/param_sweep.lua
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

local SQLITE_ROW = sqlite.SQLITE_ROW
local SQLITE_DONE = sqlite.SQLITE_DONE
local SQLITE_TRANSIENT = sqlite.SQLITE_TRANSIENT

local N = tonumber(os.getenv("N")) or 4000
local DIMS = tonumber(os.getenv("DIMS")) or 64
local Q = tonumber(os.getenv("Q")) or 40
local K = tonumber(os.getenv("K")) or 10
local METRIC = os.getenv("METRIC") or "l2"
local SEED = tonumber(os.getenv("SEED")) or 42
local OUTPUT_DIR = os.getenv("OUTPUT_DIR") or "bench/results"
local OUTPUT_CSV = OUTPUT_DIR .. "/param_sweep.csv"

local function parse_int_list(s, default)
    local raw = s or default
    local out = {}
    for token in string.gmatch(raw, "[^,]+") do
        out[#out + 1] = tonumber(token)
    end
    return out
end

local M_LIST = parse_int_list(os.getenv("M_LIST"), "8,16,32")
local EFC_LIST = parse_int_list(os.getenv("EFC_LIST"), "100,200")
local EFS_LIST = parse_int_list(os.getenv("EFS_LIST"), "16,32,64")

local function now_seconds()
    return os.clock()
end

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

math.randomseed(SEED)
local vectors = {}
for i = 1, N do
    local txt = {}
    for d = 1, DIMS do
        txt[d] = string.format("%.6f", math.random() * 2 - 1)
    end
    vectors[i] = "[" .. table.concat(txt, ",") .. "]"
end

local function query_indices()
    local idx = {}
    local step = math.max(1, math.floor(N / Q))
    for i = 1, N, step do
        idx[#idx + 1] = i
        if #idx >= Q then
            break
        end
    end
    return idx
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
        error("unsupported metric for param sweep: " .. tostring(metric))
    end
    return fn
end

local function run_case(m, efc, efs)
    local db = open_db()
    exec(db,
        string.format(
            "CREATE VIRTUAL TABLE bench USING vec0(dims=%d, metric=%s, m=%d, ef_construction=%d, ef_search=%d)", DIMS,
            METRIC, m, efc, efs))

    local ins = prepare(db, "INSERT INTO bench(vector) VALUES(?)")
    local t0 = now_seconds()
    exec(db, "BEGIN")
    for i = 1, N do
        sq.sqlite3_bind_text(ins, 1, vectors[i], #vectors[i], SQLITE_TRANSIENT)
        local rc = sq.sqlite3_step(ins)
        assert(rc == SQLITE_DONE, "insert failed: " .. rc)
        sq.sqlite3_reset(ins)
    end
    exec(db, "COMMIT")
    sq.sqlite3_finalize(ins)
    local insert_s = now_seconds() - t0

    local qidx = query_indices()
    local hnsw = {}

    local q0 = now_seconds()
    for _, qi in ipairs(qidx) do
        local stmt = prepare(db, "SELECT rowid FROM bench WHERE bench MATCH '" .. vectors[qi] .. "' LIMIT " .. K)
        local r = {}
        while true do
            local rc = sq.sqlite3_step(stmt)
            if rc == SQLITE_ROW then
                r[#r + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
            elseif rc == SQLITE_DONE then
                break
            else
                local msg = ffi.string(sq.sqlite3_errmsg(db))
                sq.sqlite3_finalize(stmt)
                error("hnsw query failed: " .. msg)
            end
        end
        sq.sqlite3_finalize(stmt)
        hnsw[qi] = r
    end
    local hnsw_s = now_seconds() - q0

    local brute = {}
    local fn = metric_sql_fn(METRIC)
    local q1 = now_seconds()
    for _, qi in ipairs(qidx) do
        local stmt = prepare(db, string.format("SELECT rowid FROM bench ORDER BY %s(vector, '%s') LIMIT %d", fn,
            vectors[qi], K))
        local r = {}
        while true do
            local rc = sq.sqlite3_step(stmt)
            if rc == SQLITE_ROW then
                r[#r + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
            elseif rc == SQLITE_DONE then
                break
            else
                local msg = ffi.string(sq.sqlite3_errmsg(db))
                sq.sqlite3_finalize(stmt)
                error("brute query failed: " .. msg)
            end
        end
        sq.sqlite3_finalize(stmt)
        brute[qi] = r
    end
    local brute_s = now_seconds() - q1

    local recall_sum = 0.0
    for _, qi in ipairs(qidx) do
        local gt = {}
        for _, rid in ipairs(brute[qi]) do
            gt[rid] = true
        end
        local hit = 0
        for _, rid in ipairs(hnsw[qi]) do
            if gt[rid] then
                hit = hit + 1
            end
        end
        recall_sum = recall_sum + (hit / K)
    end

    sq.sqlite3_close(db)

    return {
        insert_s = insert_s,
        hnsw_s = hnsw_s,
        brute_s = brute_s,
        hnsw_qps = #qidx / hnsw_s,
        brute_qps = #qidx / brute_s,
        recall = recall_sum / #qidx,
        q_count = #qidx
    }
end

ensure_output_dir(OUTPUT_DIR)

local header =
    "timestamp,seed,n,dims,q,k,metric,m,ef_construction,ef_search,insert_s,hnsw_s,hnsw_qps,brute_s,brute_qps,mean_recall"

print(string.format("param_sweep: N=%d DIMS=%d Q=%d K=%d metric=%s", N, DIMS, Q, K, METRIC))
print("m\tef_c\tef_s\thnsw_qps\trecall@k")

for _, m in ipairs(M_LIST) do
    for _, efc in ipairs(EFC_LIST) do
        for _, efs in ipairs(EFS_LIST) do
            local r = run_case(m, efc, efs)
            print(string.format("%d\t%d\t%d\t%.1f\t%.4f", m, efc, efs, r.hnsw_qps, r.recall))

            local row = string.format("%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", os.time(), SEED, N,
                DIMS, r.q_count, K, METRIC, m, efc, efs, r.insert_s, r.hnsw_s, r.hnsw_qps, r.brute_s, r.brute_qps,
                r.recall)
            append_csv(OUTPUT_CSV, header, row)
        end
    end
end

print("wrote: " .. OUTPUT_CSV)
