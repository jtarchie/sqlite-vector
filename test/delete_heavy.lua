-- test/delete_heavy.lua
-- Tests that recall remains acceptable after heavy delete+reinsert churn.
-- Insert N vectors → delete 50% → reinsert N/2 → verify recall.
--
-- Run from workspace root:  luajit test/delete_heavy.lua
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

local pass, fail = 0, 0
local function check(cond, msg)
    if cond then
        pass = pass + 1
    else
        fail = fail + 1
        io.write("  FAIL  " .. msg .. "\n")
    end
end

local N = 500
local DIMS = 32
local K = 10
local Q = 20 -- number of test queries

local function open_db()
    return sqlite.open_db({
        memory = true
    })
end
local function close_db(db)
    sqlite.close_db(db)
end
local function exec(db, sql)
    return sqlite.exec(db, sql)
end

local function query_int(db, sql)
    local result
    sqlite.query(db, sql, function(stmt)
        result = tonumber(sq.sqlite3_column_int64(stmt, 0))
    end)
    return result
end

math.randomseed(42)

local db = open_db()
exec(db, string.format("CREATE VIRTUAL TABLE v USING vec0(dims=%d, metric=l2, ef_construction=100, ef_search=32)", DIMS))

-- ── Phase 1: Insert N vectors ────────────────────────────────────────────
local vectors = {}
exec(db, "CREATE TABLE vecs_txt(id INTEGER PRIMARY KEY, vec TEXT)")
exec(db, "BEGIN")
for i = 1, N do
    local parts = {}
    for d = 1, DIMS do
        parts[d] = tostring(math.random())
    end
    vectors[i] = parts
    local vtxt = "[" .. table.concat(parts, ",") .. "]"
    exec(db, string.format("INSERT INTO v(vector) VALUES('%s')", vtxt))
    exec(db, string.format("INSERT INTO vecs_txt(id, vec) VALUES(%d, '%s')", i, vtxt))
end
exec(db, "COMMIT")

local count = query_int(db, "SELECT COUNT(*) FROM v_data")
check(count == N, string.format("initial insert: %d vectors", count))

-- No orphan edges after initial insert
local orphans = query_int(db, [[
    SELECT COUNT(*) FROM v_graph
    WHERE node_id NOT IN (SELECT id FROM v_data)
       OR neighbor_id NOT IN (SELECT id FROM v_data)
]])
check(orphans == 0, "no orphan edges after initial insert")

-- ── Phase 2: Delete 50% randomly ────────────────────────────────────────
local delete_ids = {}
for i = 1, N do
    delete_ids[i] = i
end
-- Fisher-Yates shuffle
for i = N, 2, -1 do
    local j = math.random(i)
    delete_ids[i], delete_ids[j] = delete_ids[j], delete_ids[i]
end

exec(db, "BEGIN")
for i = 1, N / 2 do
    exec(db, string.format("DELETE FROM v WHERE rowid = %d", delete_ids[i]))
    exec(db, string.format("DELETE FROM vecs_txt WHERE id = %d", delete_ids[i]))
end
exec(db, "COMMIT")

count = query_int(db, "SELECT COUNT(*) FROM v_data")
check(count == N / 2, string.format("after delete: %d vectors remain", count))

-- No orphan edges after deletion
orphans = query_int(db, [[
    SELECT COUNT(*) FROM v_graph
    WHERE node_id NOT IN (SELECT id FROM v_data)
       OR neighbor_id NOT IN (SELECT id FROM v_data)
]])
check(orphans == 0, "no orphan edges after delete")

-- ── Phase 3: Reinsert N/2 new vectors ───────────────────────────────────
exec(db, "BEGIN")
for i = 1, N / 2 do
    local parts = {}
    for d = 1, DIMS do
        parts[d] = tostring(math.random())
    end
    local vtxt = "[" .. table.concat(parts, ",") .. "]"
    exec(db, string.format("INSERT INTO v(vector) VALUES('%s')", vtxt))
    -- New rowids start at N+1
    exec(db, string.format("INSERT INTO vecs_txt(id, vec) VALUES(%d, '%s')", N + i, vtxt))
end
exec(db, "COMMIT")

count = query_int(db, "SELECT COUNT(*) FROM v_data")
check(count == N, string.format("after reinsert: %d vectors total", count))

-- No orphan edges after reinsert
orphans = query_int(db, [[
    SELECT COUNT(*) FROM v_graph
    WHERE node_id NOT IN (SELECT id FROM v_data)
       OR neighbor_id NOT IN (SELECT id FROM v_data)
]])
check(orphans == 0, "no orphan edges after reinsert")

-- ── Phase 4: Verify recall ──────────────────────────────────────────────
-- Get current data IDs
local data_ids = {}
sqlite.query(db, "SELECT id FROM v_data ORDER BY id", function(stmt)
    data_ids[#data_ids + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
end)

-- Build brute-force and HNSW results for Q random queries
local total_recall = 0
for q = 1, Q do
    local parts = {}
    for d = 1, DIMS do
        parts[d] = tostring(math.random())
    end
    local qvec = "[" .. table.concat(parts, ",") .. "]"

    -- Brute-force K nearest via text vectors table
    local bf_sql = string.format("SELECT id FROM vecs_txt ORDER BY vec_distance_l2(vec, '%s') ASC LIMIT %d", qvec, K)
    local bf_ids = {}
    sqlite.query(db, bf_sql, function(stmt)
        bf_ids[#bf_ids + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
    end)

    -- HNSW kNN
    local hnsw_sql = string.format("SELECT rowid FROM v WHERE v MATCH '%s' LIMIT %d", qvec, K)
    local hnsw_set = {}
    sqlite.query(db, hnsw_sql, function(stmt)
        hnsw_set[tonumber(sq.sqlite3_column_int64(stmt, 0))] = true
    end)

    -- Compute recall: fraction of brute-force results found in HNSW results
    local hits = 0
    for _, id in ipairs(bf_ids) do
        if hnsw_set[id] then
            hits = hits + 1
        end
    end
    total_recall = total_recall + hits / K
end

local mean_recall = total_recall / Q
check(mean_recall >= 0.50, string.format("recall@%d after churn = %.4f >= 0.50", K, mean_recall))

close_db(db)

-- ── Summary ──────────────────────────────────────────────────────────────
print(string.format("\ndelete_heavy: %d passed, %d failed (recall=%.4f)", pass, fail, mean_recall))
if fail > 0 then
    os.exit(1)
end
