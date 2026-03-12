-- test/hnsw_params.lua
-- Verifies that different HNSW parameter values (m, ef_construction,
-- ef_search) produce valid graphs and correct kNN results.
--
-- Run from workspace root:  luajit test/hnsw_params.lua
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

-- Test various m values
local m_values = {4, 8, 16, 32}
-- Test various ef_construction values
local ef_constr_values = {10, 50, 200}
-- Test various ef_search values
local ef_search_values = {1, 10, 50}

local N = 30 -- vectors per test
local DIMS = 4

-- Fixed seed for reproducibility
math.randomseed(12345)

print(
    "── HNSW parameter: m sweep ────────────────────────────────────────")
do
    for _, m in ipairs(m_values) do
        math.randomseed(m * 1000 + 7)
        local db = open_db()
        local tname = "vm_" .. m
        exec(db, string.format("CREATE VIRTUAL TABLE %s USING vec0(dims=%d, metric=l2, m=%d)", tname, DIMS, m))

        -- Insert vectors
        for i = 1, N do
            local parts = {}
            for d = 1, DIMS do
                parts[d] = tostring(math.random())
            end
            exec(db, string.format("INSERT INTO %s(vector) VALUES('[%s]')", tname, table.concat(parts, ",")))
        end

        local count = query_int(db, string.format("SELECT COUNT(*) FROM %s_data", tname))
        check(count == N, string.format("m=%d: %d rows inserted", m, count))

        -- Verify stored config
        local stored_m = query_int(db,
            string.format("SELECT CAST(value AS INTEGER) FROM %s_config WHERE key='m'", tname))
        check(stored_m == m, string.format("m=%d: config matches", m))

        -- Max neighbors on layer 0 should be <= 2*m
        local max_nbrs = query_int(db,
            string.format(
                "SELECT MAX(cnt) FROM (SELECT node_id, COUNT(*) AS cnt FROM %s_graph WHERE layer=0 GROUP BY node_id)",
                tname))
        check(max_nbrs ~= nil and max_nbrs <= 2 * m,
            string.format("m=%d: max neighbors on L0 = %s <= %d", m, tostring(max_nbrs), 2 * m))

        -- Max neighbors on layer > 0 should be <= m  (per layer, not total)
        local max_nbrs_upper = query_int(db, string.format(
            "SELECT MAX(cnt) FROM (SELECT node_id, layer, COUNT(*) AS cnt FROM %s_graph WHERE layer>0 GROUP BY node_id, layer)",
            tname))
        if max_nbrs_upper then
            check(max_nbrs_upper <= m, string.format("m=%d: max neighbors on L>0 = %d <= %d", m, max_nbrs_upper, m))
        else
            check(true, string.format("m=%d: no upper layer edges (ok for small N)", m))
        end

        -- kNN works
        local knn = query_int(db, string.format(
            "SELECT COUNT(*) FROM (SELECT rowid FROM %s WHERE %s MATCH '[0.5,0.5,0.5,0.5]' LIMIT 5)", tname, tname))
        check(knn == 5, string.format("m=%d: kNN returns 5 results", m))

        -- No graph orphans
        local orphans = query_int(db,
            string.format("SELECT COUNT(*) FROM %s_graph g WHERE g.node_id NOT IN (SELECT id FROM %s_data) " ..
                              "OR g.neighbor_id NOT IN (SELECT id FROM %s_data)", tname, tname, tname))
        check(orphans == 0, string.format("m=%d: no orphan edges", m))

        close_db(db)
    end
end

print(
    "── HNSW parameter: ef_construction sweep ───────────────────────────")
do
    for _, efc in ipairs(ef_constr_values) do
        local db = open_db()
        local tname = "vec_" .. efc
        exec(db, string.format("CREATE VIRTUAL TABLE %s USING vec0(dims=%d, metric=l2, ef_construction=%d)", tname,
            DIMS, efc))

        for i = 1, N do
            local parts = {}
            for d = 1, DIMS do
                parts[d] = tostring(math.random())
            end
            exec(db, string.format("INSERT INTO %s(vector) VALUES('[%s]')", tname, table.concat(parts, ",")))
        end

        local count = query_int(db, string.format("SELECT COUNT(*) FROM %s_data", tname))
        check(count == N, string.format("efc=%d: %d rows inserted", efc, count))

        local stored_efc = query_int(db, string.format(
            "SELECT CAST(value AS INTEGER) FROM %s_config WHERE key='ef_construction'", tname))
        check(stored_efc == efc, string.format("efc=%d: config matches", efc))

        -- kNN works
        local knn = query_int(db, string.format(
            "SELECT COUNT(*) FROM (SELECT rowid FROM %s WHERE %s MATCH '[0.5,0.5,0.5,0.5]' LIMIT 5)", tname, tname))
        check(knn == 5, string.format("efc=%d: kNN returns 5 results", efc))

        close_db(db)
    end
end

print(
    "── HNSW parameter: ef_search runtime override ──────────────────────")
do
    local db = open_db()
    exec(db, string.format("CREATE VIRTUAL TABLE v USING vec0(dims=%d, metric=l2, ef_search=10)", DIMS))

    for i = 1, N do
        local parts = {}
        for d = 1, DIMS do
            parts[d] = tostring(math.random())
        end
        exec(db, string.format("INSERT INTO v(vector) VALUES('[%s]')", table.concat(parts, ",")))
    end

    for _, efs in ipairs(ef_search_values) do
        local knn = query_int(db,
            string.format(
                "SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[0.5,0.5,0.5,0.5]' AND ef_search=%d LIMIT 5)",
                efs))
        check(knn >= 1, string.format("ef_search=%d: kNN returns >= 1 result (%d)", efs, knn))
    end

    close_db(db)
end

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.format("\nhnsw_params: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
