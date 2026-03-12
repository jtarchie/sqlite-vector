-- test/graph_integrity.lua
-- Validates HNSW graph invariants: every graph edge refers to a node
-- that actually exists in the data table, neighbor counts stay within
-- the m parameter bound, and delete+repair preserves connectivity.
--
-- Run from workspace root:  luajit test/graph_integrity.lua
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq
local ffi = sqlite.ffi

local pass, fail = 0, 0
local function check(cond, msg)
    if cond then
        pass = pass + 1
        io.write("  PASS  " .. msg .. "\n")
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

print(
    "── Graph integrity: basic invariants ───────────────────────────────")
do
    local db = open_db()
    local M_PARAM = 16
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=l2, m=" .. M_PARAM .. ")")

    -- Insert enough vectors to build a non-trivial graph
    for i = 1, 50 do
        local vals = string.format("[%f,%f,%f,%f]", math.random(), math.random(), math.random(), math.random())
        exec(db, "INSERT INTO v(vector) VALUES('" .. vals .. "')")
    end

    local node_count = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(node_count == 50, "50 nodes in data table")

    -- Invariant 1: every node_id in graph exists in data
    local orphan_nodes = query_int(db, "SELECT COUNT(*) FROM v_graph g WHERE g.node_id NOT IN (SELECT id FROM v_data)")
    check(orphan_nodes == 0, "no orphan node_ids in graph")

    -- Invariant 2: every neighbor_id in graph exists in data
    local orphan_neighbors = query_int(db,
        "SELECT COUNT(*) FROM v_graph g WHERE g.neighbor_id NOT IN (SELECT id FROM v_data)")
    check(orphan_neighbors == 0, "no orphan neighbor_ids in graph")

    -- Invariant 3: no self-edges
    local self_edges = query_int(db, "SELECT COUNT(*) FROM v_graph WHERE node_id = neighbor_id")
    check(self_edges == 0, "no self-edges in graph")

    -- Invariant 4: neighbor count per node on layer 0 is <= 2*m
    local max_neighbors = query_int(db,
        "SELECT MAX(cnt) FROM (SELECT node_id, COUNT(*) AS cnt FROM v_graph WHERE layer=0 GROUP BY node_id)")
    check(max_neighbors ~= nil and max_neighbors <= 2 * M_PARAM,
        "max neighbors on layer 0 <= 2*m (" .. tostring(max_neighbors) .. " <= " .. (2 * M_PARAM) .. ")")

    -- Invariant 5: all distances in graph are non-negative
    local neg_distances = query_int(db, "SELECT COUNT(*) FROM v_graph WHERE distance < 0")
    check(neg_distances == 0, "all graph distances >= 0")

    -- Invariant 6: every node in v_data has a layer entry in v_layers
    local missing_layers = query_int(db,
        "SELECT COUNT(*) FROM v_data d WHERE d.id NOT IN (SELECT node_id FROM v_layers)")
    check(missing_layers == 0, "every data node has a layer entry")

    -- Invariant 7: entry_point in config exists in data
    local entry_point = query_int(db, "SELECT CAST(value AS INTEGER) FROM v_config WHERE key='entry_point'")
    local ep_exists = query_int(db, "SELECT COUNT(*) FROM v_data WHERE id=" .. entry_point)
    check(ep_exists == 1, "entry_point " .. entry_point .. " exists in data")

    close_db(db)
end

print(
    "── Graph integrity: after deletes ──────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=cosine, m=8)")

    -- Insert 30 vectors
    for i = 1, 30 do
        local vals = string.format("[%f,%f,%f]", math.random(), math.random(), math.random())
        exec(db, "INSERT INTO v(vector) VALUES('" .. vals .. "')")
    end

    -- Delete every other vector
    for i = 2, 30, 2 do
        exec(db, "DELETE FROM v WHERE rowid=" .. i)
    end

    local remaining = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(remaining == 15, "15 nodes remain after delete")

    -- Invariant: no orphan edges after delete
    local orphan_nodes = query_int(db, "SELECT COUNT(*) FROM v_graph g WHERE g.node_id NOT IN (SELECT id FROM v_data)")
    check(orphan_nodes == 0, "no orphan node_ids after delete")

    local orphan_neighbors = query_int(db,
        "SELECT COUNT(*) FROM v_graph g WHERE g.neighbor_id NOT IN (SELECT id FROM v_data)")
    check(orphan_neighbors == 0, "no orphan neighbor_ids after delete")

    -- Invariant: no self-edges after delete
    local self_edges = query_int(db, "SELECT COUNT(*) FROM v_graph WHERE node_id = neighbor_id")
    check(self_edges == 0, "no self-edges after delete")

    -- kNN still works on the surviving nodes
    local knn_count = query_int(db, [[
        SELECT COUNT(*) FROM (
            SELECT rowid FROM v WHERE v MATCH '[1.0,0.0,0.0]' LIMIT 5
        )
    ]])
    check(knn_count >= 1 and knn_count <= 15, "kNN returns results after deletes")

    -- entry_point still valid
    local ep = query_int(db, "SELECT CAST(value AS INTEGER) FROM v_config WHERE key='entry_point'")
    local ep_ok = query_int(db, "SELECT COUNT(*) FROM v_data WHERE id=" .. ep)
    check(ep_ok == 1, "entry_point valid after deletes")

    close_db(db)
end

print(
    "── Graph integrity: insert after delete ────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=2, metric=l2, m=4)")

    -- Build, delete, rebuild cycle
    for i = 1, 20 do
        exec(db, string.format("INSERT INTO v(vector) VALUES('[%f,%f]')", math.random(), math.random()))
    end
    for i = 1, 10 do
        exec(db, "DELETE FROM v WHERE rowid=" .. i)
    end
    for i = 1, 10 do
        exec(db, string.format("INSERT INTO v(vector) VALUES('[%f,%f]')", math.random(), math.random()))
    end

    local total = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(total == 20, "20 nodes after delete+reinsert cycle")

    -- All graph invariants hold
    local orphans = query_int(db, "SELECT COUNT(*) FROM v_graph g WHERE g.node_id NOT IN (SELECT id FROM v_data) " ..
        "OR g.neighbor_id NOT IN (SELECT id FROM v_data)")
    check(orphans == 0, "no orphans after delete+reinsert")

    local self_edges = query_int(db, "SELECT COUNT(*) FROM v_graph WHERE node_id = neighbor_id")
    check(self_edges == 0, "no self-edges after delete+reinsert")

    close_db(db)
end

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.format("\ngraph_integrity: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
