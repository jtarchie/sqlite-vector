-- test/concurrent_rw.lua
-- Tests concurrent reads and writes using WAL mode with two connections.
-- Verifies no crashes, consistent snapshots, and correct final state.
--
-- Run from workspace root:  luajit test/concurrent_rw.lua
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

local DIMS = 8
local N = 50
local dbpath = "/tmp/sv_concurrent_rw_test.db"

-- Cleanup
os.remove(dbpath)
os.remove(dbpath .. "-wal")
os.remove(dbpath .. "-shm")

math.randomseed(99)

-- ── Open two connections in WAL mode ─────────────────────────────────────
local db1 = sqlite.open_db({
    filename = dbpath,
    memory = false
})
exec(db1, "PRAGMA journal_mode=WAL")

exec(db1, string.format("CREATE VIRTUAL TABLE v USING vec0(dims=%d, metric=l2)", DIMS))

-- Insert initial data on connection 1
exec(db1, "BEGIN")
for i = 1, N do
    local parts = {}
    for d = 1, DIMS do
        parts[d] = tostring(math.random())
    end
    exec(db1, string.format("INSERT INTO v(vector) VALUES('[%s]')", table.concat(parts, ",")))
end
exec(db1, "COMMIT")

local count1 = query_int(db1, "SELECT COUNT(*) FROM v_data")
check(count1 == N, string.format("conn1: %d rows inserted", count1))

-- Open connection 2 (reader)
local db2 = sqlite.open_db({
    filename = dbpath,
    memory = false
})
exec(db2, "PRAGMA journal_mode=WAL")

-- Connection 2 sees the committed data
local count2 = query_int(db2, "SELECT COUNT(*) FROM v_data")
check(count2 == N, string.format("conn2 sees %d committed rows", count2))

-- ── Connection 1 writes while connection 2 reads ─────────────────────────
-- Start a write transaction on conn1
exec(db1, "BEGIN")
for i = 1, N do
    local parts = {}
    for d = 1, DIMS do
        parts[d] = tostring(math.random())
    end
    exec(db1, string.format("INSERT INTO v(vector) VALUES('[%s]')", table.concat(parts, ",")))
end

-- Connection 2 should still see the old snapshot (WAL isolation)
count2 = query_int(db2, "SELECT COUNT(*) FROM v_data")
check(count2 == N, string.format("conn2 sees %d rows during conn1 write (snapshot isolation)", count2))

-- Connection 2 can run kNN queries during write
local knn_count = query_int(db2,
    string.format("SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[%s]' LIMIT 5)",
        table.concat({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, ",")))
check(knn_count >= 1, string.format("conn2 kNN works during write (%d results)", knn_count))

-- Commit connection 1's writes
exec(db1, "COMMIT")

-- Connection 1 sees all data
count1 = query_int(db1, "SELECT COUNT(*) FROM v_data")
check(count1 == 2 * N, string.format("conn1 sees %d rows after commit", count1))

-- ── Both connections can query after all writes ──────────────────────────
local qvec = table.concat({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, ",")

local knn1 = query_int(db1,
    string.format("SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[%s]' LIMIT 10)", qvec))
check(knn1 == 10, string.format("conn1 kNN returns 10 results"))

local knn2 = query_int(db2,
    string.format("SELECT COUNT(*) FROM (SELECT rowid FROM v WHERE v MATCH '[%s]' LIMIT 10)", qvec))
check(knn2 == 10, string.format("conn2 kNN returns 10 results"))

-- No orphan edges
local orphans = query_int(db1, [[
    SELECT COUNT(*) FROM v_graph
    WHERE node_id NOT IN (SELECT id FROM v_data)
       OR neighbor_id NOT IN (SELECT id FROM v_data)
]])
check(orphans == 0, "no orphan edges after concurrent operations")

-- ── Cleanup ──────────────────────────────────────────────────────────────
sqlite.close_db(db2)
sqlite.close_db(db1)
os.remove(dbpath)
os.remove(dbpath .. "-wal")
os.remove(dbpath .. "-shm")

print(string.format("\nconcurrent_rw: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
