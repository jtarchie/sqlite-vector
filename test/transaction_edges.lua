-- test/transaction_edges.lua
-- Tests transaction-related edge cases: rollback restoring data, commit
-- making changes durable, and multiple operations within a transaction.
--
-- Run from workspace root:  luajit test/transaction_edges.lua
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
    "── Transaction: rollback restores state ────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=l2)")

    -- Insert a baseline vector
    exec(db, "INSERT INTO v(vector) VALUES('[1.0,0.0,0.0]')")
    local count_before = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count_before == 1, "1 row before transaction")

    -- Begin transaction, insert, then rollback
    exec(db, "BEGIN")
    exec(db, "INSERT INTO v(vector) VALUES('[0.0,1.0,0.0]')")
    exec(db, "INSERT INTO v(vector) VALUES('[0.0,0.0,1.0]')")

    local count_in_txn = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count_in_txn == 3, "3 rows inside transaction")

    exec(db, "ROLLBACK")

    local count_after = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count_after == 1, "rollback restores to 1 row")

    -- kNN still works after rollback
    local knn_count = query_int(db, [[
        SELECT COUNT(*) FROM (
            SELECT rowid FROM v WHERE v MATCH '[1.0,0.0,0.0]' LIMIT 5
        )
    ]])
    check(knn_count == 1, "kNN returns 1 result after rollback")

    close_db(db)
end

print(
    "── Transaction: commit persists data ───────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=2, metric=cosine)")

    exec(db, "BEGIN")
    exec(db, "INSERT INTO v(vector) VALUES('[1.0,0.0]')")
    exec(db, "INSERT INTO v(vector) VALUES('[0.0,1.0]')")
    exec(db, "COMMIT")

    local count = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count == 2, "2 rows after commit")

    -- kNN works after commit
    local nearest = query_int(db, [[
        SELECT rowid FROM v WHERE v MATCH '[1.0,0.0]' LIMIT 1
    ]])
    check(nearest == 1, "kNN finds correct nearest after commit")

    close_db(db)
end

print(
    "── Transaction: rollback after delete ──────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=2, metric=l2)")

    exec(db, "INSERT INTO v(vector) VALUES('[1.0,0.0]')")
    exec(db, "INSERT INTO v(vector) VALUES('[0.0,1.0]')")

    exec(db, "BEGIN")
    exec(db, "DELETE FROM v WHERE rowid=1")
    local count_del = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count_del == 1, "1 row after delete in transaction")

    exec(db, "ROLLBACK")
    local count_restore = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count_restore == 2, "rollback restores deleted row")

    -- Verify both vectors are searchable again
    local knn_count = query_int(db, [[
        SELECT COUNT(*) FROM (
            SELECT rowid FROM v WHERE v MATCH '[1.0,0.0]' LIMIT 5
        )
    ]])
    check(knn_count == 2, "kNN returns both rows after rollback of delete")

    close_db(db)
end

print(
    "── Transaction: rollback after update ──────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=2, metric=l2)")

    exec(db, "INSERT INTO v(vector) VALUES('[1.0,0.0]')")

    -- Verify original vector is nearest to [1,0]
    local orig = query_int(db, "SELECT rowid FROM v WHERE v MATCH '[1.0,0.0]' LIMIT 1")
    check(orig == 1, "rowid 1 is nearest to [1,0] initially")

    exec(db, "BEGIN")
    exec(db, "UPDATE v SET vector = '[0.0,1.0]' WHERE rowid=1")
    exec(db, "ROLLBACK")

    -- After rollback, the original vector should be restored
    local count = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count == 1, "1 row after rollback of update")

    close_db(db)
end

print(
    "── Transaction: multiple inserts then commit ───────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=l2)")

    exec(db, "BEGIN")
    for i = 1, 20 do
        exec(db,
            string.format("INSERT INTO v(vector) VALUES('[%f,%f,%f,%f]')", math.random(), math.random(), math.random(),
                math.random()))
    end
    exec(db, "COMMIT")

    local count = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count == 20, "20 rows after bulk commit")

    -- kNN works
    local knn_count = query_int(db, [[
        SELECT COUNT(*) FROM (
            SELECT rowid FROM v WHERE v MATCH '[0.5,0.5,0.5,0.5]' LIMIT 10
        )
    ]])
    check(knn_count == 10, "kNN returns 10 results from 20-row table")

    close_db(db)
end

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.format("\ntransaction_edges: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
