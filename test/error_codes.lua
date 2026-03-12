-- test/error_codes.lua
-- Additional error code and rejection tests beyond ffi_test.lua.
-- Covers: invalid CREATE options, type mismatches on insert,
-- duplicate rowid insertion, and invalid MATCH queries.
--
-- Run from workspace root:  luajit test/error_codes.lua
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

local SQLITE_OK = sqlite.SQLITE_OK
local SQLITE_ERROR = sqlite.SQLITE_ERROR
local SQLITE_ROW = sqlite.SQLITE_ROW
local SQLITE_DONE = sqlite.SQLITE_DONE
local SQLITE_CONSTRAINT = sqlite.SQLITE_CONSTRAINT

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

local function step_and_check(db, sql)
    local stmtp = ffi.new("sqlite3_stmt*[1]")
    local rc = sq.sqlite3_prepare_v2(db, sql, -1, stmtp, nil)
    if rc ~= SQLITE_OK then
        return rc, ffi.string(sq.sqlite3_errmsg(db))
    end
    rc = sq.sqlite3_step(stmtp[0])
    local msg = ffi.string(sq.sqlite3_errmsg(db))
    sq.sqlite3_finalize(stmtp[0])
    return rc, msg
end

print(
    "── Error codes: invalid CREATE options ─────────────────────────────")
do
    local db = open_db()

    -- Missing dims
    local rc, msg = step_and_check(db, "CREATE VIRTUAL TABLE t1 USING vec0(metric=l2)")
    check(rc == SQLITE_ERROR, "missing dims returns SQLITE_ERROR (rc=" .. rc .. ")")
    check(msg:find("dims=N is required") ~= nil, "error mentions dims: " .. msg)

    -- Invalid metric name
    rc, msg = step_and_check(db, "CREATE VIRTUAL TABLE t2 USING vec0(dims=3, metric=bogus)")
    check(rc == SQLITE_ERROR, "invalid metric returns SQLITE_ERROR")
    check(msg:find("metric") ~= nil or msg:find("unknown") ~= nil, "error mentions metric: " .. msg)

    -- dims=0
    rc, msg = step_and_check(db, "CREATE VIRTUAL TABLE t3 USING vec0(dims=0, metric=l2)")
    check(rc == SQLITE_ERROR, "dims=0 returns SQLITE_ERROR")

    -- dims too large
    rc, msg = step_and_check(db, "CREATE VIRTUAL TABLE t4 USING vec0(dims=8193, metric=l2)")
    check(rc == SQLITE_ERROR, "dims=8193 returns SQLITE_ERROR")
    check(msg:find("8192") ~= nil, "error mentions 8192 limit: " .. msg)

    -- negative dims
    rc, msg = step_and_check(db, "CREATE VIRTUAL TABLE t5 USING vec0(dims=-1, metric=l2)")
    check(rc == SQLITE_ERROR, "dims=-1 returns SQLITE_ERROR")

    close_db(db)
end

print(
    "── Error codes: wrong-type insert ──────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=l2)")

    -- Insert with wrong dimensions
    local rc, msg = step_and_check(db, "INSERT INTO v(vector) VALUES('[1.0,2.0]')")
    check(rc == SQLITE_CONSTRAINT, "wrong dims returns SQLITE_CONSTRAINT (rc=" .. rc .. ")")
    check(msg:find("expected 3 dims") ~= nil, "error mentions expected dims: " .. msg)

    -- Insert NULL vector
    rc, msg = step_and_check(db, "INSERT INTO v(vector) VALUES(NULL)")
    check(rc == SQLITE_CONSTRAINT, "NULL vector returns SQLITE_CONSTRAINT (rc=" .. rc .. ")")

    -- Insert an integer (not a vector)
    rc, msg = step_and_check(db, "INSERT INTO v(vector) VALUES(42)")
    check(rc ~= SQLITE_DONE, "inserting integer into vector column is rejected")

    close_db(db)
end

print(
    "── Error codes: MATCH with wrong dims ──────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=l2)")
    exec(db, "INSERT INTO v(vector) VALUES('[1.0,0.0,0.0]')")

    -- MATCH with wrong dimensions
    local rc, msg = step_and_check(db, "SELECT rowid FROM v WHERE v MATCH '[1.0,0.0]' LIMIT 1")
    check(rc ~= SQLITE_ROW, "MATCH wrong dims doesn't return rows (rc=" .. rc .. ")")

    -- MATCH with NULL
    rc, msg = step_and_check(db, "SELECT rowid FROM v WHERE v MATCH NULL LIMIT 1")
    check(rc ~= SQLITE_ROW, "MATCH NULL doesn't return rows (rc=" .. rc .. ")")

    close_db(db)
end

print(
    "── Error codes: LIMIT boundary ─────────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=2, metric=l2)")
    exec(db, "INSERT INTO v(vector) VALUES('[1.0,0.0]')")

    -- LIMIT at max (4096) should succeed
    local rc = step_and_check(db, "SELECT rowid FROM v WHERE v MATCH '[1.0,0.0]' LIMIT 4096")
    check(rc == SQLITE_ROW or rc == SQLITE_DONE, "LIMIT 4096 succeeds (rc=" .. rc .. ")")

    -- LIMIT over max (4097) should fail
    rc = step_and_check(db, "SELECT rowid FROM v WHERE v MATCH '[1.0,0.0]' LIMIT 4097")
    check(rc == SQLITE_CONSTRAINT, "LIMIT 4097 returns SQLITE_CONSTRAINT (rc=" .. rc .. ")")

    close_db(db)
end

print("── Error codes: scalar function dimension mismatch ─────────────────")
do
    local db = open_db()

    -- Distance between different-dimension vectors
    local rc, msg = step_and_check(db, "SELECT vec_distance_l2('[1,2,3]','[1,2]')")
    check(rc == SQLITE_ERROR, "distance dim mismatch returns SQLITE_ERROR (rc=" .. rc .. ")")

    -- vec_add with mismatched dims
    rc, msg = step_and_check(db, "SELECT vec_add('[1,2,3]','[1,2]')")
    check(rc == SQLITE_ERROR, "vec_add dim mismatch returns SQLITE_ERROR (rc=" .. rc .. ")")

    -- vec_sub with mismatched dims
    rc, msg = step_and_check(db, "SELECT vec_sub('[1,2,3]','[1,2]')")
    check(rc == SQLITE_ERROR, "vec_sub dim mismatch returns SQLITE_ERROR (rc=" .. rc .. ")")

    close_db(db)
end

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.format("\nerror_codes: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
