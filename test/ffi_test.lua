-- test/ffi_test.lua
-- LuaJIT FFI tests for sqlite-vector.
-- These cover ground that SQL scripts cannot: binary float encoding,
-- exact SQLite error codes, column types, and raw BLOB byte values.
--
-- Run from the workspace root:
--   luajit test/ffi_test.lua
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

local SQLITE_OK = sqlite.SQLITE_OK
local SQLITE_ERROR = sqlite.SQLITE_ERROR
local SQLITE_DONE = sqlite.SQLITE_DONE
local SQLITE_INTEGER = sqlite.SQLITE_INTEGER
local SQLITE_FLOAT = sqlite.SQLITE_FLOAT
local SQLITE_TEXT = sqlite.SQLITE_TEXT
local SQLITE_BLOB = sqlite.SQLITE_BLOB
local SQLITE_NULL = sqlite.SQLITE_NULL
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

-- Open an in-memory SQLite database with the extension loaded.
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

-- Prepare and step a SELECT, return callback with (stmt) for each row.
local function query(db, sql, fn)
    sqlite.query(db, sql, fn)
end

-- ── Test suites ──────────────────────────────────────────────────────────────

print(
    "── FFI: binary float encoding ──────────────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=l2)")
    -- Insert a vector with exact known float values
    exec(db, "INSERT INTO v(vector) VALUES('[1.0, 2.0, 3.0]')")

    query(db, "SELECT vector FROM v_data WHERE id=1", function(stmt)
        -- Column type must be BLOB
        check(sq.sqlite3_column_type(stmt, 0) == SQLITE_BLOB, "stored vector column type is BLOB")

        -- Length must be 3 * 4 = 12 bytes
        local nbytes = sq.sqlite3_column_bytes(stmt, 0)
        check(nbytes == 12, "BLOB length is 12 bytes for dims=3")

        -- Decode the three float32 values using FFI cast
        local blob = sq.sqlite3_column_blob(stmt, 0)
        local floats = ffi.cast("const float*", blob)
        check(math.abs(floats[0] - 1.0) < 1e-6, "float[0] == 1.0")
        check(math.abs(floats[1] - 2.0) < 1e-6, "float[1] == 2.0")
        check(math.abs(floats[2] - 3.0) < 1e-6, "float[2] == 3.0")
    end)

    -- Negative values
    exec(db, "INSERT INTO v(vector) VALUES('[-0.5, 0.0, 128.0]')")
    query(db, "SELECT vector FROM v_data WHERE id=2", function(stmt)
        local blob = sq.sqlite3_column_blob(stmt, 0)
        local floats = ffi.cast("const float*", blob)
        check(math.abs(floats[0] - (-0.5)) < 1e-6, "float[0] == -0.5")
        check(math.abs(floats[1]) < 1e-6, "float[1] == 0.0")
        check(math.abs(floats[2] - 128.0) < 1e-6, "float[2] == 128.0")
    end)

    close_db(db)
end

print(
    "── FFI: error codes ────────────────────────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=cosine)")

    -- Wrong dims: SQLITE_CONSTRAINT (19)
    local stmtp = ffi.new("sqlite3_stmt*[1]")
    sq.sqlite3_prepare_v2(db, "INSERT INTO v(vector) VALUES('[1.0,2.0,3.0]')", -1, stmtp, nil)
    local rc = sq.sqlite3_step(stmtp[0])
    sq.sqlite3_finalize(stmtp[0])
    check(rc == SQLITE_CONSTRAINT, "wrong-dims INSERT returns SQLITE_CONSTRAINT (" .. rc .. ")")

    local errcode = sq.sqlite3_errcode(db)
    check(errcode == SQLITE_CONSTRAINT, "sqlite3_errcode() == SQLITE_CONSTRAINT after bad dims")

    local errmsg = ffi.string(sq.sqlite3_errmsg(db))
    check(errmsg:find("expected 4 dims") ~= nil, "error message mentions expected dims: " .. errmsg)

    -- NULL vector: SQLITE_CONSTRAINT
    sq.sqlite3_prepare_v2(db, "INSERT INTO v(vector) VALUES(NULL)", -1, stmtp, nil)
    rc = sq.sqlite3_step(stmtp[0])
    sq.sqlite3_finalize(stmtp[0])
    check(rc == SQLITE_CONSTRAINT, "NULL vector INSERT returns SQLITE_CONSTRAINT")

    -- DELETE: xFilter is still a stub that returns 0 rows, so the WHERE
    -- clause never matches and xUpdate(argc=1) is never called — rc is DONE.
    -- Once kNN search is wired up this will change to SQLITE_CONSTRAINT.
    exec(db, "INSERT INTO v(vector) VALUES('[0.1,0.2,0.3,0.4]')")
    sq.sqlite3_prepare_v2(db, "DELETE FROM v WHERE rowid=1", -1, stmtp, nil)
    rc = sq.sqlite3_step(stmtp[0])
    sq.sqlite3_finalize(stmtp[0])
    -- SQLITE_DONE (101): no rows found by empty xFilter → xUpdate never called
    check(rc == SQLITE_DONE, "DELETE returns SQLITE_DONE while xFilter is a stub (rc=" .. rc .. ")")

    close_db(db)
end

print(
    "── FFI: rowid assignment ───────────────────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=2, metric=cosine)")

    local rowids = {}
    for i = 1, 5 do
        exec(db, ("INSERT INTO v(vector) VALUES('[%d.0, %d.0]')"):format(i, i + 1))
        rowids[i] = tonumber(sq.sqlite3_last_insert_rowid(db))
    end

    -- Rowids must be strictly increasing
    local monotonic = true
    for i = 2, 5 do
        if rowids[i] <= rowids[i - 1] then
            monotonic = false
        end
    end
    check(monotonic, "rowids are strictly increasing across 5 inserts")
    check(rowids[1] == 1, "first insert gets rowid 1")
    check(rowids[5] == 5, "fifth insert gets rowid 5")

    -- Verify rowids match what's in _data
    local count = 0
    query(db, "SELECT id FROM v_data ORDER BY id", function(stmt)
        count = count + 1
        local id = tonumber(sq.sqlite3_column_int64(stmt, 0))
        check(id == count, ("v_data row %d has id %d"):format(count, id))
    end)
    check(count == 5, "v_data has exactly 5 rows")

    close_db(db)
end

print(
    "── FFI: scalar function return types ───────────────────────────────────")
do
    local db = open_db()

    -- vec() must return TEXT
    query(db, "SELECT vec('[1.0,2.0,3.0]')", function(stmt)
        check(sq.sqlite3_column_type(stmt, 0) == SQLITE_TEXT, "vec() returns SQLITE_TEXT")
        local s = ffi.string(sq.sqlite3_column_text(stmt, 0))
        -- vec_format uses %.7g; 1.0→"1", 2.0→"2", 3.0→"3"
        check(s:find("1") and s:find("2") and s:find("3"), "vec() round-trips floats, got: " .. s)
    end)

    -- vec_dims() must return INTEGER
    query(db, "SELECT vec_dims('[1.0,2.0,3.0]')", function(stmt)
        check(sq.sqlite3_column_type(stmt, 0) == SQLITE_INTEGER, "vec_dims() returns SQLITE_INTEGER")
        check(sq.sqlite3_column_int64(stmt, 0) == 3, "vec_dims('[1,2,3]') == 3")
    end)

    -- vec_norm() must return FLOAT; 3-4-5 triangle → norm=5
    query(db, "SELECT vec_norm('[3.0,4.0]')", function(stmt)
        check(sq.sqlite3_column_type(stmt, 0) == SQLITE_FLOAT, "vec_norm() returns SQLITE_FLOAT")
        local v = sq.sqlite3_column_double(stmt, 0)
        check(math.abs(v - 5.0) < 1e-5, ("vec_norm('[3,4]') == 5.0, got %.6f"):format(v))
    end)

    -- vec_distance_l2() must return FLOAT
    query(db, "SELECT vec_distance_l2('[3,0]','[0,4]')", function(stmt)
        check(sq.sqlite3_column_type(stmt, 0) == SQLITE_FLOAT, "vec_distance_l2() returns SQLITE_FLOAT")
        local v = sq.sqlite3_column_double(stmt, 0)
        check(math.abs(v - 5.0) < 1e-5, ("vec_distance_l2 3-4-5 == 5.0, got %.6f"):format(v))
    end)

    -- NULL propagation: vec(NULL) → NULL
    query(db, "SELECT vec(NULL)", function(stmt)
        check(sq.sqlite3_column_type(stmt, 0) == SQLITE_NULL, "vec(NULL) returns SQLITE_NULL")
    end)

    close_db(db)
end

print(
    "── FFI: vec0 MATCH query column types ──────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=cosine)")
    exec(db, "INSERT INTO v(vector) VALUES('[1.0,0.0,0.0]')")
    exec(db, "INSERT INTO v(vector) VALUES('[0.0,1.0,0.0]')")
    exec(db, "INSERT INTO v(vector) VALUES('[0.0,0.0,1.0]')")

    -- A MATCH query currently returns 0 rows (kNN not yet implemented),
    -- so we verify the schema / column count via prepare only.
    local stmtp = ffi.new("sqlite3_stmt*[1]")
    local rc = sq.sqlite3_prepare_v2(db, "SELECT * FROM v WHERE v MATCH '[1.0,0.0,0.0]' LIMIT 3", -1, stmtp, nil)
    check(rc == SQLITE_OK, "MATCH query prepares successfully")
    local ncols = sq.sqlite3_column_count(stmtp[0])
    -- Col 0: hidden, col 1: vector TEXT, col 2: distance REAL HIDDEN
    -- SELECT * exposes all non-hidden columns → just 'vector' (col 1)
    check(ncols >= 1, "MATCH query has at least 1 output column (got " .. ncols .. ")")
    sq.sqlite3_finalize(stmtp[0])

    -- Full-scan (no MATCH): also verify it prepares
    rc = sq.sqlite3_prepare_v2(db, "SELECT vector FROM v", -1, stmtp, nil)
    check(rc == SQLITE_OK, "full-scan SELECT prepares successfully")
    sq.sqlite3_finalize(stmtp[0])

    close_db(db)
end

print(
    "── FFI: scalar error paths ───────────────────────────────────────────────")
do
    local db = open_db()
    local stmtp = ffi.new("sqlite3_stmt*[1]")

    -- malformed vector literal should surface SQLITE_ERROR
    sq.sqlite3_prepare_v2(db, "SELECT vec('[1,2')", -1, stmtp, nil)
    local rc = sq.sqlite3_step(stmtp[0])
    sq.sqlite3_finalize(stmtp[0])
    check(rc == SQLITE_ERROR, "vec() malformed literal returns SQLITE_ERROR")
    local msg = ffi.string(sq.sqlite3_errmsg(db))
    check(msg:find("invalid") ~= nil or msg:find("vector") ~= nil,
        "vec() malformed literal sets helpful error message: " .. msg)

    -- distance mismatch should also be a SQL error
    sq.sqlite3_prepare_v2(db, "SELECT vec_distance_l2('[1,2,3]', '[1,2]')", -1, stmtp, nil)
    rc = sq.sqlite3_step(stmtp[0])
    sq.sqlite3_finalize(stmtp[0])
    check(rc == SQLITE_ERROR, "vec_distance_l2 dimension mismatch returns SQLITE_ERROR")
    msg = ffi.string(sq.sqlite3_errmsg(db))
    check(msg:find("dimension mismatch") ~= nil, "distance mismatch error message mentions dimensions: " .. msg)

    close_db(db)
end

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.rep("-", 72))
if fail == 0 then
    print(("ffi_test: all %d checks passed"):format(pass))
else
    print(("ffi_test: %d passed, %d FAILED"):format(pass, fail))
    os.exit(1)
end
