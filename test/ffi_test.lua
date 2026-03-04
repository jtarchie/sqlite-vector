-- test/ffi_test.lua
-- LuaJIT FFI tests for sqlite-vector.
-- These cover ground that SQL scripts cannot: binary float encoding,
-- exact SQLite error codes, column types, and raw BLOB byte values.
--
-- Run from the workspace root:
--   luajit test/ffi_test.lua
local ffi = require "ffi"

-- ── SQLite C API declarations ───────────────────────────────────────────────
ffi.cdef [[
  typedef struct sqlite3      sqlite3;
  typedef struct sqlite3_stmt sqlite3_stmt;

  int    sqlite3_open_v2(const char *filename, sqlite3 **ppDb, int flags, const char *zVfs);
  int    sqlite3_close(sqlite3 *db);
  int    sqlite3_exec(sqlite3 *db, const char *sql, void *cb, void *arg, char **errmsg);
  void   sqlite3_free(void *p);
  int    sqlite3_enable_load_extension(sqlite3 *db, int onoff);
  int    sqlite3_load_extension(sqlite3 *db, const char *file, const char *proc, char **errmsg);

  int    sqlite3_prepare_v2(sqlite3 *db, const char *sql, int nBytes,
                            sqlite3_stmt **ppStmt, const char **pzTail);
  int    sqlite3_step(sqlite3_stmt *stmt);
  int    sqlite3_finalize(sqlite3_stmt *stmt);
  int    sqlite3_reset(sqlite3_stmt *stmt);

  int         sqlite3_column_count(sqlite3_stmt *stmt);
  int         sqlite3_column_type(sqlite3_stmt *stmt, int col);
  const void *sqlite3_column_blob(sqlite3_stmt *stmt, int col);
  int         sqlite3_column_bytes(sqlite3_stmt *stmt, int col);
  double      sqlite3_column_double(sqlite3_stmt *stmt, int col);
  int64_t     sqlite3_column_int64(sqlite3_stmt *stmt, int col);
  const unsigned char *sqlite3_column_text(sqlite3_stmt *stmt, int col);

  int64_t     sqlite3_last_insert_rowid(sqlite3 *db);
  const char *sqlite3_errmsg(sqlite3 *db);
  int         sqlite3_errcode(sqlite3 *db);
  int         sqlite3_extended_errcode(sqlite3 *db);
]]

-- SQLite constants
local SQLITE_OK = 0
local SQLITE_ROW = 100
local SQLITE_DONE = 101
local SQLITE_OPEN_READWRITE = 0x00000002
local SQLITE_OPEN_CREATE = 0x00000004
local SQLITE_OPEN_MEMORY = 0x00000080
local SQLITE_OPEN_URI = 0x00000040
-- Column types
local SQLITE_INTEGER = 1
local SQLITE_FLOAT = 2
local SQLITE_TEXT = 3
local SQLITE_BLOB = 4
local SQLITE_NULL = 5
-- Error codes
local SQLITE_ERROR = 1
local SQLITE_CONSTRAINT = 19

local sq = ffi.load("/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib")

-- ── Helpers ─────────────────────────────────────────────────────────────────

local DYLIB = "build/macosx/arm64/release/libsqlite_vector.dylib"

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
    local flags = SQLITE_OPEN_READWRITE + SQLITE_OPEN_CREATE + SQLITE_OPEN_MEMORY + SQLITE_OPEN_URI
    local dbp = ffi.new("sqlite3*[1]")
    local rc = sq.sqlite3_open_v2(":memory:", dbp, flags, nil)
    assert(rc == SQLITE_OK, "sqlite3_open_v2 failed: " .. rc)
    local db = dbp[0]
    sq.sqlite3_enable_load_extension(db, 1)
    local errmsg = ffi.new("char*[1]")
    rc = sq.sqlite3_load_extension(db, DYLIB, nil, errmsg)
    if rc ~= SQLITE_OK then
        local msg = ffi.string(errmsg[0])
        sq.sqlite3_free(errmsg[0])
        error("load_extension failed: " .. msg)
    end
    return db
end

local function close_db(db)
    sq.sqlite3_close(db)
end

local function exec(db, sql)
    local errmsg = ffi.new("char*[1]")
    local rc = sq.sqlite3_exec(db, sql, nil, nil, errmsg)
    if rc ~= SQLITE_OK then
        local msg = ffi.string(errmsg[0])
        sq.sqlite3_free(errmsg[0])
        error("exec failed (" .. rc .. "): " .. msg .. "\nSQL: " .. sql)
    end
    return rc
end

-- Prepare and step a SELECT, return callback with (stmt) for each row.
local function query(db, sql, fn)
    local stmtp = ffi.new("sqlite3_stmt*[1]")
    local rc = sq.sqlite3_prepare_v2(db, sql, -1, stmtp, nil)
    assert(rc == SQLITE_OK, "prepare failed: " .. ffi.string(sq.sqlite3_errmsg(db)))
    local stmt = stmtp[0]
    while sq.sqlite3_step(stmt) == SQLITE_ROW do
        fn(stmt)
    end
    sq.sqlite3_finalize(stmt)
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

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.rep("-", 72))
if fail == 0 then
    print(("ffi_test: all %d checks passed"):format(pass))
else
    print(("ffi_test: %d passed, %d FAILED"):format(pass, fail))
    os.exit(1)
end
