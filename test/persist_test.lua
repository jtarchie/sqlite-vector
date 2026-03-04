-- test/persist_test.lua
-- Verifies that the extension survives a close/reopen cycle on a file-based
-- database.  The shadow tables (_config, _data, _graph, _layers) must persist
-- so that xConnect can restore the in-memory Vec0Table state and subsequent
-- kNN searches return the same results as before the close.
--
-- Run from the workspace root:
--   luajit test/persist_test.lua
local ffi = require "ffi"

-- ── SQLite C API ─────────────────────────────────────────────────────────────
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

  int64_t              sqlite3_column_int64(sqlite3_stmt *stmt, int col);
  double               sqlite3_column_double(sqlite3_stmt *stmt, int col);
  const unsigned char *sqlite3_column_text(sqlite3_stmt *stmt, int col);

  const char *sqlite3_errmsg(sqlite3 *db);
]]

local SQLITE_OK       = 0
local SQLITE_ROW      = 100
local SQLITE_OPEN_READWRITE = 0x00000002
local SQLITE_OPEN_CREATE    = 0x00000004

local sq = ffi.load("/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib")

local DYLIB = "build/macosx/arm64/release/libsqlite_vector.dylib"

-- ── Helpers ──────────────────────────────────────────────────────────────────
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

local function open_file_db(path)
    local flags = SQLITE_OPEN_READWRITE + SQLITE_OPEN_CREATE
    local dbp = ffi.new("sqlite3*[1]")
    local rc = sq.sqlite3_open_v2(path, dbp, flags, nil)
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

local function exec(db, sql)
    local errmsg = ffi.new("char*[1]")
    local rc = sq.sqlite3_exec(db, sql, nil, nil, errmsg)
    if rc ~= SQLITE_OK then
        local msg = ffi.string(errmsg[0])
        sq.sqlite3_free(errmsg[0])
        error("exec failed (" .. rc .. "): " .. msg .. "\nSQL: " .. sql)
    end
end

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

-- ── Setup: temp file ─────────────────────────────────────────────────────────
local dbpath = os.tmpname() .. "_sv_persist.db"
-- Ensure the file doesn't exist from a previous run
os.remove(dbpath)

print("── Persist: initial open, insert, close ────────────────────────────────")

-- Vectors placed on cardinal / diagonal directions (dims=3, L2 metric).
local vectors = {
    "[1.0, 0.0, 0.0]",   -- rowid 1
    "[0.0, 1.0, 0.0]",   -- rowid 2
    "[0.0, 0.0, 1.0]",   -- rowid 3
    "[0.9, 0.1, 0.0]",   -- rowid 4
    "[0.1, 0.9, 0.0]",   -- rowid 5
    "[0.1, 0.1, 0.8]",   -- rowid 6
    "[0.6, 0.6, 0.0]",   -- rowid 7
    "[0.6, 0.0, 0.6]",   -- rowid 8
    "[0.0, 0.6, 0.6]",   -- rowid 9
    "[0.3, 0.3, 0.3]",   -- rowid 10
}

local first_rowids  = {}  -- kNN results from cycle 1
local first_config  = {}  -- _config snapshot from cycle 1

do
    local db = open_file_db(dbpath)
    exec(db, "CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, ef_search=50)")

    exec(db, "BEGIN")
    for _, v in ipairs(vectors) do
        exec(db, "INSERT INTO vecs(vector) VALUES(vec('" .. v .. "'))")
    end
    exec(db, "COMMIT")

    -- kNN query: nearest 5 to [1,0,0]
    query(db,
        "SELECT rowid FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 5",
        function(stmt)
            first_rowids[#first_rowids + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
        end)

    check(#first_rowids >= 3, "cycle-1: kNN returns at least 3 results")
    check(first_rowids[1] == 1, "cycle-1: nearest to [1,0,0] is rowid 1 (d=0)")

    -- Read _config
    query(db,
        "SELECT key, value FROM vecs_config",
        function(stmt)
            local k = ffi.string(sq.sqlite3_column_text(stmt, 0))
            local v = ffi.string(sq.sqlite3_column_text(stmt, 1))
            first_config[k] = v
        end)

    check(first_config["count"] == "10",    "cycle-1: config count == 10")
    check(first_config["dims"]  == "3",     "cycle-1: config dims == 3")
    check(tonumber(first_config["entry_point"]) >= 1,
          "cycle-1: config entry_point is a valid rowid")

    sq.sqlite3_close(db)
end

-- ── Cycle 2: reopen the same file (xConnect fires, no CREATE) ────────────────
print("── Persist: reopen, verify kNN and config ──────────────────────────────")

do
    local db = open_file_db(dbpath)

    -- kNN with the same query — must return identical rowids in same order.
    local second_rowids = {}
    query(db,
        "SELECT rowid FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 5",
        function(stmt)
            second_rowids[#second_rowids + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
        end)

    check(#second_rowids == #first_rowids,
          "cycle-2: same number of kNN results (" .. #second_rowids .. ")")
    for i = 1, #first_rowids do
        check(second_rowids[i] == first_rowids[i],
              "cycle-2: result[" .. i .. "] rowid matches cycle-1 (" .. (second_rowids[i] or "nil") .. ")")
    end

    -- _config must be unchanged
    local second_config = {}
    query(db, "SELECT key, value FROM vecs_config", function(stmt)
        local k = ffi.string(sq.sqlite3_column_text(stmt, 0))
        local v = ffi.string(sq.sqlite3_column_text(stmt, 1))
        second_config[k] = v
    end)

    for _, k in ipairs({"count", "dims", "metric", "m", "ef_construction",
                         "entry_point", "max_layer"}) do
        check(second_config[k] == first_config[k],
              "cycle-2: config[" .. k .. "] == '" .. (second_config[k] or "nil") .. "'")
    end

    -- A fresh insert still works after reopen
    exec(db, "INSERT INTO vecs(vector) VALUES(vec('[0.5, 0.5, 0.0]'))")
    local new_count = "0"
    query(db, "SELECT value FROM vecs_config WHERE key='count'", function(stmt)
        new_count = ffi.string(sq.sqlite3_column_text(stmt, 0))
    end)
    check(tonumber(new_count) == 11, "cycle-2: insert after reopen increments count to 11")

    sq.sqlite3_close(db)
end

-- ── Cleanup ───────────────────────────────────────────────────────────────────
os.remove(dbpath)

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.format("\npersist_test: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
print("persist_test: all tests passed")
