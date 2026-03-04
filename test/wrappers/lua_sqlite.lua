local ffi = require "ffi"

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
  int    sqlite3_bind_text(sqlite3_stmt *stmt, int idx, const char *val, int n, void *dtor);

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

local M = {}

M.SQLITE_OK = 0
M.SQLITE_ERROR = 1
M.SQLITE_CONSTRAINT = 19
M.SQLITE_MISMATCH = 20
M.SQLITE_ROW = 100
M.SQLITE_DONE = 101

M.SQLITE_OPEN_READWRITE = 0x00000002
M.SQLITE_OPEN_CREATE = 0x00000004
M.SQLITE_OPEN_URI = 0x00000040
M.SQLITE_OPEN_MEMORY = 0x00000080

M.SQLITE_INTEGER = 1
M.SQLITE_FLOAT = 2
M.SQLITE_TEXT = 3
M.SQLITE_BLOB = 4
M.SQLITE_NULL = 5

M.SQLITE_TRANSIENT = ffi.cast("void*", -1)

local function file_exists(path)
    if not path or path == "" then
        return false
    end
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

local function load_library(candidates, desc)
    local errs = {}
    for _, candidate in ipairs(candidates) do
        if candidate and candidate ~= "" then
            local ok, lib = pcall(ffi.load, candidate)
            if ok then
                return lib, candidate
            end
            errs[#errs + 1] = tostring(candidate)
        end
    end
    error("unable to load " .. desc .. "; tried: " .. table.concat(errs, ", "))
end

local function extension_candidates()
    local env = os.getenv("SQLITE_VECTOR_DYLIB")
    local out = {}
    if env and env ~= "" then
        out[#out + 1] = env
    end
    out[#out + 1] = "build/macosx/arm64/release/libsqlite_vector.dylib"
    out[#out + 1] = "build/macosx/x86_64/release/libsqlite_vector.dylib"
    out[#out + 1] = "build/libsqlite_vector.dylib"
    out[#out + 1] = "build/libsqlite_vector.so"
    return out
end

local function resolve_extension_path(explicit_path)
    local candidates = {}
    if explicit_path and explicit_path ~= "" then
        candidates[#candidates + 1] = explicit_path
    end
    for _, candidate in ipairs(extension_candidates()) do
        candidates[#candidates + 1] = candidate
    end
    for _, candidate in ipairs(candidates) do
        if file_exists(candidate) then
            return candidate
        end
    end
    return candidates[1]
end

local sqlite_candidates = {}
do
    local env = os.getenv("SQLITE_DYLIB")
    if env and env ~= "" then
        sqlite_candidates[#sqlite_candidates + 1] = env
    end
    sqlite_candidates[#sqlite_candidates + 1] = "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib"
    sqlite_candidates[#sqlite_candidates + 1] = "/usr/local/opt/sqlite/lib/libsqlite3.dylib"
    sqlite_candidates[#sqlite_candidates + 1] = "libsqlite3.dylib"
    sqlite_candidates[#sqlite_candidates + 1] = "sqlite3"
end

M.ffi = ffi
M.sq, M.sqlite_library = load_library(sqlite_candidates, "SQLite library")

function M.open_db(opts)
    opts = opts or {}
    local filename = opts.filename or ":memory:"
    local memory = opts.memory
    if memory == nil then
        memory = (filename == ":memory:")
    end
    local flags = M.SQLITE_OPEN_READWRITE + M.SQLITE_OPEN_CREATE
    if memory then
        flags = flags + M.SQLITE_OPEN_MEMORY + M.SQLITE_OPEN_URI
    end

    local dbp = ffi.new("sqlite3*[1]")
    local rc = M.sq.sqlite3_open_v2(filename, dbp, flags, nil)
    assert(rc == M.SQLITE_OK, "sqlite3_open_v2 failed: " .. rc)

    local db = dbp[0]
    M.sq.sqlite3_enable_load_extension(db, 1)

    local ext_path = resolve_extension_path(opts.extension_path)
    local errmsg = ffi.new("char*[1]")
    rc = M.sq.sqlite3_load_extension(db, ext_path, nil, errmsg)
    if rc ~= M.SQLITE_OK then
        local msg = errmsg[0] ~= nil and ffi.string(errmsg[0]) or "unknown"
        if errmsg[0] ~= nil then
            M.sq.sqlite3_free(errmsg[0])
        end
        error("load_extension failed: " .. msg .. " (path=" .. ext_path .. ")")
    end
    return db
end

function M.close_db(db)
    return M.sq.sqlite3_close(db)
end

function M.exec(db, sql)
    local errmsg = ffi.new("char*[1]")
    local rc = M.sq.sqlite3_exec(db, sql, nil, nil, errmsg)
    if rc ~= M.SQLITE_OK then
        local msg = errmsg[0] ~= nil and ffi.string(errmsg[0]) or "unknown"
        if errmsg[0] ~= nil then
            M.sq.sqlite3_free(errmsg[0])
        end
        error("exec failed (" .. rc .. "): " .. msg .. "\nSQL: " .. sql)
    end
    return rc
end

function M.prepare(db, sql)
    local stmtp = ffi.new("sqlite3_stmt*[1]")
    local rc = M.sq.sqlite3_prepare_v2(db, sql, -1, stmtp, nil)
    assert(rc == M.SQLITE_OK, "prepare failed: " .. ffi.string(M.sq.sqlite3_errmsg(db)) .. "\nSQL: " .. sql)
    return stmtp[0]
end

function M.query(db, sql, fn)
    local stmt = M.prepare(db, sql)
    while M.sq.sqlite3_step(stmt) == M.SQLITE_ROW do
        fn(stmt)
    end
    M.sq.sqlite3_finalize(stmt)
end

return M
