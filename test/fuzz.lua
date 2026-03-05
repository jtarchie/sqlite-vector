-- test/fuzz.lua
-- Crash-safety fuzz sweep for scalar/vector entrypoints.
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

local SQLITE_OK = sqlite.SQLITE_OK
local SQLITE_ERROR = sqlite.SQLITE_ERROR
local SQLITE_CONSTRAINT = sqlite.SQLITE_CONSTRAINT
local SQLITE_MISMATCH = sqlite.SQLITE_MISMATCH
local SQLITE_ROW = sqlite.SQLITE_ROW
local SQLITE_DONE = sqlite.SQLITE_DONE

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

local function prep_step_finalize(db, sql)
    local stmtp = ffi.new("sqlite3_stmt*[1]")
    local rc = sq.sqlite3_prepare_v2(db, sql, -1, stmtp, nil)
    if rc ~= SQLITE_OK then
        return rc, ffi.string(sq.sqlite3_errmsg(db))
    end

    local step_rc = sq.sqlite3_step(stmtp[0])
    sq.sqlite3_finalize(stmtp[0])
    return step_rc, ffi.string(sq.sqlite3_errmsg(db))
end

print(
    "── Fuzz: crash-safety sweep ─────────────────────────────────────────────")
do
    local db = open_db()

    local setup_rc, setup_msg = prep_step_finalize(db, "CREATE VIRTUAL TABLE v USING vec0(dims=3, metric=l2)")
    check(setup_rc == SQLITE_DONE, "create vtab for fuzz setup (rc=" .. tostring(setup_rc) .. ")")
    if setup_rc ~= SQLITE_DONE then
        io.write("    setup error: " .. tostring(setup_msg) .. "\n")
    end

    local fuzz_sql = {"SELECT vec(NULL)", "SELECT vec('')", "SELECT vec('asdf')", "SELECT vec('[1,2')",
                      "SELECT vec('[,]')", "SELECT vec('[1,,3]')", "SELECT vec(CAST(X'' AS TEXT))",
                      "SELECT vec(CAST(X'00' AS TEXT))", "SELECT vec_dims(NULL)", "SELECT vec_dims('[]')",
                      "SELECT vec_norm(NULL)", "SELECT vec_norm('asdf')", "SELECT vec_distance_l2(NULL, '[1]')",
                      "SELECT vec_distance_l2('[1,2,3]', '[1,2]')", "SELECT vec_distance_cosine('[1,2,3]', 'bad')",
                      "SELECT vec_distance_ip('[1,2,3]', '[1,2]')", "SELECT vec_distance_l1('[1,2,3]', '[1,2]')",
                      "SELECT vec_distance_hamming('[1,2,3]', '[1,2]')",
                      "SELECT vec_distance_jaccard('[1,2,3]', '[1,2]')", "INSERT INTO v(vector) VALUES(NULL)",
                      "INSERT INTO v(vector) VALUES('[1,2]')", "INSERT INTO v(vector) VALUES('[1,2,3]')",
                      "SELECT rowid FROM v WHERE v MATCH '[1,2]' LIMIT 5",
                      "SELECT rowid FROM v WHERE v MATCH 'bad' LIMIT 5",
                      "SELECT rowid FROM v WHERE v MATCH '[1,2,3]' AND ef_search='bad' LIMIT 5",
                      "SELECT rowid FROM v WHERE vec_distance_l2(v, '[1,2,3]') LIMIT 5",
                      "SELECT rowid FROM v WHERE vec_distance_hamming(v, '[1,2,3]') LIMIT 5",
                      "SELECT rowid FROM v WHERE vec_distance_jaccard(v, '[1,2,3]') LIMIT 5"}

    local allowed = {
        [SQLITE_ROW] = true,
        [SQLITE_DONE] = true,
        [SQLITE_ERROR] = true,
        [SQLITE_CONSTRAINT] = true,
        [SQLITE_MISMATCH] = true
    }

    for i, sql in ipairs(fuzz_sql) do
        local rc, msg = prep_step_finalize(db, sql)
        check(allowed[rc] == true, string.format("fuzz[%d] rc %d accepted for: %s", i, rc or -1, sql))
        if not allowed[rc] then
            io.write("    unexpected rc message: " .. tostring(msg) .. "\n")
        end
    end

    close_db(db)
end

print(string.format("\nfuzz: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
print("fuzz: all tests passed")
