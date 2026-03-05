-- test/wal_persist.lua
-- Verifies persistence and kNN correctness across close/reopen in WAL mode.
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

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
    return sqlite.open_db({
        filename = path,
        memory = false
    })
end

local function exec(db, sql)
    sqlite.exec(db, sql)
end

local function query(db, sql, fn)
    sqlite.query(db, sql, fn)
end

local dbpath = os.tmpname() .. "_sv_wal.db"
os.remove(dbpath)
os.remove(dbpath .. "-wal")
os.remove(dbpath .. "-shm")

local first_knn = {}

print(
    "── WAL persist: open + insert ──────────────────────────────────────────")
do
    local db = open_file_db(dbpath)

    local mode = ""
    query(db, "PRAGMA journal_mode=WAL", function(stmt)
        mode = ffi.string(sq.sqlite3_column_text(stmt, 0))
    end)
    check(mode == "wal", "journal_mode is wal")

    exec(db, "CREATE VIRTUAL TABLE vecs USING vec0(dims=3, metric=l2, ef_search=32)")
    exec(db, "BEGIN")
    exec(db, "INSERT INTO vecs(vector) VALUES(vec('[1.0,0.0,0.0]'))")
    exec(db, "INSERT INTO vecs(vector) VALUES(vec('[0.0,1.0,0.0]'))")
    exec(db, "INSERT INTO vecs(vector) VALUES(vec('[0.0,0.0,1.0]'))")
    exec(db, "INSERT INTO vecs(vector) VALUES(vec('[0.9,0.1,0.0]'))")
    exec(db, "COMMIT")

    query(db, "SELECT rowid FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 3", function(stmt)
        first_knn[#first_knn + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
    end)
    check(#first_knn == 3, "cycle-1: kNN returns 3 rows")
    check(first_knn[1] == 1, "cycle-1: nearest is rowid 1")

    local cnt = "0"
    query(db, "SELECT value FROM vecs_config WHERE key='count'", function(stmt)
        cnt = ffi.string(sq.sqlite3_column_text(stmt, 0))
    end)
    check(cnt == "4", "cycle-1: config count is 4")

    sq.sqlite3_close(db)
end

print(
    "── WAL persist: reopen + verify ─────────────────────────────────────────")
do
    local db = open_file_db(dbpath)

    local mode = ""
    query(db, "PRAGMA journal_mode", function(stmt)
        mode = ffi.string(sq.sqlite3_column_text(stmt, 0))
    end)
    check(mode == "wal", "reopen preserves wal journal mode")

    exec(db, "PRAGMA wal_checkpoint(PASSIVE)")
    check(true, "cycle-2: wal_checkpoint(PASSIVE) executed")

    local second_knn = {}
    query(db, "SELECT rowid FROM vecs WHERE vecs MATCH '[1.0,0.0,0.0]' LIMIT 3", function(stmt)
        second_knn[#second_knn + 1] = tonumber(sq.sqlite3_column_int64(stmt, 0))
    end)

    check(#second_knn == #first_knn, "cycle-2: same kNN result count")
    for i = 1, #first_knn do
        check(second_knn[i] == first_knn[i], "cycle-2: result[" .. i .. "] stable across reopen")
    end

    local cnt = "0"
    query(db, "SELECT value FROM vecs_config WHERE key='count'", function(stmt)
        cnt = ffi.string(sq.sqlite3_column_text(stmt, 0))
    end)
    check(cnt == "4", "cycle-2: config count is still 4")

    sq.sqlite3_close(db)
end

os.remove(dbpath)
os.remove(dbpath .. "-wal")
os.remove(dbpath .. "-shm")

print(string.format("\nwal_persist: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
print("wal_persist: all tests passed")
