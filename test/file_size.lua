-- test/file_size.lua
-- Validates that database file sizes scale correctly with vector type:
--   float32 > int8 > bit (for data portion)
-- Uses on-disk databases with VACUUM to get accurate sizes.
--
-- Run from workspace root:  luajit test/file_size.lua
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

local DIMS = 64
local N = 200

local function file_size(path)
    local f = io.open(path, "rb")
    if not f then
        return 0
    end
    local size = f:seek("end")
    f:close()
    return size
end

-- ── Create databases for each type ───────────────────────────────────────
math.randomseed(42)

local configs = {{
    name = "f32",
    type_opt = "",
    metric = "l2",
    suffix = "f32"
}, {
    name = "i8",
    type_opt = "type=int8",
    metric = "l2",
    suffix = "i8"
}, {
    name = "bit",
    type_opt = "type=bit",
    metric = "hamming",
    suffix = "bit"
}}

local sizes = {}

for _, cfg in ipairs(configs) do
    local dbpath = "/tmp/sv_filesize_" .. cfg.suffix .. ".db"
    os.remove(dbpath)
    os.remove(dbpath .. "-wal")
    os.remove(dbpath .. "-shm")

    local db = sqlite.open_db({
        filename = dbpath,
        memory = false
    })

    local create_sql
    if cfg.type_opt ~= "" then
        create_sql = string.format("CREATE VIRTUAL TABLE v USING vec0(dims=%d, metric=%s, %s)", DIMS, cfg.metric,
            cfg.type_opt)
    else
        create_sql = string.format("CREATE VIRTUAL TABLE v USING vec0(dims=%d, metric=%s)", DIMS, cfg.metric)
    end
    exec(db, create_sql)

    exec(db, "BEGIN")
    for i = 1, N do
        if cfg.suffix == "bit" then
            -- bit vectors: use 0/1 values
            local parts = {}
            for d = 1, DIMS do
                parts[d] = tostring(math.random(0, 1))
            end
            exec(db, string.format("INSERT INTO v(vector) VALUES('[%s]')", table.concat(parts, ",")))
        elseif cfg.suffix == "i8" then
            -- int8 vectors: use integer values in [-128, 127]
            local parts = {}
            for d = 1, DIMS do
                parts[d] = tostring(math.random(-128, 127))
            end
            exec(db, string.format("INSERT INTO v(vector) VALUES('[%s]')", table.concat(parts, ",")))
        else
            -- float32 vectors: use random floats
            local parts = {}
            for d = 1, DIMS do
                parts[d] = tostring(math.random())
            end
            exec(db, string.format("INSERT INTO v(vector) VALUES('[%s]')", table.concat(parts, ",")))
        end
    end
    exec(db, "COMMIT")

    -- Verify count
    local count = query_int(db, "SELECT COUNT(*) FROM v_data")
    check(count == N, string.format("%s: %d vectors inserted", cfg.name, count))

    sqlite.close_db(db)

    local sz = file_size(dbpath)
    sizes[cfg.suffix] = sz

    -- Cleanup
    os.remove(dbpath)
    os.remove(dbpath .. "-wal")
    os.remove(dbpath .. "-shm")
end

-- ── Verify relative ordering: float32 > int8 > bit ──────────────────────
check(sizes.f32 > sizes.i8, string.format("float32 (%d) > int8 (%d)", sizes.f32, sizes.i8))
check(sizes.i8 > sizes.bit, string.format("int8 (%d) > bit (%d)", sizes.i8, sizes.bit))
check(sizes.f32 > sizes.bit, string.format("float32 (%d) > bit (%d)", sizes.f32, sizes.bit))

-- ── Summary ──────────────────────────────────────────────────────────────
print(string.format("\nfile_size: %d passed, %d failed", pass, fail))
print(string.format("  sizes: f32=%d  i8=%d  bit=%d", sizes.f32, sizes.i8, sizes.bit))
if fail > 0 then
    os.exit(1)
end
