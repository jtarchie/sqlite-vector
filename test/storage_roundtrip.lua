-- test/storage_roundtrip.lua
-- Validates BLOB storage sizes and data round-trip fidelity for all types.
-- For float32: exact float32 byte encoding is verified.
-- For int8: values are verified within clamping range.
-- For bit: bit-packing MSB-first is verified.
--
-- Run from workspace root:  luajit test/storage_roundtrip.lua
local ffi = require "ffi"
local sqlite = require "test.wrappers.lua_sqlite"
local sq = sqlite.sq

local SQLITE_ROW = sqlite.SQLITE_ROW
local SQLITE_BLOB = sqlite.SQLITE_BLOB

local pass, fail = 0, 0
local function check(cond, msg)
    if cond then
        pass = pass + 1
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

print(
    "── Storage: float32 BLOB encoding ─────────────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=l2, type=float32)")
    exec(db, "INSERT INTO v(vector) VALUES('[1.5, -2.5, 0.0, 100.0]')")

    local stmt = sqlite.prepare(db, "SELECT vector FROM v_data WHERE id=1")
    local rc = sq.sqlite3_step(stmt)
    check(rc == SQLITE_ROW, "f32: query returns a row")

    local col_type = sq.sqlite3_column_type(stmt, 0)
    check(col_type == SQLITE_BLOB, "f32: stored as BLOB")

    local nbytes = sq.sqlite3_column_bytes(stmt, 0)
    check(nbytes == 16, "f32: 4 dims × 4 bytes = 16")

    local blob = sq.sqlite3_column_blob(stmt, 0)
    local floats = ffi.cast("const float*", blob)
    check(math.abs(floats[0] - 1.5) < 1e-6, "f32: float[0] = 1.5")
    check(math.abs(floats[1] - (-2.5)) < 1e-6, "f32: float[1] = -2.5")
    check(math.abs(floats[2]) < 1e-6, "f32: float[2] = 0.0")
    check(math.abs(floats[3] - 100.0) < 1e-6, "f32: float[3] = 100.0")

    sq.sqlite3_finalize(stmt)

    -- Round-trip: vec() text output should match input values
    local text_out = nil
    sqlite.query(db, "SELECT vector FROM v WHERE rowid=1", function(s)
        text_out = ffi.string(sq.sqlite3_column_text(s, 0))
    end)
    check(text_out ~= nil, "f32: text round-trip returns text")
    check(text_out:find("1.5") ~= nil, "f32: text contains 1.5")
    check(text_out:find("-2.5") ~= nil, "f32: text contains -2.5")

    close_db(db)
end

print(
    "── Storage: float32 various dimensions ─────────────────────────────")
do
    local db = open_db()
    for _, dims in ipairs({1, 2, 8, 64, 128, 256}) do
        local tname = "vf_" .. dims
        exec(db, string.format("CREATE VIRTUAL TABLE %s USING vec0(dims=%d, metric=l2)", tname, dims))

        local parts = {}
        for i = 1, dims do
            parts[i] = tostring(i * 0.1)
        end
        local vec_str = "[" .. table.concat(parts, ",") .. "]"
        exec(db, string.format("INSERT INTO %s(vector) VALUES('%s')", tname, vec_str))

        local nbytes = nil
        sqlite.query(db, string.format("SELECT LENGTH(vector) FROM %s_data WHERE id=1", tname), function(s)
            nbytes = tonumber(sq.sqlite3_column_int64(s, 0))
        end)
        check(nbytes == dims * 4,
            string.format("f32 dims=%d: BLOB = %d bytes (expected %d)", dims, nbytes or 0, dims * 4))
    end
    close_db(db)
end

print(
    "── Storage: int8 BLOB encoding + clamping ─────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=4, metric=l2, type=int8)")

    -- Values within range
    exec(db, "INSERT INTO v(vector) VALUES('[10, -10, 127, -128]')")

    local stmt = sqlite.prepare(db, "SELECT vector FROM v_data WHERE id=1")
    sq.sqlite3_step(stmt)

    local nbytes = sq.sqlite3_column_bytes(stmt, 0)
    check(nbytes == 4, "i8: 4 dims × 1 byte = 4")

    local blob = sq.sqlite3_column_blob(stmt, 0)
    local bytes = ffi.cast("const int8_t*", blob)
    check(bytes[0] == 10, "i8: byte[0] = 10")
    check(bytes[1] == -10, "i8: byte[1] = -10")
    check(bytes[2] == 127, "i8: byte[2] = 127")
    check(bytes[3] == -128, "i8: byte[3] = -128")

    sq.sqlite3_finalize(stmt)

    -- Clamping: values outside [-128, 127] are clamped
    exec(db, "INSERT INTO v(vector) VALUES('[200, -200, 0, 50]')")

    stmt = sqlite.prepare(db, "SELECT vector FROM v_data WHERE id=2")
    sq.sqlite3_step(stmt)
    blob = sq.sqlite3_column_blob(stmt, 0)
    bytes = ffi.cast("const int8_t*", blob)
    check(bytes[0] == 127, "i8 clamp: 200 → 127")
    check(bytes[1] == -128, "i8 clamp: -200 → -128")
    check(bytes[2] == 0, "i8 clamp: 0 → 0")
    check(bytes[3] == 50, "i8 clamp: 50 → 50")
    sq.sqlite3_finalize(stmt)

    close_db(db)
end

print(
    "── Storage: int8 various dimensions ────────────────────────────────")
do
    local db = open_db()
    for _, dims in ipairs({1, 4, 16, 64}) do
        local tname = "vi_" .. dims
        exec(db, string.format("CREATE VIRTUAL TABLE %s USING vec0(dims=%d, metric=l2, type=int8)", tname, dims))

        local parts = {}
        for i = 1, dims do
            parts[i] = tostring(i)
        end
        exec(db, string.format("INSERT INTO %s(vector) VALUES('[%s]')", tname, table.concat(parts, ",")))

        local nbytes = nil
        sqlite.query(db, string.format("SELECT LENGTH(vector) FROM %s_data WHERE id=1", tname), function(s)
            nbytes = tonumber(sq.sqlite3_column_int64(s, 0))
        end)
        check(nbytes == dims, string.format("i8 dims=%d: BLOB = %d bytes", dims, nbytes or 0))
    end
    close_db(db)
end

print(
    "── Storage: bit BLOB encoding (MSB-first) ─────────────────────────")
do
    local db = open_db()
    exec(db, "CREATE VIRTUAL TABLE v USING vec0(dims=16, metric=hamming, type=bit)")

    -- [1,1,1,1,0,0,0,0, 1,0,1,0,0,1,0,1] → bytes: 0xF0, 0xA5
    exec(db, "INSERT INTO v(vector) VALUES('[1,1,1,1,0,0,0,0,1,0,1,0,0,1,0,1]')")

    local stmt = sqlite.prepare(db, "SELECT vector FROM v_data WHERE id=1")
    sq.sqlite3_step(stmt)

    local nbytes = sq.sqlite3_column_bytes(stmt, 0)
    check(nbytes == 2, "bit: 16 dims / 8 = 2 bytes")

    local blob = sq.sqlite3_column_blob(stmt, 0)
    local bytes = ffi.cast("const uint8_t*", blob)
    check(bytes[0] == 0xF0, string.format("bit: byte[0] = 0x%02X (expected 0xF0)", bytes[0]))
    check(bytes[1] == 0xA5, string.format("bit: byte[1] = 0x%02X (expected 0xA5)", bytes[1]))

    sq.sqlite3_finalize(stmt)

    -- All zeros
    exec(db, "INSERT INTO v(vector) VALUES('[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]')")
    stmt = sqlite.prepare(db, "SELECT vector FROM v_data WHERE id=2")
    sq.sqlite3_step(stmt)
    blob = sq.sqlite3_column_blob(stmt, 0)
    bytes = ffi.cast("const uint8_t*", blob)
    check(bytes[0] == 0x00, "bit: all-zero byte[0] = 0x00")
    check(bytes[1] == 0x00, "bit: all-zero byte[1] = 0x00")
    sq.sqlite3_finalize(stmt)

    -- All ones
    exec(db, "INSERT INTO v(vector) VALUES('[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]')")
    stmt = sqlite.prepare(db, "SELECT vector FROM v_data WHERE id=3")
    sq.sqlite3_step(stmt)
    blob = sq.sqlite3_column_blob(stmt, 0)
    bytes = ffi.cast("const uint8_t*", blob)
    check(bytes[0] == 0xFF, "bit: all-one byte[0] = 0xFF")
    check(bytes[1] == 0xFF, "bit: all-one byte[1] = 0xFF")
    sq.sqlite3_finalize(stmt)

    close_db(db)
end

print(
    "── Storage: bit various dimensions ─────────────────────────────────")
do
    local db = open_db()
    for _, dims in ipairs({8, 16, 32, 64, 128}) do
        local tname = "vb_" .. dims
        exec(db, string.format("CREATE VIRTUAL TABLE %s USING vec0(dims=%d, metric=hamming, type=bit)", tname, dims))

        local parts = {}
        for i = 1, dims do
            parts[i] = (i % 2 == 0) and "1" or "0"
        end
        exec(db, string.format("INSERT INTO %s(vector) VALUES('[%s]')", tname, table.concat(parts, ",")))

        local nbytes = nil
        sqlite.query(db, string.format("SELECT LENGTH(vector) FROM %s_data WHERE id=1", tname), function(s)
            nbytes = tonumber(sq.sqlite3_column_int64(s, 0))
        end)
        check(nbytes == dims / 8,
            string.format("bit dims=%d: BLOB = %d bytes (expected %d)", dims, nbytes or 0, dims / 8))
    end
    close_db(db)
end

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.format("\nstorage_roundtrip: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
