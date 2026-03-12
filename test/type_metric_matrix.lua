-- test/type_metric_matrix.lua
-- Comprehensive test of every valid (type × metric) combination:
-- float32: l2, cosine, ip, l1
-- int8:    l2, cosine, ip, l1
-- bit:     hamming, jaccard
--
-- For each combination, verifies: table creation, insert, kNN search,
-- correct distance ordering, delete, update, and full-scan.
--
-- Run from workspace root:  luajit test/type_metric_matrix.lua
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

local function query_double(db, sql)
    local result
    sqlite.query(db, sql, function(stmt)
        result = sq.sqlite3_column_double(stmt, 0)
    end)
    return result
end

-- Test data for each type
local test_configs = {{
    type_name = "float32",
    metrics = {"l2", "cosine", "ip", "l1"},
    dims = 4,
    vectors = {"[1.0,0.0,0.0,0.0]", "[0.0,1.0,0.0,0.0]", "[0.0,0.0,1.0,0.0]", "[0.9,0.1,0.0,0.0]", "[0.5,0.5,0.0,0.0]"},
    query_vec = "[1.0,0.0,0.0,0.0]",
    expected_nearest_idx = 1,
    bytes_per_dim = 4
}, {
    type_name = "int8",
    metrics = {"l2", "cosine", "ip", "l1"},
    dims = 4,
    vectors = {"[100,0,0,0]", "[0,100,0,0]", "[0,0,100,0]", "[90,10,0,0]", "[50,50,0,0]"},
    query_vec = "[100,0,0,0]",
    expected_nearest_idx = 1,
    bytes_per_dim = 1
}, {
    type_name = "bit",
    metrics = {"hamming", "jaccard"},
    -- bit vectors: dims must be multiple of 8
    dims = 16,
    vectors = {"[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]", "[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0]",
               "[0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0]", "[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]",
               "[1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]"},
    query_vec = "[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]",
    expected_nearest_idx = 1,
    bytes_per_dim = nil -- bit: dims/8 bytes
}}

local table_idx = 0

for _, cfg in ipairs(test_configs) do
    for _, metric in ipairs(cfg.metrics) do
        table_idx = table_idx + 1
        local tname = string.format("v%d", table_idx)
        local label = string.format("%s/%s", cfg.type_name, metric)

        print(string.format(
            "── %s ────────────────────────────────────────",
            label))

        local db = open_db()

        -- ── CREATE ──────────────────────────────────────────────────────
        local create_sql = string.format("CREATE VIRTUAL TABLE %s USING vec0(dims=%d, metric=%s, type=%s)", tname,
            cfg.dims, metric, cfg.type_name)
        exec(db, create_sql)

        -- Verify config
        local stored_dims = query_int(db, string.format("SELECT CAST(value AS INTEGER) FROM %s_config WHERE key='dims'",
            tname))
        check(stored_dims == cfg.dims, label .. ": dims config == " .. cfg.dims)

        local stored_metric = nil
        sqlite.query(db, string.format("SELECT value FROM %s_config WHERE key='metric'", tname), function(stmt)
            stored_metric = ffi.string(sq.sqlite3_column_text(stmt, 0))
        end)
        check(stored_metric == metric, label .. ": metric config == " .. metric)

        -- ── INSERT ──────────────────────────────────────────────────────
        for _, vec in ipairs(cfg.vectors) do
            exec(db, string.format("INSERT INTO %s(vector) VALUES('%s')", tname, vec))
        end
        local count = query_int(db, string.format("SELECT COUNT(*) FROM %s_data", tname))
        check(count == #cfg.vectors, label .. ": " .. #cfg.vectors .. " vectors inserted")

        -- ── STORAGE SIZE ────────────────────────────────────────────────
        local blob_len = query_int(db, string.format("SELECT LENGTH(vector) FROM %s_data WHERE id=1", tname))
        local expected_len
        if cfg.type_name == "bit" then
            expected_len = cfg.dims / 8
        else
            expected_len = cfg.dims * cfg.bytes_per_dim
        end
        check(blob_len == expected_len, label .. ": BLOB size " .. tostring(blob_len) .. " == " .. expected_len)

        -- ── KNN ─────────────────────────────────────────────────────────
        local knn_sql = string.format("SELECT rowid, distance FROM %s WHERE %s MATCH '%s' LIMIT 3", tname, tname,
            cfg.query_vec)

        local knn_results = {}
        sqlite.query(db, knn_sql, function(stmt)
            knn_results[#knn_results + 1] = {
                rowid = tonumber(sq.sqlite3_column_int64(stmt, 0)),
                distance = sq.sqlite3_column_double(stmt, 1)
            }
        end)

        check(#knn_results == 3, label .. ": kNN returns 3 results")

        -- Nearest should be the first inserted vector (self-match)
        if #knn_results > 0 then
            check(knn_results[1].rowid == cfg.expected_nearest_idx,
                label .. ": nearest is rowid " .. cfg.expected_nearest_idx .. " (got " .. knn_results[1].rowid .. ")")
            -- Inner product can be negative; only check non-negative for non-IP metrics
            if metric ~= "ip" then
                check(knn_results[1].distance >= 0, label .. ": nearest distance >= 0")
            end
        end

        -- Distances should be non-decreasing
        local ordered = true
        for i = 2, #knn_results do
            if knn_results[i].distance < knn_results[i - 1].distance - 1e-6 then
                ordered = false
            end
        end
        check(ordered, label .. ": kNN distances are non-decreasing")

        -- ── FULL SCAN ───────────────────────────────────────────────────
        local scan_count = query_int(db, string.format("SELECT COUNT(*) FROM %s", tname))
        check(scan_count == #cfg.vectors, label .. ": full scan returns all " .. #cfg.vectors)

        -- ── DELETE ──────────────────────────────────────────────────────
        exec(db, string.format("DELETE FROM %s WHERE rowid=3", tname))
        local after_del = query_int(db, string.format("SELECT COUNT(*) FROM %s_data", tname))
        check(after_del == #cfg.vectors - 1, label .. ": delete removes one row")

        -- kNN still works after delete
        local knn_after = 0
        sqlite.query(db, string.format("SELECT rowid FROM %s WHERE %s MATCH '%s' LIMIT 3", tname, tname, cfg.query_vec),
            function()
                knn_after = knn_after + 1
            end)
        check(knn_after >= 1, label .. ": kNN works after delete")

        -- ── UPDATE ──────────────────────────────────────────────────────
        exec(db, string.format("UPDATE %s SET vector = '%s' WHERE rowid=1", tname, cfg.vectors[2]))
        local update_count = query_int(db, string.format("SELECT COUNT(*) FROM %s_data", tname))
        check(update_count == #cfg.vectors - 1, label .. ": update preserves row count")

        -- ── GRAPH INTEGRITY ─────────────────────────────────────────────
        local orphans = query_int(db,
            string.format("SELECT COUNT(*) FROM %s_graph g " .. "WHERE g.node_id NOT IN (SELECT id FROM %s_data) " ..
                              "OR g.neighbor_id NOT IN (SELECT id FROM %s_data)", tname, tname, tname))
        check(orphans == 0, label .. ": no orphan graph edges")

        close_db(db)
    end
end

-- ── Summary ──────────────────────────────────────────────────────────────────
print(string.format("\ntype_metric_matrix: %d passed, %d failed", pass, fail))
if fail > 0 then
    os.exit(1)
end
