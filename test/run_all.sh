#!/bin/sh
# test/run_all.sh — run every test suite in order.
# Usage: sh test/run_all.sh [--no-build]
set -e
cd "$(dirname "$0")/.."

BUILD=1
for arg in "$@"; do
  [ "$arg" = "--no-build" ] && BUILD=0
done

if [ "$BUILD" -eq 1 ]; then
  echo "==> Building..."
  xmake -q
fi

run() {
  name="$1"; shift
  printf "==> %-30s " "$name"
  if "$@" > /tmp/sv_test_out.txt 2>&1; then
    echo "PASS"
  else
    echo "FAIL"
    cat /tmp/sv_test_out.txt
    exit 1
  fi
}

run "basic"          sqlite3 :memory: < test/basic.sql
run "vec_parse"      sqlite3 :memory: < test/vec_parse.sql
run "vec_parse_edge" sqlite3 :memory: < test/vec_parse_edge.sql
run "vec_ops"        sqlite3 :memory: < test/vec_ops.sql
run "distance"       sqlite3 :memory: < test/distance.sql
run "dist_thresh"    sqlite3 :memory: < test/distance_constraints.sql
run "vector_types"   sqlite3 :memory: < test/vector_types.sql
run "hamm_jacc"      sqlite3 :memory: < test/hamming_jaccard.sql
run "shadow"         sqlite3 :memory: < test/shadow.sql
run "insert"         sqlite3 :memory: < test/insert.sql
run "ffi"            luajit test/ffi_test.lua
run "shadow_connect" sh test/wrappers/run_shadow_connect.sh
run "knn"            sqlite3 :memory: < test/knn.sql
run "rename"         sqlite3 :memory: < test/rename.sql
run "operators"      sqlite3 :memory: < test/operators.sql
run "bulk_insert"    sqlite3 :memory: < test/bulk_insert.sql
run "delete_repair"  sqlite3 :memory: < test/delete_repair.sql
run "atomic_update"  sqlite3 :memory: < test/atomic_update.sql
run "persist"        luajit test/persist_test.lua
run "wal_persist"    luajit test/wal_persist.lua
run "fuzz"           luajit test/fuzz.lua
run "recall"         luajit test/recall_bench.lua

echo "==> All tests passed."
