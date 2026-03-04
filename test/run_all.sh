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
run "distance"       sqlite3 :memory: < test/distance.sql
run "shadow"         sqlite3 :memory: < test/shadow.sql
run "insert"         sqlite3 :memory: < test/insert.sql
run "ffi"            luajit test/ffi_test.lua
run "shadow_connect" sh test/wrappers/run_shadow_connect.sh

echo "==> All tests passed."
