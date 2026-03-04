#!/bin/sh
# test/wrappers/run_shadow_connect.sh
# Sets up an on-disk database, then re-opens it to test xConnect.
set -e
cd "$(dirname "$0")/../.."

DB=/tmp/sv_connect_test.db
rm -f "$DB"
sqlite3 "$DB" < test/shadow_setup.sql
sqlite3 "$DB" < test/shadow_connect.sql
rm -f "$DB"
