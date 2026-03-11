# Copilot Coding Agent Onboarding for sqlite-vector

Trust this file first. Do not start with broad repo search. Only search when information here is missing or proves incorrect.

## Repository Summary

- `sqlite-vector` is a C11 SQLite loadable extension that adds vector similarity search (`vec0` virtual table) using HNSW.
- It stores vectors and HNSW graph data in SQLite shadow tables (SQL-native persistence, ACID behavior).
- It also includes Ruby scripts for Simple Wikipedia embedding ingest/query using Ollama.

## High-Level Repo Facts

- Project type: native SQLite extension + test/benchmark harness + Ruby demo scripts.
- Primary languages: C (core extension), SQL (integration tests), LuaJIT (FFI tests/benchmarks), Ruby (wiki scripts).
- Build system: `xmake` (`xmake.lua`).
- Vendored dependency: `third_party/simsimd` (git submodule; headers used at compile time).
- Approximate size: medium (core logic in `src/`, largest file is `src/vtab.c`; large README).
- Target/runtime: SQLite loadable extension (`libsqlite_vector.dylib/.so/.dll`).

## Toolchain and Versions (validated on macOS arm64)

Always verify these before debugging build failures:

- `xmake v3.0.7+20260210`
- `sqlite3 3.51.2`
- `LuaJIT 2.1.1772619647`
- `ruby 4.0.1`
- `bundler 4.0.3`
- `ollama` present at `/opt/homebrew/bin/ollama` (required for wiki ingest/query runtime)

## Bootstrap (Always do in this order)

1. Clone with submodules (required for SimSIMD headers):
   - `git clone --recursive <repo-url>`
   - If already cloned: `git submodule update --init --recursive`
2. Install Ruby gems (needed for script lint/run):
   - `bundle check || bundle install`
3. Build extension:
   - `xmake`

Expected build artifact:

- `build/macosx/arm64/release/libsqlite_vector.dylib` on this machine.

## Build, Run, Test, Lint (validated commands)

### Build

- `xmake`
- Clean build path:
  - `xmake clean && rm -rf build .xmake && xmake`
- Observed clean build time: ~0.85-0.94s.

Notes:

- Build emits SimSIMD warnings from `third_party/simsimd/include/...` about `nonnull` attributes; these were non-fatal.

### Run / Smoke Test

- `sqlite3 :memory: ".load build/macosx/arm64/release/libsqlite_vector" "SELECT vec_distance_l2('[1,2]','[1,2]');"`
- Expected output: `0.0`

### Tests

Canonical:

- `sh test/run_all.sh`

Faster when already built:

- `sh test/run_all.sh --no-build`

Observed times:

- `sh test/run_all.sh --no-build` after successful build: ~5.4s
- `sh test/run_all.sh` (includes build): ~5.3s

Critical precondition:

- Always build before `--no-build`.
- Verified failure when skipped build:
  - `Error: dlopen(build/macosx/arm64/release/libsqlite_vector.dylib, ... no such file)`
- Mitigation: run `xmake`, then rerun tests.

### Lint

Ruby scripts:

- `bundle exec rubocop scripts/wiki_ingest.rb scripts/wiki_query.rb`
- `bundle exec rubocop -A`

Current repo behavior (validated):

- RuboCop exits non-zero (`code 1`) due many metrics/style offenses in `scripts/wiki_ingest.rb` and `scripts/wiki_query.rb`.
- There is no repo `.rubocop.yml`, so RuboCop also prints long "new cops not configured" notices.
- Treat RuboCop as informational unless your task explicitly requires Ruby lint cleanup.

### Wiki Script Entrypoints

Option/argument validation (safe, fast):

- `bundle exec ruby scripts/wiki_ingest.rb --help`
- `bundle exec ruby scripts/wiki_query.rb --help`

Runtime preconditions for real ingest/query:

- Built extension present (auto-detected from `build/*/*/release/libsqlite_vector.*` unless `--extension-path` is set).
- Ollama model available and server running:
  - `ollama pull embeddinggemma`
  - `ollama serve`
- Ingest additionally requires `bzcat` (bzip2 tools); script raises explicit error if missing.

## Command Ordering That Works Reliably

Always use this flow for code changes:

1. `xmake`
2. `sh test/run_all.sh --no-build` (or `sh test/run_all.sh`)
3. If touching Ruby scripts: `bundle exec rubocop scripts/wiki_ingest.rb scripts/wiki_query.rb`
4. Optional smoke: sqlite3 `.load` + simple function query

If anything fails due missing artifact, re-run `xmake` first before deeper debugging.

## Architecture and Layout (where to edit)

Core extension:

- `src/extension.c`: SQLite extension entrypoint (`sqlite3_sqlitevector_init`), registers module/functions.
- `src/vtab.c` + `src/vtab.h`: `vec0` virtual table implementation and module symbol.
- `src/hnsw.c` + `src/hnsw.h`: HNSW graph/index operations.
- `src/distance.c` + `src/distance.h`: metric kernels + SQL distance functions.
- `src/vec_parse.c` + `src/vec_parse.h`: vector text parsing/formatting.
- `src/vec_ops.c` + `src/vec_ops.h`: vector math utility SQL functions.

Build/config:

- `xmake.lua`: shared-library build, platform-specific flags.
- `include/sqlite3.h`, `include/sqlite3ext.h`: bundled headers.
- `.gitmodules`: defines `third_party/simsimd` submodule.

Validation and tests:

- `test/run_all.sh`: authoritative local validation sequence.
- `test/*.sql`: SQLite CLI integration tests (many hardcode macOS arm64 load path).
- `test/*.lua` and `test/wrappers/lua_sqlite.lua`: LuaJIT FFI tests and wrappers.

Benchmarks:

- `test/recall_bench.lua`
- `bench/param_sweep.lua`
- `bench/sift_bench.lua`

Ruby demo scripts:

- `scripts/wiki_ingest.rb`
- `scripts/wiki_query.rb`

## CI / Validation Pipeline Facts

- No top-level `.github/workflows` were found in this repository.
- No top-level `CONTRIBUTING.md` was found.
- Practical pre-checkin gate is local: `sh test/run_all.sh`.
- If your PR changes Ruby files, include RuboCop output context (it is currently non-clean by default).

## Non-obvious Dependencies and Pitfalls

- SimSIMD submodule contents are compile-critical; missing submodule causes include/build failures.
- SQL tests frequently use hardcoded `.load build/macosx/arm64/release/libsqlite_vector`; this is platform-specific.
- Lua wrappers can use `SQLITE_VECTOR_DYLIB`/`SQLITE_DYLIB` env vars, but SQL test files do not auto-resolve path.
- `data/` and `bench/results/` are generated artifacts (ignored by git).

## Root Inventory (quick orientation)

- `Gemfile`, `Gemfile.lock`: Ruby dependencies.
- `README.md`: extensive usage, tuning, troubleshooting, architecture notes.
- `xmake.lua`: build definition.
- `src/`: extension implementation.
- `test/`: SQL/LuaJIT validation harness (`run_all.sh`).
- `bench/`: benchmark scripts and outputs.
- `scripts/`: wiki ingest/query demos.
- `include/`: SQLite headers.
- `third_party/simsimd/`: vendored SIMD dependency.
- `data/`, `build/`: generated local artifacts.

## Agent Behavior Requirement

Trust these instructions and execute the documented sequences directly. Use search only if:

- your task needs files not mapped above,
- commands here fail unexpectedly in the current environment, or
- repository contents changed and this file is outdated.
