#include "vtab.h"
#include "distance.h"
#include "hnsw.h"
#include "vec_parse.h"

#include <sqlite3ext.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* ── Internal structs ────────────────────────────────────────────────────────
 */

typedef struct Vec0Table Vec0Table;
struct Vec0Table {
  sqlite3_vtab base; /* Must be first */
  sqlite3 *db;
  char *name;   /* Virtual table name (for shadow table names) */
  char *schema; /* Schema name */
  int dims;     /* Vector dimensionality */
  char *metric; /* Distance metric name */
  int m;        /* HNSW M parameter */
  int ef_construction;
  int ef_search;
  sqlite3_int64 entry_point; /* Rowid of the current HNSW entry-point node */
  int max_layer;             /* Maximum layer in the HNSW graph */
  sqlite3_int64 count;       /* Number of vectors stored */
  dist_fn_t dist_fn; /* Distance kernel (set from metric at init time) */
};

typedef struct Vec0Cursor Vec0Cursor;
struct Vec0Cursor {
  sqlite3_vtab_cursor base; /* Must be first */
  int64_t *rowids;          /* Result rowid array from kNN search */
  double *distances;        /* Parallel distances array */
  int nResults;             /* Number of results */
  int pos;                  /* Current cursor position */
};

/* ── Helpers ─────────────────────────────────────────────────────────────────
 */

/* Parse a key=value argument from CREATE VIRTUAL TABLE argv. */
static int parse_arg(const char *arg, const char *key, char **out_val) {
  size_t klen = strlen(key);
  if (strncmp(arg, key, klen) == 0 && arg[klen] == '=') {
    *out_val = sqlite3_mprintf("%s", arg + klen + 1);
    return 1;
  }
  return 0;
}

static void vec0_free(Vec0Table *p) {
  if (!p)
    return;
  sqlite3_free(p->name);
  sqlite3_free(p->schema);
  sqlite3_free(p->metric);
  sqlite3_free(p);
}

/* ── Shadow table helpers ─────────────────────────────────────────────────
 */

/* Declare the virtual table schema.  Must be called from both xCreate and
 * xConnect before returning. */
static int vec0_declare(sqlite3 *db, const char *name, char **pzErr) {
  char *schema = sqlite3_mprintf(
      "CREATE TABLE x(\"%w\" HIDDEN, vector TEXT, distance REAL HIDDEN)", name);
  if (!schema)
    return SQLITE_NOMEM;
  int rc = sqlite3_declare_vtab(db, schema);
  sqlite3_free(schema);
  if (rc != SQLITE_OK)
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(db));
  return rc;
}

/* Create the four shadow tables and persist config.  Called only from
 * xCreate (first-time table creation, not database re-open). */
static int vec0_create_shadow(Vec0Table *p, char **pzErr) {
  const char *n = p->name;
  const char *s = p->schema;
  int rc;
  char *sql;

  /* _config */
  sql = sqlite3_mprintf("CREATE TABLE \"%w\".\"%w_config\" ("
                        "  key TEXT PRIMARY KEY, "
                        "  value TEXT"
                        ") WITHOUT ROWID",
                        s, n);
  rc = sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }

  /* _data */
  sql = sqlite3_mprintf("CREATE TABLE \"%w\".\"%w_data\" ("
                        "  id     INTEGER PRIMARY KEY, "
                        "  vector BLOB NOT NULL"
                        ")",
                        s, n);
  rc = sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }

  /* _graph */
  sql = sqlite3_mprintf("CREATE TABLE \"%w\".\"%w_graph\" ("
                        "  layer       INTEGER NOT NULL, "
                        "  node_id     INTEGER NOT NULL, "
                        "  neighbor_id INTEGER NOT NULL, "
                        "  distance    REAL    NOT NULL, "
                        "  PRIMARY KEY (layer, node_id, neighbor_id)"
                        ") WITHOUT ROWID",
                        s, n);
  rc = sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }

  /* _layers */
  sql = sqlite3_mprintf("CREATE TABLE \"%w\".\"%w_layers\" ("
                        "  node_id   INTEGER PRIMARY KEY, "
                        "  max_layer INTEGER NOT NULL"
                        ") WITHOUT ROWID",
                        s, n);
  rc = sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }

  /* Write config */
  sql = sqlite3_mprintf("INSERT INTO \"%w\".\"%w_config\" (key, value) VALUES "
                        "('dims','%d'),('metric','%q'),('m','%d'),"
                        "('ef_construction','%d'),('ef_search','%d'),"
                        "('entry_point','-1'),('max_layer','0'),('count','0')",
                        s, n, p->dims, p->metric, p->m, p->ef_construction,
                        p->ef_search);
  rc = sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
  return rc;
}

/* Read config from _config shadow table into p.  Called from xConnect. */
static int vec0_read_config(Vec0Table *p, char **pzErr) {
  char *sql = sqlite3_mprintf("SELECT key, value FROM \"%w\".\"%w_config\"",
                              p->schema, p->name);
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char *key = (const char *)sqlite3_column_text(stmt, 0);
    const char *val = (const char *)sqlite3_column_text(stmt, 1);
    if (!key || !val)
      continue;
    if (strcmp(key, "dims") == 0)
      p->dims = atoi(val);
    else if (strcmp(key, "metric") == 0) {
      sqlite3_free(p->metric);
      p->metric = sqlite3_mprintf("%s", val);
    } else if (strcmp(key, "m") == 0)
      p->m = atoi(val);
    else if (strcmp(key, "ef_construction") == 0)
      p->ef_construction = atoi(val);
    else if (strcmp(key, "ef_search") == 0)
      p->ef_search = atoi(val);
    else if (strcmp(key, "entry_point") == 0)
      p->entry_point = (sqlite3_int64)atoll(val);
    else if (strcmp(key, "max_layer") == 0)
      p->max_layer = atoi(val);
    else if (strcmp(key, "count") == 0)
      p->count = (sqlite3_int64)atoll(val);
  }
  sqlite3_finalize(stmt);
  return SQLITE_OK;
}

/* ── xCreate / xConnect ──────────────────────────────────────────────────────
 */

static int vec0Init(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                    sqlite3_vtab **ppVtab, char **pzErr, int isCreate) {
  (void)pAux;

  Vec0Table *p = sqlite3_malloc(sizeof(*p));
  if (!p)
    return SQLITE_NOMEM;
  memset(p, 0, sizeof(*p));

  p->db = db;
  p->schema = sqlite3_mprintf("%s", argc >= 2 ? argv[1] : "main");
  p->name = sqlite3_mprintf("%s", argc >= 3 ? argv[2] : "vec0");

  /* Defaults */
  p->metric = sqlite3_mprintf("cosine");
  p->m = 16;
  p->ef_construction = 200;
  p->ef_search = 10;
  p->entry_point = -1;

  if (isCreate) {
    /* Parse options from argv[3..] */
    for (int i = 3; i < argc; i++) {
      char *val = NULL;
      if (parse_arg(argv[i], "dims", &val)) {
        p->dims = atoi(val);
        sqlite3_free(val);
      } else if (parse_arg(argv[i], "metric", &val)) {
        sqlite3_free(p->metric);
        p->metric = val;
      } else if (parse_arg(argv[i], "m", &val)) {
        p->m = atoi(val);
        sqlite3_free(val);
      } else if (parse_arg(argv[i], "ef_construction", &val)) {
        p->ef_construction = atoi(val);
        sqlite3_free(val);
      } else if (parse_arg(argv[i], "ef_search", &val)) {
        p->ef_search = atoi(val);
        sqlite3_free(val);
      }
    }

    if (p->dims <= 0) {
      *pzErr = sqlite3_mprintf("vec0: dims=N is required and must be > 0");
      vec0_free(p);
      return SQLITE_ERROR;
    }

    int rc = vec0_create_shadow(p, pzErr);
    if (rc != SQLITE_OK) {
      vec0_free(p);
      return rc;
    }
  } else {
    /* xConnect: read config persisted in shadow table */
    int rc = vec0_read_config(p, pzErr);
    if (rc != SQLITE_OK) {
      vec0_free(p);
      return rc;
    }
  }

  /* Resolve distance kernel from metric string */
  p->dist_fn = distance_for_metric(p->metric);
  if (!p->dist_fn) {
    *pzErr = sqlite3_mprintf("vec0: unknown metric '%s'", p->metric);
    vec0_free(p);
    return SQLITE_ERROR;
  }

  {
    int rc = vec0_declare(db, p->name, pzErr);
    if (rc != SQLITE_OK) {
      vec0_free(p);
      return rc;
    }
  }

  *ppVtab = (sqlite3_vtab *)p;
  return SQLITE_OK;
}

static int vec0Create(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  return vec0Init(db, pAux, argc, argv, ppVtab, pzErr, 1);
}

static int vec0Connect(sqlite3 *db, void *pAux, int argc,
                       const char *const *argv, sqlite3_vtab **ppVtab,
                       char **pzErr) {
  return vec0Init(db, pAux, argc, argv, ppVtab, pzErr, 0);
}

/* ── xDisconnect / xDestroy ──────────────────────────────────────────────────
 */

static int vec0Disconnect(sqlite3_vtab *pVtab) {
  vec0_free((Vec0Table *)pVtab);
  return SQLITE_OK;
}

static int vec0Destroy(sqlite3_vtab *pVtab) {
  Vec0Table *p = (Vec0Table *)pVtab;
  static const char *shadows[] = {"config", "data", "graph", "layers"};
  for (int i = 0; i < 4; i++) {
    char *sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w\".\"%w_%w\"",
                                p->schema, p->name, shadows[i]);
    sqlite3_exec(p->db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
  }
  vec0_free(p);
  return SQLITE_OK;
}

/* ── xBestIndex ──────────────────────────────────────────────────────────────
 */

static int vec0BestIndex(sqlite3_vtab *pVtab, sqlite3_index_info *pInfo) {
  (void)pVtab;

  /* 1. Look for a MATCH constraint on col 0 (the hidden table-name column).
   * WHERE vecs MATCH query  →  kNN path */
  for (int i = 0; i < pInfo->nConstraint; i++) {
    if (!pInfo->aConstraint[i].usable)
      continue;
    if (pInfo->aConstraint[i].iColumn == 0 &&
        pInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_MATCH) {
      pInfo->aConstraintUsage[i].argvIndex =
          1; /* query vector → xFilter argv[0] */
      pInfo->aConstraintUsage[i].omit = 1;
      pInfo->idxNum = 1; /* kNN path */
      pInfo->estimatedCost = 10.0;
      pInfo->estimatedRows = 10;
      return SQLITE_OK;
    }
  }

  /* 2. Rowid equality — used for DELETE/point lookups. */
  for (int i = 0; i < pInfo->nConstraint; i++) {
    if (!pInfo->aConstraint[i].usable)
      continue;
    if (pInfo->aConstraint[i].iColumn == -1 &&
        pInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_EQ) {
      pInfo->aConstraintUsage[i].argvIndex = 1;
      pInfo->aConstraintUsage[i].omit = 1;
      pInfo->idxNum = 2; /* rowid lookup */
      pInfo->idxFlags = SQLITE_INDEX_SCAN_UNIQUE;
      pInfo->estimatedCost = 1.0;
      pInfo->estimatedRows = 1;
      return SQLITE_OK;
    }
  }

  /* 3. Full scan fallback */
  pInfo->idxNum = 0;
  pInfo->estimatedCost = 1e9;
  pInfo->estimatedRows = 100000;
  return SQLITE_OK;
}

/* ── xOpen / xClose ──────────────────────────────────────────────────────────
 */

static int vec0Open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor) {
  (void)pVtab;
  Vec0Cursor *cur = sqlite3_malloc(sizeof(*cur));
  if (!cur)
    return SQLITE_NOMEM;
  memset(cur, 0, sizeof(*cur));
  *ppCursor = (sqlite3_vtab_cursor *)cur;
  return SQLITE_OK;
}

static int vec0Close(sqlite3_vtab_cursor *pCursor) {
  Vec0Cursor *cur = (Vec0Cursor *)pCursor;
  sqlite3_free(cur->rowids);
  sqlite3_free(cur->distances);
  sqlite3_free(cur);
  return SQLITE_OK;
}

/* ── xFilter ─────────────────────────────────────────────────────────────────
 */

static int vec0Filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
  (void)idxStr;
  (void)argc;
  (void)argv;

  Vec0Cursor *cur = (Vec0Cursor *)pCursor;

  /* Free previous results */
  sqlite3_free(cur->rowids);
  sqlite3_free(cur->distances);
  cur->rowids = NULL;
  cur->distances = NULL;
  cur->nResults = 0;
  cur->pos = 0;

  if (idxNum == 0) {
    /* Full scan — read every rowid from _data */
    Vec0Table *p = (Vec0Table *)pCursor->pVtab;
    char *sql = sqlite3_mprintf("SELECT id FROM \"%w\".\"%w_data\" ORDER BY id",
                                p->schema, p->name);
    sqlite3_stmt *stmt = NULL;
    if (!sql || sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL) != SQLITE_OK) {
      sqlite3_free(sql);
      return SQLITE_OK; /* return empty on error */
    }
    sqlite3_free(sql);
    /* Count rows first */
    int cap = 16, n = 0;
    cur->rowids = sqlite3_malloc(cap * (int)sizeof(int64_t));
    if (!cur->rowids) {
      sqlite3_finalize(stmt);
      return SQLITE_NOMEM;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      if (n == cap) {
        cap *= 2;
        int64_t *tmp = sqlite3_realloc(cur->rowids, cap * (int)sizeof(int64_t));
        if (!tmp) {
          sqlite3_finalize(stmt);
          return SQLITE_NOMEM;
        }
        cur->rowids = tmp;
      }
      cur->rowids[n++] = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    cur->nResults = n;
    /* distances left NULL — xColumn col 2 returns NULL */
    return SQLITE_OK;
  }

  if (idxNum == 2) {
    /* Rowid equality lookup */
    Vec0Table *p = (Vec0Table *)pCursor->pVtab;
    if (argc < 1)
      return SQLITE_OK;
    sqlite3_int64 target = sqlite3_value_int64(argv[0]);
    char *sql =
        sqlite3_mprintf("SELECT id FROM \"%w\".\"%w_data\" WHERE id=%lld",
                        p->schema, p->name, target);
    sqlite3_stmt *stmt = NULL;
    if (sql && sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL) == SQLITE_OK) {
      sqlite3_free(sql);
      if (sqlite3_step(stmt) == SQLITE_ROW) {
        cur->rowids = sqlite3_malloc((int)sizeof(int64_t));
        if (!cur->rowids) {
          sqlite3_finalize(stmt);
          return SQLITE_NOMEM;
        }
        cur->rowids[0] = sqlite3_column_int64(stmt, 0);
        cur->nResults = 1;
      }
      sqlite3_finalize(stmt);
    } else {
      sqlite3_free(sql);
    }
    return SQLITE_OK;
  }

  /* kNN path */
  Vec0Table *p = (Vec0Table *)pCursor->pVtab;

  if (argc < 1 || sqlite3_value_type(argv[0]) == SQLITE_NULL)
    return SQLITE_OK;

  const char *query_text = (const char *)sqlite3_value_text(argv[0]);
  float *query_vec = NULL;
  int query_dims = 0;
  char *parse_err = NULL;
  int rc = vec_parse(query_text, &query_vec, &query_dims, &parse_err);
  if (rc != SQLITE_OK) {
    sqlite3_free(parse_err);
    return rc;
  }
  if (query_dims != p->dims) {
    sqlite3_free(query_vec);
    pCursor->pVtab->zErrMsg = sqlite3_mprintf(
        "vec0: query has %d dims, table has %d", query_dims, p->dims);
    return SQLITE_ERROR;
  }

  /* Default k: from LIMIT argv[1] if present, else ef_search */
  int k = p->ef_search;
  if (argc >= 2 && sqlite3_value_type(argv[1]) == SQLITE_INTEGER) {
    int lim = sqlite3_value_int(argv[1]);
    if (lim > 0)
      k = lim;
  }

  HnswCtx hctx;
  hctx.db = p->db;
  hctx.schema = p->schema;
  hctx.tbl_name = p->name;
  hctx.dims = p->dims;
  hctx.m = p->m;
  hctx.ef_construction = p->ef_construction;
  hctx.ef_search = p->ef_search;
  hctx.entry_point = p->entry_point;
  hctx.max_layer = p->max_layer;
  hctx.dist_fn = p->dist_fn;

  HnswResult *results = NULL;
  int nResults = 0;
  rc = hnsw_search(&hctx, query_vec, k, &results, &nResults);
  sqlite3_free(query_vec);
  if (rc != SQLITE_OK)
    return rc;

  cur->rowids =
      sqlite3_malloc((nResults > 0 ? nResults : 1) * (int)sizeof(int64_t));
  cur->distances =
      sqlite3_malloc((nResults > 0 ? nResults : 1) * (int)sizeof(double));
  if (!cur->rowids || !cur->distances) {
    sqlite3_free(results);
    sqlite3_free(cur->rowids);
    sqlite3_free(cur->distances);
    cur->rowids = NULL;
    cur->distances = NULL;
    return SQLITE_NOMEM;
  }
  for (int i = 0; i < nResults; i++) {
    cur->rowids[i] = results[i].rowid;
    cur->distances[i] = results[i].dist;
  }
  cur->nResults = nResults;
  sqlite3_free(results);
  return SQLITE_OK;
}

/* ── xNext / xEof ────────────────────────────────────────────────────────────
 */

static int vec0Next(sqlite3_vtab_cursor *pCursor) {
  Vec0Cursor *cur = (Vec0Cursor *)pCursor;
  cur->pos++;
  return SQLITE_OK;
}

static int vec0Eof(sqlite3_vtab_cursor *pCursor) {
  Vec0Cursor *cur = (Vec0Cursor *)pCursor;
  return cur->pos >= cur->nResults;
}

/* ── xColumn ─────────────────────────────────────────────────────────────────
 */

static int vec0Column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx,
                      int col) {
  Vec0Cursor *cur = (Vec0Cursor *)pCursor;
  if (cur->pos >= cur->nResults) {
    sqlite3_result_null(ctx);
    return SQLITE_OK;
  }
  switch (col) {
  case 0: /* hidden table-name column (MATCH input) — return NULL */
    sqlite3_result_null(ctx);
    break;
  case 1: { /* vector TEXT — fetch blob from _data and format as [x,y,...] */
    Vec0Table *p = (Vec0Table *)pCursor->pVtab;
    char *sql =
        sqlite3_mprintf("SELECT vector FROM \"%w\".\"%w_data\" WHERE id=%lld",
                        p->schema, p->name, cur->rowids[cur->pos]);
    sqlite3_stmt *stmt = NULL;
    if (sql && sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL) == SQLITE_OK) {
      sqlite3_free(sql);
      if (sqlite3_step(stmt) == SQLITE_ROW) {
        int bytes = sqlite3_column_bytes(stmt, 0);
        const float *blob = (const float *)sqlite3_column_blob(stmt, 0);
        int dims = bytes / (int)sizeof(float);
        char *text = vec_format(blob, dims);
        if (text) {
          sqlite3_result_text(ctx, text, -1, sqlite3_free);
        } else {
          sqlite3_result_null(ctx);
        }
      } else {
        sqlite3_result_null(ctx);
      }
      sqlite3_finalize(stmt);
    } else {
      sqlite3_free(sql);
      sqlite3_result_null(ctx);
    }
    break;
  }
  case 2: /* distance REAL */
    if (cur->distances)
      sqlite3_result_double(ctx, cur->distances[cur->pos]);
    else
      sqlite3_result_null(ctx);
    break;
  default:
    sqlite3_result_null(ctx);
  }
  return SQLITE_OK;
}

/* ── xRowid ──────────────────────────────────────────────────────────────────
 */

static int vec0Rowid(sqlite3_vtab_cursor *pCursor, sqlite_int64 *pRowid) {
  Vec0Cursor *cur = (Vec0Cursor *)pCursor;
  if (cur->rowids && cur->pos < cur->nResults)
    *pRowid = cur->rowids[cur->pos];
  else
    *pRowid = -1;
  return SQLITE_OK;
}

/* ── xUpdate ────────────────────────────────────────────────────────────────
 *
 * argv mapping for a 3-column schema (hidden, vector TEXT, distance REAL
 * HIDDEN): argv[0]  old rowid        (NULL on INSERT) argv[1]  new rowid (NULL
 * → auto-assign) argv[2]  col 0 value      (hidden table-name col — ignore)
 *   argv[3]  col 1 value      (vector TEXT — the payload)
 *   argv[4]  col 2 value      (distance REAL HIDDEN — ignore on INSERT)
 */

static int vec0Update(sqlite3_vtab *pVtab, int argc, sqlite3_value **argv,
                      sqlite_int64 *pRowid) {
  Vec0Table *p = (Vec0Table *)pVtab;

  /* DELETE: argc == 1 */
  if (argc == 1) {
    sqlite3_int64 del_rowid = sqlite3_value_int64(argv[0]);

    /* Remove from _data */
    char *dsql = sqlite3_mprintf("DELETE FROM \"%w\".\"%w_data\" WHERE id=%lld",
                                 p->schema, p->name, del_rowid);
    sqlite3_exec(p->db, dsql, NULL, NULL, NULL);
    sqlite3_free(dsql);

    /* Remove from HNSW graph */
    HnswCtx hctx;
    hctx.db = p->db;
    hctx.schema = p->schema;
    hctx.tbl_name = p->name;
    hctx.dims = p->dims;
    hctx.m = p->m;
    hctx.ef_construction = p->ef_construction;
    hctx.ef_search = p->ef_search;
    hctx.entry_point = p->entry_point;
    hctx.max_layer = p->max_layer;
    hctx.dist_fn = p->dist_fn;
    hnsw_delete(&hctx, del_rowid);
    p->entry_point = hctx.entry_point;
    p->max_layer = hctx.max_layer;

    if (p->count > 0)
      p->count--;
    char *csql = sqlite3_mprintf(
        "UPDATE \"%w\".\"%w_config\" SET value='%lld' WHERE key='count'",
        p->schema, p->name, p->count);
    sqlite3_exec(p->db, csql, NULL, NULL, NULL);
    sqlite3_free(csql);
    return SQLITE_OK;
  }

  /* UPDATE: argv[0] is not NULL (existing row being replaced) */
  if (sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    pVtab->zErrMsg = sqlite3_mprintf("vec0: UPDATE not yet supported");
    return SQLITE_CONSTRAINT;
  }

  /* INSERT: argv[1] is NULL (auto-assign rowid) or an explicit integer. */

  /* argv[3] == col 1 == vector TEXT */
  if (sqlite3_value_type(argv[3]) == SQLITE_NULL) {
    pVtab->zErrMsg = sqlite3_mprintf("vec0: vector must not be NULL");
    return SQLITE_CONSTRAINT;
  }
  const char *text = (const char *)sqlite3_value_text(argv[3]);

  float *vec = NULL;
  int dims = 0;
  char *err = NULL;
  int rc = vec_parse(text, &vec, &dims, &err);
  if (rc != SQLITE_OK) {
    pVtab->zErrMsg = err ? err : sqlite3_mprintf("vec0: invalid vector");
    return SQLITE_ERROR;
  }
  if (dims != p->dims) {
    sqlite3_free(vec);
    pVtab->zErrMsg =
        sqlite3_mprintf("vec0: expected %d dims, got %d", p->dims, dims);
    return SQLITE_CONSTRAINT;
  }

  /* Insert raw float BLOB into _data */
  char *sql = sqlite3_mprintf(
      "INSERT INTO \"%w\".\"%w_data\" (vector) VALUES (?)", p->schema, p->name);
  sqlite3_stmt *stmt = NULL;
  rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    sqlite3_free(vec);
    pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }
  /* Use SQLITE_STATIC so vec stays alive until after hnsw_insert. */
  sqlite3_bind_blob(stmt, 1, vec, dims * (int)sizeof(float), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    sqlite3_free(vec);
    pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return SQLITE_ERROR;
  }

  *pRowid = sqlite3_last_insert_rowid(p->db);
  p->count++;

  /* Keep count in _config in sync */
  sql = sqlite3_mprintf(
      "UPDATE \"%w\".\"%w_config\" SET value = '%lld' WHERE key = 'count'",
      p->schema, p->name, p->count);
  sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);

  /* Insert into HNSW graph */
  HnswCtx hctx;
  hctx.db = p->db;
  hctx.schema = p->schema;
  hctx.tbl_name = p->name;
  hctx.dims = p->dims;
  hctx.m = p->m;
  hctx.ef_construction = p->ef_construction;
  hctx.ef_search = p->ef_search;
  hctx.entry_point = p->entry_point;
  hctx.max_layer = p->max_layer;
  hctx.dist_fn = p->dist_fn;
  rc = hnsw_insert(&hctx, *pRowid, vec);
  sqlite3_free(vec);
  p->entry_point = hctx.entry_point;
  p->max_layer = hctx.max_layer;
  return rc;
}

/* ── xShadowName ─────────────────────────────────────────────────────────────
 */

static int vec0ShadowName(const char *zName) {
  static const char *azShadow[] = {"config", "data", "graph", "layers", NULL};
  for (int i = 0; azShadow[i]; i++) {
    if (strcmp(zName, azShadow[i]) == 0)
      return 1;
  }
  return 0;
}

/* ── Module definition ───────────────────────────────────────────────────────
 */

sqlite3_module vec0Module = {
    /* iVersion    */ 3,
    /* xCreate     */ vec0Create,
    /* xConnect    */ vec0Connect,
    /* xBestIndex  */ vec0BestIndex,
    /* xDisconnect */ vec0Disconnect,
    /* xDestroy    */ vec0Destroy,
    /* xOpen       */ vec0Open,
    /* xClose      */ vec0Close,
    /* xFilter     */ vec0Filter,
    /* xNext       */ vec0Next,
    /* xEof        */ vec0Eof,
    /* xColumn     */ vec0Column,
    /* xRowid      */ vec0Rowid,
    /* xUpdate     */ vec0Update,
    /* xBegin      */ NULL,
    /* xSync       */ NULL,
    /* xCommit     */ NULL,
    /* xRollback   */ NULL,
    /* xFindFunction */ NULL,
    /* xRename     */ NULL,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ vec0ShadowName,
    /* xIntegrity  */ NULL,
};
