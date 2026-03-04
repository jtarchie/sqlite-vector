#include "vtab.h"

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

  /* Look for a MATCH constraint on col 0 (the hidden table-name column).
   * WHERE vecs MATCH query  →  constraint on col 0, op =
   * SQLITE_INDEX_CONSTRAINT_MATCH */
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

  /* Full scan fallback (very expensive — discourages the planner from using it
   * without a MATCH constraint) */
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
    /* Full scan: not yet implemented — return empty */
    return SQLITE_OK;
  }

  /* kNN path: HNSW search will be implemented in a subsequent commit */
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
  case 1: /* vector TEXT — fetch from _data in a later commit */
    sqlite3_result_null(ctx);
    break;
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
    /* xUpdate     */ NULL, /* read-only until INSERT is implemented */
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
