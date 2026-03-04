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

  /* Cached prepared statements — prepared once in xCreate/xConnect,
   * finalized in xDisconnect/xDestroy.  Passed to hnsw_* via HnswCtx
   * to avoid per-call sqlite3_prepare_v2/sqlite3_finalize overhead. */
  sqlite3_stmt *sc_get_nbrs;  /* SELECT neighbor_id, distance FROM _graph WHERE
                                 layer=? AND node_id=? */
  sqlite3_stmt *sc_get_vec;   /* SELECT vector FROM _data WHERE id=? */
  sqlite3_stmt *sc_scan_ids;  /* SELECT id FROM _data ORDER BY id */
  sqlite3_stmt *sc_lookup_id; /* SELECT id FROM _data WHERE id=? */
  sqlite3_stmt
      *sc_ins_edge; /* INSERT OR REPLACE INTO _graph(...) VALUES(?,?,?,?) */
  sqlite3_stmt
      *sc_del_edges; /* DELETE FROM _graph WHERE layer=? AND node_id=? */
  sqlite3_stmt *sc_ins_layer; /* INSERT OR REPLACE INTO
                                 _layers(node_id,max_layer) VALUES(?,?) */
  sqlite3_stmt *sc_nbr_count; /* SELECT COUNT(*) FROM _graph WHERE layer=? AND
                                 node_id=? */
  sqlite3_stmt *sc_rev_nbrs;  /* SELECT DISTINCT node_id FROM _graph WHERE
                                 layer=? AND neighbor_id=? */

  /* ── Transaction state ─────────────────────────────────────────────────
   * Set by xBegin/cleared by xCommit or xRollback.  While in_txn==1,
   * config writes (count, entry_point, max_layer) are deferred: shadow-table
   * rows are written by SQLite's own transaction machinery, but the _config
   * UPDATE is suppressed until xCommit so that a ROLLBACK leaves _config
   * consistent with the rolled-back shadow tables. */
  int in_txn;                      /* non-zero inside a vtab transaction */
  sqlite3_int64 saved_count;       /* count at BEGIN time */
  sqlite3_int64 saved_entry_point; /* entry_point at BEGIN time */
  int saved_max_layer;             /* max_layer at BEGIN time */
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
  sqlite3_finalize(p->sc_get_nbrs);
  sqlite3_finalize(p->sc_get_vec);
  sqlite3_finalize(p->sc_scan_ids);
  sqlite3_finalize(p->sc_lookup_id);
  sqlite3_finalize(p->sc_ins_edge);
  sqlite3_finalize(p->sc_del_edges);
  sqlite3_finalize(p->sc_ins_layer);
  sqlite3_finalize(p->sc_nbr_count);
  sqlite3_finalize(p->sc_rev_nbrs);
  sqlite3_free(p);
}

/* Prepare shared statements and store them on the Vec0Table.
 * Called from vec0Init after name/schema are set and shadow tables exist. */
static int vec0_prepare_stmts(Vec0Table *p, char **pzErr) {
  char *sql;
  int rc;

#define PREP(field, fmt, ...)                                                  \
  do {                                                                         \
    sql = sqlite3_mprintf(fmt, ##__VA_ARGS__);                                 \
    rc = sqlite3_prepare_v2(p->db, sql, -1, &p->field, NULL);                  \
    sqlite3_free(sql);                                                         \
    if (rc != SQLITE_OK) {                                                     \
      if (pzErr)                                                               \
        *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));                 \
      return rc;                                                               \
    }                                                                          \
  } while (0)

  PREP(sc_get_nbrs,
       "SELECT neighbor_id, distance FROM \"%w\".\"%w_graph\""
       " WHERE layer=? AND node_id=?",
       p->schema, p->name);
  PREP(sc_get_vec, "SELECT vector FROM \"%w\".\"%w_data\" WHERE id=?",
       p->schema, p->name);
  PREP(sc_scan_ids, "SELECT id FROM \"%w\".\"%w_data\" ORDER BY id", p->schema,
       p->name);
  PREP(sc_lookup_id, "SELECT id FROM \"%w\".\"%w_data\" WHERE id=?", p->schema,
       p->name);
  PREP(sc_ins_edge,
       "INSERT OR REPLACE INTO \"%w\".\"%w_graph\""
       "(layer, node_id, neighbor_id, distance) VALUES (?,?,?,?)",
       p->schema, p->name);
  PREP(sc_del_edges,
       "DELETE FROM \"%w\".\"%w_graph\" WHERE layer=? AND node_id=?", p->schema,
       p->name);
  PREP(sc_ins_layer,
       "INSERT OR REPLACE INTO \"%w\".\"%w_layers\"(node_id, max_layer)"
       " VALUES (?,?)",
       p->schema, p->name);
  PREP(sc_nbr_count,
       "SELECT COUNT(*) FROM \"%w\".\"%w_graph\" WHERE layer=? AND node_id=?",
       p->schema, p->name);
  PREP(sc_rev_nbrs,
       "SELECT DISTINCT node_id FROM \"%w\".\"%w_graph\""
       " WHERE layer=? AND neighbor_id=?",
       p->schema, p->name);

#undef PREP
  return SQLITE_OK;
}

/* Populate an HnswCtx from the Vec0Table, including cached stmt pointers. */
static void vec0_make_hctx(Vec0Table *p, HnswCtx *out) {
  out->db = p->db;
  out->schema = p->schema;
  out->tbl_name = p->name;
  out->dims = p->dims;
  out->m = p->m;
  out->ef_construction = p->ef_construction;
  out->ef_search = p->ef_search;
  out->entry_point = p->entry_point;
  out->max_layer = p->max_layer;
  out->dist_fn = p->dist_fn;
  out->sc_get_nbrs = p->sc_get_nbrs;
  out->sc_get_vec = p->sc_get_vec;
  out->sc_ins_edge = p->sc_ins_edge;
  out->sc_del_edges = p->sc_del_edges;
  out->sc_ins_layer = p->sc_ins_layer;
  out->sc_nbr_count = p->sc_nbr_count;
  out->sc_rev_nbrs = p->sc_rev_nbrs;
  /* Suppress _config DB writes inside a vtab transaction; xCommit flushes. */
  out->defer_config = p->in_txn;
}

/* ── Shadow table helpers ─────────────────────────────────────────────────
 */

/* Declare the virtual table schema.  Must be called from both xCreate and
 * xConnect before returning. */
static int vec0_declare(sqlite3 *db, const char *name, char **pzErr) {
  char *schema = sqlite3_mprintf(
      "CREATE TABLE x(\"%w\" HIDDEN, vector TEXT, distance REAL HIDDEN,"
      " ef_search INTEGER HIDDEN)",
      name);
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

  /* Prepare shared HNSW statements (shadow tables must exist at this point) */
  {
    int rc = vec0_prepare_stmts(p, pzErr);
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

/* ── Operator-alias distance scalar ──────────────────────────────────────────
 * Used as the *pxFunc returned by xFindFunction for <->, <=>, <#>, <+>.
 * sqlite3_user_data() carries the metric name string (static literal).
 */
static void op_dist_fn(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;
  const char *metric = (const char *)sqlite3_user_data(ctx);
  dist_fn_t fn = distance_for_metric(metric);
  if (!fn) {
    sqlite3_result_error(ctx, "vec0: unknown metric", -1);
    return;
  }
  if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }
  const char *ta = (const char *)sqlite3_value_text(argv[0]);
  const char *tb = (const char *)sqlite3_value_text(argv[1]);
  float *va = NULL, *vb = NULL;
  int da = 0, db = 0;
  char *err = NULL;
  if (vec_parse(ta, &va, &da, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : "vec0: invalid vector", -1);
    sqlite3_free(err);
    return;
  }
  if (vec_parse(tb, &vb, &db, &err) != SQLITE_OK) {
    sqlite3_free(va);
    sqlite3_result_error(ctx, err ? err : "vec0: invalid vector", -1);
    sqlite3_free(err);
    return;
  }
  if (da != db) {
    sqlite3_free(va);
    sqlite3_free(vb);
    sqlite3_result_error(ctx, "vec0: dimension mismatch", -1);
    return;
  }
  double dist = 0.0;
  fn(va, vb, da, &dist);
  sqlite3_free(va);
  sqlite3_free(vb);
  sqlite3_result_double(ctx, dist);
}

/* ── xBestIndex ──────────────────────────────────────────────────────────────
 */

static int vec0BestIndex(sqlite3_vtab *pVtab, sqlite3_index_info *pInfo) {
  (void)pVtab;

/* Helper: look for an EQ constraint on col 3 (ef_search HIDDEN) and, if
 * found, assign it argvIndex 2 so xFilter receives it as argv[1]. */
#define ASSIGN_EF_SEARCH_ARG()                                                 \
  do {                                                                         \
    for (int _j = 0; _j < pInfo->nConstraint; _j++) {                          \
      if (!pInfo->aConstraint[_j].usable)                                      \
        continue;                                                              \
      if (pInfo->aConstraint[_j].iColumn == 3 &&                               \
          pInfo->aConstraint[_j].op == SQLITE_INDEX_CONSTRAINT_EQ) {           \
        pInfo->aConstraintUsage[_j].argvIndex = 2;                             \
        pInfo->aConstraintUsage[_j].omit = 1;                                  \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  } while (0)

  /* 0. Operator-alias kNN constraints (op 151-156, set by xFindFunction). */
  for (int i = 0; i < pInfo->nConstraint; i++) {
    if (!pInfo->aConstraint[i].usable)
      continue;
    int op = pInfo->aConstraint[i].op;
    if (op >= 151 && op <= 156) {
      pInfo->aConstraintUsage[i].argvIndex = 1;
      pInfo->aConstraintUsage[i].omit = 1;
      ASSIGN_EF_SEARCH_ARG();
      pInfo->idxNum =
          op; /* 151=l2 152=cosine 153=ip 154=l1 155=hamming 156=jaccard */
      pInfo->estimatedCost = 10.0;
      pInfo->estimatedRows = 10;
      return SQLITE_OK;
    }
  }

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
      ASSIGN_EF_SEARCH_ARG();
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

static void vec0_cursor_clear_results(Vec0Cursor *cur) {
  sqlite3_free(cur->rowids);
  sqlite3_free(cur->distances);
  cur->rowids = NULL;
  cur->distances = NULL;
  cur->nResults = 0;
  cur->pos = 0;
}

static void vec0_stmt_rewind(sqlite3_stmt *stmt) {
  if (!stmt)
    return;
  sqlite3_reset(stmt);
  sqlite3_clear_bindings(stmt);
}

static int vec0_filter_fullscan(Vec0Cursor *cur) {
  Vec0Table *p = (Vec0Table *)cur->base.pVtab;
  if (!p->sc_scan_ids)
    return SQLITE_OK; /* return empty on error */

  vec0_stmt_rewind(p->sc_scan_ids);

  int cap = 16;
  int n = 0;
  cur->rowids = sqlite3_malloc(cap * (int)sizeof(int64_t));
  if (!cur->rowids) {
    vec0_stmt_rewind(p->sc_scan_ids);
    return SQLITE_NOMEM;
  }

  int step_rc;
  while ((step_rc = sqlite3_step(p->sc_scan_ids)) == SQLITE_ROW) {
    if (n == cap) {
      cap *= 2;
      int64_t *tmp = sqlite3_realloc(cur->rowids, cap * (int)sizeof(int64_t));
      if (!tmp) {
        vec0_stmt_rewind(p->sc_scan_ids);
        return SQLITE_NOMEM;
      }
      cur->rowids = tmp;
    }
    cur->rowids[n++] = sqlite3_column_int64(p->sc_scan_ids, 0);
  }

  vec0_stmt_rewind(p->sc_scan_ids);
  if (step_rc != SQLITE_DONE) {
    sqlite3_free(cur->rowids);
    cur->rowids = NULL;
    cur->nResults = 0;
    return SQLITE_ERROR;
  }

  cur->nResults = n;
  return SQLITE_OK;
}

static int vec0_filter_rowid_eq(Vec0Cursor *cur, int argc,
                                sqlite3_value **argv) {
  Vec0Table *p = (Vec0Table *)cur->base.pVtab;
  if (argc < 1 || !p->sc_lookup_id)
    return SQLITE_OK;

  sqlite3_int64 target = sqlite3_value_int64(argv[0]);
  vec0_stmt_rewind(p->sc_lookup_id);
  sqlite3_bind_int64(p->sc_lookup_id, 1, target);

  int step_rc = sqlite3_step(p->sc_lookup_id);
  if (step_rc == SQLITE_ROW) {
    cur->rowids = sqlite3_malloc((int)sizeof(int64_t));
    if (!cur->rowids) {
      vec0_stmt_rewind(p->sc_lookup_id);
      return SQLITE_NOMEM;
    }
    cur->rowids[0] = sqlite3_column_int64(p->sc_lookup_id, 0);
    cur->nResults = 1;
  } else if (step_rc != SQLITE_DONE) {
    vec0_stmt_rewind(p->sc_lookup_id);
    return SQLITE_ERROR;
  }
  vec0_stmt_rewind(p->sc_lookup_id);
  return SQLITE_OK;
}

static int vec0_filter_knn(Vec0Cursor *cur, int idxNum, int argc,
                           sqlite3_value **argv) {
  Vec0Table *p = (Vec0Table *)cur->base.pVtab;

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
    cur->base.pVtab->zErrMsg = sqlite3_mprintf(
        "vec0: query has %d dims, table has %d", query_dims, p->dims);
    return SQLITE_ERROR;
  }

  int ef = p->ef_search;
  if (argc >= 2 && sqlite3_value_type(argv[1]) == SQLITE_INTEGER) {
    int v = sqlite3_value_int(argv[1]);
    if (v > 0)
      ef = v;
  }
  int k = ef;

  dist_fn_t knn_dist_fn = p->dist_fn;
  if (idxNum >= 151 && idxNum <= 156) {
    static const char *op_metrics[] = {"l2", "cosine",  "ip",
                                       "l1", "hamming", "jaccard"};
    dist_fn_t override = distance_for_metric(op_metrics[idxNum - 151]);
    if (override)
      knn_dist_fn = override;
  }

  HnswCtx hctx;
  vec0_make_hctx(p, &hctx);
  hctx.ef_search = ef;
  hctx.dist_fn = knn_dist_fn;

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

/* ── xFilter ─────────────────────────────────────────────────────────────────
 */

static int vec0Filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
  (void)idxStr;
  (void)argc;
  (void)argv;

  Vec0Cursor *cur = (Vec0Cursor *)pCursor;
  vec0_cursor_clear_results(cur);

  if (idxNum == 0)
    return vec0_filter_fullscan(cur);

  if (idxNum == 2)
    return vec0_filter_rowid_eq(cur, argc, argv);

  if (idxNum != 1 && (idxNum < 151 || idxNum > 156))
    return SQLITE_OK;

  return vec0_filter_knn(cur, idxNum, argc, argv);
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

static void vec0_result_vector_text(Vec0Cursor *cur, sqlite3_context *ctx) {
  Vec0Table *p = (Vec0Table *)cur->base.pVtab;
  if (p->sc_get_vec) {
    vec0_stmt_rewind(p->sc_get_vec);
    sqlite3_bind_int64(p->sc_get_vec, 1, cur->rowids[cur->pos]);
    int step_rc = sqlite3_step(p->sc_get_vec);
    if (step_rc == SQLITE_ROW) {
      int bytes = sqlite3_column_bytes(p->sc_get_vec, 0);
      const float *blob = (const float *)sqlite3_column_blob(p->sc_get_vec, 0);
      int dims = bytes / (int)sizeof(float);
      char *text = vec_format(blob, dims);
      if (text)
        sqlite3_result_text(ctx, text, -1, sqlite3_free);
      else
        sqlite3_result_null(ctx);
    } else {
      sqlite3_result_null(ctx);
    }
    vec0_stmt_rewind(p->sc_get_vec);
    return;
  }

  char *sql =
      sqlite3_mprintf("SELECT vector FROM \"%w\".\"%w_data\" WHERE id=%lld",
                      p->schema, p->name, cur->rowids[cur->pos]);
  sqlite3_stmt *stmt = NULL;
  if (!sql || sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL) != SQLITE_OK) {
    sqlite3_free(sql);
    sqlite3_result_null(ctx);
    return;
  }
  sqlite3_free(sql);
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    int bytes = sqlite3_column_bytes(stmt, 0);
    const float *blob = (const float *)sqlite3_column_blob(stmt, 0);
    int dims = bytes / (int)sizeof(float);
    char *text = vec_format(blob, dims);
    if (text)
      sqlite3_result_text(ctx, text, -1, sqlite3_free);
    else
      sqlite3_result_null(ctx);
  } else {
    sqlite3_result_null(ctx);
  }
  sqlite3_finalize(stmt);
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
  case 1: { /* vector TEXT */
    vec0_result_vector_text(cur, ctx);
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
 *
 * Shadow-table atomicity: SQLite automatically creates a statement-level
 * savepoint before calling xUpdate.  If xUpdate returns an error, SQLite
 * rolls back all shadow-table writes made during this callback.  We only
 * need to restore the in-memory Vec0Table fields (count, entry_point,
 * max_layer) on error paths.
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

    /* Remove from HNSW graph — check return code (was previously ignored). */
    HnswCtx hctx;
    vec0_make_hctx(p, &hctx);
    int drc = hnsw_delete(&hctx, del_rowid);
    if (drc != SQLITE_OK) {
      /* SQLite auto-rollback undoes the _data DELETE and any graph writes.
       * In-memory state is still unchanged (p fields not yet modified). */
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return drc;
    }
    p->entry_point = hctx.entry_point;
    p->max_layer = hctx.max_layer;

    if (p->count > 0)
      p->count--;
    if (!p->in_txn) {
      char *csql = sqlite3_mprintf(
          "UPDATE \"%w\".\"%w_config\" SET value='%lld' WHERE key='count'",
          p->schema, p->name, p->count);
      sqlite3_exec(p->db, csql, NULL, NULL, NULL);
      sqlite3_free(csql);
    }
    return SQLITE_OK;
  }

  /* UPDATE: argv[0] is not NULL (existing row being replaced) */
  if (sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    sqlite3_int64 old_rowid = sqlite3_value_int64(argv[0]);
    sqlite3_int64 new_rowid = sqlite3_value_int64(argv[1]);
    if (new_rowid != old_rowid) {
      pVtab->zErrMsg = sqlite3_mprintf("vec0: cannot change rowid");
      return SQLITE_MISMATCH;
    }
    if (sqlite3_value_type(argv[3]) == SQLITE_NULL) {
      pVtab->zErrMsg = sqlite3_mprintf("vec0: vector must not be NULL");
      return SQLITE_CONSTRAINT;
    }
    const char *upd_text = (const char *)sqlite3_value_text(argv[3]);
    float *upd_vec = NULL;
    int upd_dims = 0;
    char *upd_err = NULL;
    int upd_rc = vec_parse(upd_text, &upd_vec, &upd_dims, &upd_err);
    if (upd_rc != SQLITE_OK) {
      pVtab->zErrMsg =
          upd_err ? upd_err : sqlite3_mprintf("vec0: invalid vector");
      return SQLITE_ERROR;
    }
    if (upd_dims != p->dims) {
      sqlite3_free(upd_vec);
      pVtab->zErrMsg =
          sqlite3_mprintf("vec0: expected %d dims, got %d", p->dims, upd_dims);
      return SQLITE_CONSTRAINT;
    }

    /* Save pre-state so we can restore in-memory fields on error. */
    sqlite3_int64 pre_count = p->count;
    sqlite3_int64 pre_ep = p->entry_point;
    int pre_ml = p->max_layer;

    /* Delete old row from _data and HNSW graph */
    char *dsql = sqlite3_mprintf("DELETE FROM \"%w\".\"%w_data\" WHERE id=%lld",
                                 p->schema, p->name, old_rowid);
    sqlite3_exec(p->db, dsql, NULL, NULL, NULL);
    sqlite3_free(dsql);

    HnswCtx del_ctx;
    vec0_make_hctx(p, &del_ctx);
    upd_rc = hnsw_delete(&del_ctx, old_rowid);
    if (upd_rc != SQLITE_OK) {
      /* SQLite auto-rollback undoes shadow writes; restore in-memory state. */
      p->count = pre_count;
      p->entry_point = pre_ep;
      p->max_layer = pre_ml;
      sqlite3_free(upd_vec);
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return upd_rc;
    }
    p->entry_point = del_ctx.entry_point;
    p->max_layer = del_ctx.max_layer;
    if (p->count > 0)
      p->count--;

    /* Re-insert with same rowid */
    char *ins_sql = sqlite3_mprintf(
        "INSERT INTO \"%w\".\"%w_data\" (id, vector) VALUES (%lld, ?)",
        p->schema, p->name, old_rowid);
    sqlite3_stmt *ins_stmt = NULL;
    upd_rc = sqlite3_prepare_v2(p->db, ins_sql, -1, &ins_stmt, NULL);
    sqlite3_free(ins_sql);
    if (upd_rc != SQLITE_OK) {
      p->count = pre_count;
      p->entry_point = pre_ep;
      p->max_layer = pre_ml;
      sqlite3_free(upd_vec);
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return upd_rc;
    }
    sqlite3_bind_blob(ins_stmt, 1, upd_vec, upd_dims * (int)sizeof(float),
                      SQLITE_STATIC);
    upd_rc = sqlite3_step(ins_stmt);
    sqlite3_finalize(ins_stmt);
    if (upd_rc != SQLITE_DONE) {
      p->count = pre_count;
      p->entry_point = pre_ep;
      p->max_layer = pre_ml;
      sqlite3_free(upd_vec);
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return SQLITE_ERROR;
    }
    p->count++;
    *pRowid = old_rowid;

    /* Re-insert into HNSW graph */
    HnswCtx ins_ctx;
    vec0_make_hctx(p, &ins_ctx);
    /* entry_point/max_layer were updated by hnsw_delete above */
    ins_ctx.entry_point = p->entry_point;
    ins_ctx.max_layer = p->max_layer;
    upd_rc = hnsw_insert(&ins_ctx, old_rowid, upd_vec);
    sqlite3_free(upd_vec);
    if (upd_rc != SQLITE_OK) {
      /* count: net of delete+insert is still pre_count */
      p->count = pre_count;
      p->entry_point = pre_ep;
      p->max_layer = pre_ml;
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return upd_rc;
    }
    p->entry_point = ins_ctx.entry_point;
    p->max_layer = ins_ctx.max_layer;
    /* count is unchanged net (delete+reinsert); write config if not deferred */
    if (!p->in_txn) {
      char *cnt_sql =
          sqlite3_mprintf("UPDATE \"%w\".\"%w_config\" SET value='%lld'"
                          " WHERE key='count'",
                          p->schema, p->name, p->count);
      sqlite3_exec(p->db, cnt_sql, NULL, NULL, NULL);
      sqlite3_free(cnt_sql);
    }
    return SQLITE_OK;
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

  /* Insert raw float BLOB into _data.
   * SQLite's automatic statement savepoint ensures that if hnsw_insert later
   * fails and we return an error, the _data row and any partial graph writes
   * are rolled back automatically by SQLite. */
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

  /* Insert into HNSW graph */
  HnswCtx hctx;
  vec0_make_hctx(p, &hctx);
  rc = hnsw_insert(&hctx, *pRowid, vec);
  sqlite3_free(vec);
  if (rc != SQLITE_OK) {
    /* SQLite auto-rollback undoes the _data insert and any partial graph
     * writes.  Undo the in-memory count increment. */
    p->count--;
    pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }
  p->entry_point = hctx.entry_point;
  p->max_layer = hctx.max_layer;

  /* Keep count in _config in sync (deferred to xCommit when in_txn) */
  if (!p->in_txn) {
    sql = sqlite3_mprintf(
        "UPDATE \"%w\".\"%w_config\" SET value = '%lld' WHERE key = 'count'",
        p->schema, p->name, p->count);
    sqlite3_exec(p->db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
  }
  return SQLITE_OK;
}

/* ── xFindFunction ───────────────────────────────────────────────────────────
 * Intercept vec_distance_* calls on a vtab column so SQLite routes them
 * through xBestIndex as index-accelerated kNN scans.
 * Returns a nonzero code that becomes aConstraint[i].op in xBestIndex.
 *   151 = vec_distance_l2      (L2 / Euclidean)       <->
 *   152 = vec_distance_cosine  (cosine distance)       <=>
 *   153 = vec_distance_ip      (inner product)         <#>
 *   154 = vec_distance_l1      (L1 / Manhattan)        <+>
 *   155 = vec_distance_hamming (Hamming, binary vecs)  <~>
 *   156 = vec_distance_jaccard (Jaccard, binary vecs)  <%>
 *
 * When used on regular (non-vtab) columns the pre-registered scalars in
 * distance.c handle the call normally — xFindFunction is never invoked.
 */
static int vec0FindFunction(sqlite3_vtab *pVtab, int nArg, const char *zName,
                            void (**pxFunc)(sqlite3_context *, int,
                                            sqlite3_value **),
                            void **ppArg) {
  (void)pVtab;
  if (nArg != 2)
    return 0;
  if (strcmp(zName, "vec_distance_l2") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"l2";
    return 151;
  }
  if (strcmp(zName, "vec_distance_cosine") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"cosine";
    return 152;
  }
  if (strcmp(zName, "vec_distance_ip") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"ip";
    return 153;
  }
  if (strcmp(zName, "vec_distance_l1") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"l1";
    return 154;
  }
  if (strcmp(zName, "vec_distance_hamming") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"hamming";
    return 155;
  }
  if (strcmp(zName, "vec_distance_jaccard") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"jaccard";
    return 156;
  }
  return 0;
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

/* ── xBegin / xCommit / xRollback ────────────────────────────────────────────
 *
 * xBegin: snapshot in-memory config state so xRollback can restore it.
 * xCommit: flush the (possibly updated) in-memory count/entry_point/max_layer
 *          to _config in a single shot.  During the transaction these were
 *          suppressed (defer_config=1) so they were not written row-by-row.
 * xRollback: the shadow-table writes were rolled back by SQLite automatically;
 *            restore in-memory state to match.
 */
static int vec0Begin(sqlite3_vtab *pVtab) {
  Vec0Table *p = (Vec0Table *)pVtab;
  p->in_txn = 1;
  p->saved_count = p->count;
  p->saved_entry_point = p->entry_point;
  p->saved_max_layer = p->max_layer;
  return SQLITE_OK;
}

static int vec0Commit(sqlite3_vtab *pVtab) {
  Vec0Table *p = (Vec0Table *)pVtab;
  char *sql;
  /* Write all three mutable config values in one go. */
  sql = sqlite3_mprintf("UPDATE \"%w\".\"%w_config\" SET value='%lld'"
                        " WHERE key='count'",
                        p->schema, p->name, p->count);
  sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  sql = sqlite3_mprintf("UPDATE \"%w\".\"%w_config\" SET value='%lld'"
                        " WHERE key='entry_point'",
                        p->schema, p->name, p->entry_point);
  sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  sql = sqlite3_mprintf("UPDATE \"%w\".\"%w_config\" SET value='%d'"
                        " WHERE key='max_layer'",
                        p->schema, p->name, p->max_layer);
  sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  p->in_txn = 0;
  return SQLITE_OK;
}

static int vec0Rollback(sqlite3_vtab *pVtab) {
  Vec0Table *p = (Vec0Table *)pVtab;
  /* Shadow-table writes were already rolled back by SQLite.
   * Restore in-memory state to match. */
  p->count = p->saved_count;
  p->entry_point = p->saved_entry_point;
  p->max_layer = p->saved_max_layer;
  p->in_txn = 0;
  return SQLITE_OK;
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
    /* xBegin      */ vec0Begin,
    /* xSync       */ NULL,
    /* xCommit     */ vec0Commit,
    /* xRollback   */ vec0Rollback,
    /* xFindFunction */ vec0FindFunction,
    /* xRename     */ NULL,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ vec0ShadowName,
    /* xIntegrity  */ NULL,
};
