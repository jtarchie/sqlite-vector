#include "vtab.h"
#include "distance.h"
#include "hnsw.h"
#include "vec_parse.h"

#include <sqlite3ext.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* Maximum vector dimensionality.  8192 × 4 = 32 KiB per float32 vector,
 * which fits comfortably in a single SQLite page at any page_size ≥ 64 KiB
 * and keeps per-row allocations reasonable. */
#define VEC0_MAX_DIMS 8192

/* Maximum value accepted in a LIMIT clause on a kNN query.
 * Caps the result-set memory to 4096 × 16 bytes = 64 KiB per cursor. */
#define VEC0_MAX_LIMIT 4096

/* ── Operator codes returned by xFindFunction ────────────────────────────
 * When vec_distance_*() is called on a virtual-table column, xFindFunction
 * returns one of these codes.  SQLite stores the value as aConstraint[i].op
 * and xBestIndex reads it to determine which distance function the query
 * requested.  The base values encode the metric; higher flag bits are OR'd
 * in to signal optional per-query constraints (ef_search, LIMIT, distance
 * thresholds). */
#define VEC0_OP_DISTANCE_L2 151      /* vec_distance_l2      <-> */
#define VEC0_OP_DISTANCE_COSINE 152  /* vec_distance_cosine  <=> */
#define VEC0_OP_DISTANCE_IP 153      /* vec_distance_ip      <#> */
#define VEC0_OP_DISTANCE_L1 154      /* vec_distance_l1      <+> */
#define VEC0_OP_DISTANCE_HAMMING 155 /* vec_distance_hamming <~> */
#define VEC0_OP_DISTANCE_JACCARD 156 /* vec_distance_jaccard <%> */

/* ── idxNum flag bits ────────────────────────────────────────────────────
 * OR'd onto the base idxNum (1=MATCH, 2=rowid-eq, 151-156=operator) to
 * signal which optional constraints are present in argv[]. */
#define VEC0_IDX_FLAG_EF 0x1000       /* ef_search override in argv */
#define VEC0_IDX_FLAG_LIMIT 0x2000    /* explicit LIMIT in argv */
#define VEC0_IDX_FLAG_DIST_LT 0x4000  /* distance < threshold */
#define VEC0_IDX_FLAG_DIST_LE 0x8000  /* distance <= threshold */
#define VEC0_IDX_FLAG_DIST_GT 0x10000 /* distance > threshold */
#define VEC0_IDX_FLAG_DIST_GE 0x20000 /* distance >= threshold */

/* Mask to strip all flag bits and recover the base idxNum. */
#define VEC0_IDX_BASE_MASK                                                     \
  ~(VEC0_IDX_FLAG_EF | VEC0_IDX_FLAG_LIMIT | VEC0_IDX_FLAG_DIST_LT |           \
    VEC0_IDX_FLAG_DIST_LE | VEC0_IDX_FLAG_DIST_GT | VEC0_IDX_FLAG_DIST_GE)

/* ── Internal structs ────────────────────────────────────────────────────────
 */

typedef struct Vec0Table Vec0Table;
struct Vec0Table {
  sqlite3_vtab base; /* Must be first */
  sqlite3 *db;
  char *name;       /* Virtual table name (for shadow table names) */
  char *schema;     /* Schema name */
  int dims;         /* Vector dimensionality */
  char *metric;     /* Distance metric name */
  int storage_type; /* VEC_STORAGE_F32/INT8/BIT */
  int m;            /* HNSW M parameter */
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

  /* Additional cached statements for hnsw_delete and config updates */
  sqlite3_stmt *sc_get_layer;      /* SELECT max_layer FROM _layers WHERE
                                      node_id=? */
  sqlite3_stmt *sc_del_layer;      /* DELETE FROM _layers WHERE node_id=? */
  sqlite3_stmt *sc_del_node_edges; /* DELETE FROM _graph WHERE node_id=? OR
                                      neighbor_id=? */
  sqlite3_stmt *sc_elect_ep;       /* SELECT node_id, max_layer FROM _layers
                                      ORDER BY max_layer DESC LIMIT 1 */
  sqlite3_stmt *sc_upd_config;     /* UPDATE _config SET value=? WHERE key=? */

  /* Cached statements for _data table operations */
  sqlite3_stmt *sc_ins_data;    /* INSERT INTO _data (vector) VALUES (?) */
  sqlite3_stmt *sc_ins_data_id; /* INSERT INTO _data (id, vector) VALUES
                                   (?,?) */
  sqlite3_stmt *sc_del_data;    /* DELETE FROM _data WHERE id=? */

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

/* Convert float vector to the native storage type.
 * Returns a newly allocated blob (caller must sqlite3_free) and sets *out_bytes
 * to the blob size.  For VEC_STORAGE_F32, returns NULL (caller should use the
 * float array directly).
 *
 * INT8 clamping: values outside [-128, 127] are clamped to the range boundary
 * with rounding (±0.5) so that e.g. 127.3 → 127, -128.7 → -128.
 *
 * BIT packing: each float > 0 becomes a 1-bit; MSB-first within each byte.
 * Formula: byte_count = (dims + 7) / 8  (ceiling division by 8). */
static void *vec_convert_to_storage(const float *fvec, int dims,
                                    int storage_type, int *out_bytes) {
  if (storage_type == VEC_STORAGE_INT8) {
    int nbytes = dims;
    int8_t *buf = sqlite3_malloc(nbytes);
    if (!buf)
      return NULL;
    for (int i = 0; i < dims; i++) {
      float v = fvec[i];
      if (v < -128.0f)
        v = -128.0f;
      if (v > 127.0f)
        v = 127.0f;
      buf[i] = (int8_t)(v < 0 ? v - 0.5f : v + 0.5f);
    }
    *out_bytes = nbytes;
    return buf;
  }
  if (storage_type == VEC_STORAGE_BIT) {
    int nbytes = (dims + 7) / 8;
    uint8_t *buf = sqlite3_malloc(nbytes);
    if (!buf)
      return NULL;
    memset(buf, 0, nbytes);
    for (int i = 0; i < dims; i++) {
      if (fvec[i] > 0.0f)
        buf[i / 8] |= (uint8_t)(1 << (7 - (i % 8)));
    }
    *out_bytes = nbytes;
    return buf;
  }
  /* VEC_STORAGE_F32 — signal caller to use float array directly */
  *out_bytes = dims * (int)sizeof(float);
  return NULL;
}

static void vec0_finalize_stmts(Vec0Table *p) {
  sqlite3_finalize(p->sc_get_nbrs);
  sqlite3_finalize(p->sc_get_vec);
  sqlite3_finalize(p->sc_scan_ids);
  sqlite3_finalize(p->sc_lookup_id);
  sqlite3_finalize(p->sc_ins_edge);
  sqlite3_finalize(p->sc_del_edges);
  sqlite3_finalize(p->sc_ins_layer);
  sqlite3_finalize(p->sc_nbr_count);
  sqlite3_finalize(p->sc_rev_nbrs);
  sqlite3_finalize(p->sc_get_layer);
  sqlite3_finalize(p->sc_del_layer);
  sqlite3_finalize(p->sc_del_node_edges);
  sqlite3_finalize(p->sc_elect_ep);
  sqlite3_finalize(p->sc_upd_config);
  sqlite3_finalize(p->sc_ins_data);
  sqlite3_finalize(p->sc_ins_data_id);
  sqlite3_finalize(p->sc_del_data);
  p->sc_get_nbrs = NULL;
  p->sc_get_vec = NULL;
  p->sc_scan_ids = NULL;
  p->sc_lookup_id = NULL;
  p->sc_ins_edge = NULL;
  p->sc_del_edges = NULL;
  p->sc_ins_layer = NULL;
  p->sc_nbr_count = NULL;
  p->sc_rev_nbrs = NULL;
  p->sc_get_layer = NULL;
  p->sc_del_layer = NULL;
  p->sc_del_node_edges = NULL;
  p->sc_elect_ep = NULL;
  p->sc_upd_config = NULL;
  p->sc_ins_data = NULL;
  p->sc_ins_data_id = NULL;
  p->sc_del_data = NULL;
}

static void vec0_free(Vec0Table *p) {
  if (!p)
    return;
  sqlite3_free(p->name);
  sqlite3_free(p->schema);
  sqlite3_free(p->metric);
  vec0_finalize_stmts(p);
  sqlite3_free(p);
}

/* Prepare shared statements and store them on the Vec0Table.
 * Called from vec0Init after name/schema are set and shadow tables exist.
 *
 * The PREP() macro below formats SQL via sqlite3_mprintf, compiles it with
 * sqlite3_prepare_v2, and stores the result on the Vec0Table field.  On any
 * failure it sets *pzErr and returns immediately — the caller (vec0Init) is
 * responsible for calling vec0_free() which finalizes all non-NULL stmts. */
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
  PREP(sc_get_layer,
       "SELECT max_layer FROM \"%w\".\"%w_layers\" WHERE node_id=?", p->schema,
       p->name);
  PREP(sc_del_layer, "DELETE FROM \"%w\".\"%w_layers\" WHERE node_id=?",
       p->schema, p->name);
  PREP(sc_del_node_edges,
       "DELETE FROM \"%w\".\"%w_graph\" WHERE node_id=? OR neighbor_id=?",
       p->schema, p->name);
  PREP(sc_elect_ep,
       "SELECT node_id, max_layer FROM \"%w\".\"%w_layers\""
       " ORDER BY max_layer DESC LIMIT 1",
       p->schema, p->name);
  PREP(sc_upd_config, "UPDATE \"%w\".\"%w_config\" SET value=? WHERE key=?",
       p->schema, p->name);
  PREP(sc_ins_data, "INSERT INTO \"%w\".\"%w_data\" (vector) VALUES (?)",
       p->schema, p->name);
  PREP(sc_ins_data_id,
       "INSERT INTO \"%w\".\"%w_data\" (id, vector) VALUES (?,?)", p->schema,
       p->name);
  PREP(sc_del_data, "DELETE FROM \"%w\".\"%w_data\" WHERE id=?", p->schema,
       p->name);

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
  out->storage_type = p->storage_type;
  out->dist_fn = p->dist_fn;
  out->sc_get_nbrs = p->sc_get_nbrs;
  out->sc_get_vec = p->sc_get_vec;
  out->sc_ins_edge = p->sc_ins_edge;
  out->sc_del_edges = p->sc_del_edges;
  out->sc_ins_layer = p->sc_ins_layer;
  out->sc_nbr_count = p->sc_nbr_count;
  out->sc_rev_nbrs = p->sc_rev_nbrs;
  out->sc_get_layer = p->sc_get_layer;
  out->sc_del_layer = p->sc_del_layer;
  out->sc_del_node_edges = p->sc_del_node_edges;
  out->sc_elect_ep = p->sc_elect_ep;
  out->sc_upd_config = p->sc_upd_config;
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

/* Ensure shadow-table indexes exist for both new and pre-existing vec0 tables.
 * This runs in xCreate and xConnect so existing databases get online index
 * migration without manual intervention. */
static int vec0_ensure_shadow_indexes(Vec0Table *p, char **pzErr) {
  char *sql =
      sqlite3_mprintf("CREATE INDEX IF NOT EXISTS \"%w_graph_neighbor_idx\""
                      " ON \"%w_graph\"(layer, neighbor_id)",
                      p->name, p->name);
  int rc = sqlite3_exec(p->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK && pzErr) {
    *pzErr = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
  }
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
                        "('type','%q'),"
                        "('entry_point','-1'),('max_layer','0'),('count','0')",
                        s, n, p->dims, p->metric, p->m, p->ef_construction,
                        p->ef_search,
                        p->storage_type == VEC_STORAGE_INT8  ? "int8"
                        : p->storage_type == VEC_STORAGE_BIT ? "binary"
                                                             : "float32");
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
    else if (strcmp(key, "type") == 0) {
      if (strcmp(val, "int8") == 0)
        p->storage_type = VEC_STORAGE_INT8;
      else if (strcmp(val, "binary") == 0 || strcmp(val, "bit") == 0)
        p->storage_type = VEC_STORAGE_BIT;
      else
        p->storage_type = VEC_STORAGE_F32;
    }
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
      } else if (parse_arg(argv[i], "type", &val)) {
        if (strcmp(val, "float32") == 0 || strcmp(val, "f32") == 0)
          p->storage_type = VEC_STORAGE_F32;
        else if (strcmp(val, "int8") == 0 || strcmp(val, "i8") == 0)
          p->storage_type = VEC_STORAGE_INT8;
        else if (strcmp(val, "binary") == 0 || strcmp(val, "bit") == 0)
          p->storage_type = VEC_STORAGE_BIT;
        else {
          *pzErr = sqlite3_mprintf("vec0: unknown type '%s'", val);
          sqlite3_free(val);
          vec0_free(p);
          return SQLITE_ERROR;
        }
        sqlite3_free(val);
      }
    }

    if (p->dims <= 0) {
      *pzErr = sqlite3_mprintf("vec0: dims=N is required and must be > 0");
      vec0_free(p);
      return SQLITE_ERROR;
    }
    if (p->dims > VEC0_MAX_DIMS) {
      *pzErr = sqlite3_mprintf("vec0: dims must be <= %d", VEC0_MAX_DIMS);
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

  if (p->dims <= 0 || p->dims > VEC0_MAX_DIMS) {
    *pzErr = sqlite3_mprintf("vec0: invalid dims %d (must be 1..%d)", p->dims,
                             VEC0_MAX_DIMS);
    vec0_free(p);
    return SQLITE_ERROR;
  }

  /* Resolve distance kernel from metric string + storage type */
  p->dist_fn = distance_for_metric_typed(p->metric, p->storage_type);
  if (!p->dist_fn) {
    *pzErr = sqlite3_mprintf("vec0: metric '%s' not supported for type '%s'",
                             p->metric,
                             p->storage_type == VEC_STORAGE_INT8  ? "int8"
                             : p->storage_type == VEC_STORAGE_BIT ? "binary"
                                                                  : "float32");
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

  {
    int rc = vec0_ensure_shadow_indexes(p, pzErr);
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
  /* Finalize prepared statements before dropping the shadow tables they
   * reference — SQLite cannot drop tables with outstanding statements. */
  vec0_finalize_stmts(p);
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

  /* Helpers: optional ef_search hidden-column EQ and LIMIT constraints.
   * If present on a kNN plan, they are passed to xFilter as extra argv. */
  int flags = 0;
  int nextArg = 2;
  for (int j = 0; j < pInfo->nConstraint; j++) {
    if (!pInfo->aConstraint[j].usable)
      continue;
    if (pInfo->aConstraint[j].iColumn == 3 &&
        pInfo->aConstraint[j].op == SQLITE_INDEX_CONSTRAINT_EQ) {
      pInfo->aConstraintUsage[j].argvIndex = nextArg++;
      pInfo->aConstraintUsage[j].omit = 1;
      flags |= VEC0_IDX_FLAG_EF;
      break;
    }
  }
  for (int j = 0; j < pInfo->nConstraint; j++) {
    if (!pInfo->aConstraint[j].usable)
      continue;
    if (pInfo->aConstraint[j].op == SQLITE_INDEX_CONSTRAINT_LIMIT) {
      pInfo->aConstraintUsage[j].argvIndex = nextArg++;
      pInfo->aConstraintUsage[j].omit = 1;
      flags |= VEC0_IDX_FLAG_LIMIT;
      break;
    }
  }

  /* Distance threshold constraints on column 2 (distance REAL HIDDEN).
   * For kNN queries, these post-filter the result set.
   * Process each operator type in separate loops to ensure consistent argv
   * ordering. */
  for (int j = 0; j < pInfo->nConstraint; j++) {
    if (!pInfo->aConstraint[j].usable)
      continue;
    if (pInfo->aConstraint[j].iColumn == 2 &&
        pInfo->aConstraint[j].op == SQLITE_INDEX_CONSTRAINT_LT) {
      pInfo->aConstraintUsage[j].argvIndex = nextArg++;
      pInfo->aConstraintUsage[j].omit = 1;
      flags |= VEC0_IDX_FLAG_DIST_LT;
      break;
    }
  }
  for (int j = 0; j < pInfo->nConstraint; j++) {
    if (!pInfo->aConstraint[j].usable)
      continue;
    if (pInfo->aConstraint[j].iColumn == 2 &&
        pInfo->aConstraint[j].op == SQLITE_INDEX_CONSTRAINT_LE) {
      pInfo->aConstraintUsage[j].argvIndex = nextArg++;
      pInfo->aConstraintUsage[j].omit = 1;
      flags |= VEC0_IDX_FLAG_DIST_LE;
      break;
    }
  }
  for (int j = 0; j < pInfo->nConstraint; j++) {
    if (!pInfo->aConstraint[j].usable)
      continue;
    if (pInfo->aConstraint[j].iColumn == 2 &&
        pInfo->aConstraint[j].op == SQLITE_INDEX_CONSTRAINT_GT) {
      pInfo->aConstraintUsage[j].argvIndex = nextArg++;
      pInfo->aConstraintUsage[j].omit = 1;
      flags |= VEC0_IDX_FLAG_DIST_GT;
      break;
    }
  }
  for (int j = 0; j < pInfo->nConstraint; j++) {
    if (!pInfo->aConstraint[j].usable)
      continue;
    if (pInfo->aConstraint[j].iColumn == 2 &&
        pInfo->aConstraint[j].op == SQLITE_INDEX_CONSTRAINT_GE) {
      pInfo->aConstraintUsage[j].argvIndex = nextArg++;
      pInfo->aConstraintUsage[j].omit = 1;
      flags |= VEC0_IDX_FLAG_DIST_GE;
      break;
    }
  }

  /* 0. Operator-alias kNN constraints (op 151-156, set by xFindFunction). */
  for (int i = 0; i < pInfo->nConstraint; i++) {
    if (!pInfo->aConstraint[i].usable)
      continue;
    int op = pInfo->aConstraint[i].op;
    if (op >= VEC0_OP_DISTANCE_L2 && op <= VEC0_OP_DISTANCE_JACCARD) {
      pInfo->aConstraintUsage[i].argvIndex = 1;
      pInfo->aConstraintUsage[i].omit = 1;
      pInfo->idxNum = flags | op;
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
      pInfo->idxNum = flags | 1; /* kNN path */
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

  int has_ef_arg = (idxNum & VEC0_IDX_FLAG_EF) != 0;
  int has_limit_arg = (idxNum & VEC0_IDX_FLAG_LIMIT) != 0;
  int has_dist_lt = (idxNum & VEC0_IDX_FLAG_DIST_LT) != 0;
  int has_dist_le = (idxNum & VEC0_IDX_FLAG_DIST_LE) != 0;
  int has_dist_gt = (idxNum & VEC0_IDX_FLAG_DIST_GT) != 0;
  int has_dist_ge = (idxNum & VEC0_IDX_FLAG_DIST_GE) != 0;
  int base_idxnum = idxNum & VEC0_IDX_BASE_MASK;

  int ef = p->ef_search;
  int argi = 1;
  if (has_ef_arg && argi < argc) {
    if (sqlite3_value_type(argv[argi]) == SQLITE_INTEGER) {
      int v = sqlite3_value_int(argv[argi]);
      if (v > 0)
        ef = v;
    }
    argi++;
  }

  int k = ef;
  if (has_limit_arg && argi < argc) {
    if (sqlite3_value_type(argv[argi]) == SQLITE_INTEGER) {
      int lim = sqlite3_value_int(argv[argi]);
      if (lim > VEC0_MAX_LIMIT) {
        sqlite3_free(query_vec);
        cur->base.pVtab->zErrMsg =
            sqlite3_mprintf("vec0: LIMIT must be <= %d", VEC0_MAX_LIMIT);
        return SQLITE_CONSTRAINT;
      }
      if (lim > 0)
        k = lim;
    }
    argi++;
  }

  /* Parse distance threshold constraints */
  double dist_lt_val = 0.0, dist_le_val = 0.0;
  double dist_gt_val = 0.0, dist_ge_val = 0.0;
  if (has_dist_lt && argi < argc) {
    dist_lt_val = sqlite3_value_double(argv[argi++]);
  }
  if (has_dist_le && argi < argc) {
    dist_le_val = sqlite3_value_double(argv[argi++]);
  }
  if (has_dist_gt && argi < argc) {
    dist_gt_val = sqlite3_value_double(argv[argi++]);
  }
  if (has_dist_ge && argi < argc) {
    dist_ge_val = sqlite3_value_double(argv[argi++]);
  }

  dist_fn_t knn_dist_fn = p->dist_fn;
  if (base_idxnum >= VEC0_OP_DISTANCE_L2 &&
      base_idxnum <= VEC0_OP_DISTANCE_JACCARD) {
    static const char *op_metrics[] = {"l2", "cosine",  "ip",
                                       "l1", "hamming", "jaccard"};
    dist_fn_t override = distance_for_metric_typed(
        op_metrics[base_idxnum - VEC0_OP_DISTANCE_L2], p->storage_type);
    if (override)
      knn_dist_fn = override;
  }

  /* Convert query vector to storage type for HNSW search */
  int q_blob_bytes = 0;
  void *q_storage = vec_convert_to_storage(query_vec, query_dims,
                                           p->storage_type, &q_blob_bytes);
  const void *q_ptr = q_storage ? q_storage : (const void *)query_vec;

  HnswCtx hctx;
  vec0_make_hctx(p, &hctx);
  hctx.ef_search = ef;
  hctx.dist_fn = knn_dist_fn;

  HnswResult *results = NULL;
  int nResults = 0;
  rc = hnsw_search(&hctx, q_ptr, k, &results, &nResults);
  sqlite3_free(q_storage);
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

  /* Apply distance threshold post-filters */
  int filtered_count = 0;
  for (int i = 0; i < nResults; i++) {
    double d = results[i].dist;
    int pass = 1;
    if (has_dist_lt && d >= dist_lt_val)
      pass = 0;
    if (has_dist_le && d > dist_le_val)
      pass = 0;
    if (has_dist_gt && d <= dist_gt_val)
      pass = 0;
    if (has_dist_ge && d < dist_ge_val)
      pass = 0;
    if (pass) {
      cur->rowids[filtered_count] = results[i].rowid;
      cur->distances[filtered_count] = d;
      filtered_count++;
    }
  }
  cur->nResults = filtered_count;
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
  int base_idxnum = idxNum & VEC0_IDX_BASE_MASK;
  vec0_cursor_clear_results(cur);

  if (base_idxnum == 0)
    return vec0_filter_fullscan(cur);

  if (base_idxnum == 2)
    return vec0_filter_rowid_eq(cur, argc, argv);

  if (base_idxnum != 1 && (base_idxnum < VEC0_OP_DISTANCE_L2 ||
                           base_idxnum > VEC0_OP_DISTANCE_JACCARD))
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
  vec0_stmt_rewind(p->sc_get_vec);
  sqlite3_bind_int64(p->sc_get_vec, 1, cur->rowids[cur->pos]);
  int step_rc = sqlite3_step(p->sc_get_vec);
  if (step_rc == SQLITE_ROW) {
    int bytes = sqlite3_column_bytes(p->sc_get_vec, 0);
    const void *blob = sqlite3_column_blob(p->sc_get_vec, 0);

    if (p->storage_type == VEC_STORAGE_INT8) {
      /* Convert int8 back to float for text output */
      int dims = bytes;
      float *tmp = sqlite3_malloc(dims * (int)sizeof(float));
      if (tmp) {
        const int8_t *ib = (const int8_t *)blob;
        for (int i = 0; i < dims; i++)
          tmp[i] = (float)ib[i];
        char *text = vec_format(tmp, dims);
        sqlite3_free(tmp);
        if (text)
          sqlite3_result_text(ctx, text, -1, sqlite3_free);
        else
          sqlite3_result_null(ctx);
      } else {
        sqlite3_result_null(ctx);
      }
    } else if (p->storage_type == VEC_STORAGE_BIT) {
      /* Format bit-packed vector as [0,1,0,1,...] */
      const uint8_t *bits = (const uint8_t *)blob;
      int dims = p->dims;
      /* Each dim takes 2 chars (digit+comma) minus trailing comma, plus
       * brackets */
      int buflen = 1 + dims * 2 + 1;
      char *text = sqlite3_malloc(buflen);
      if (text) {
        int pos = 0;
        text[pos++] = '[';
        for (int i = 0; i < dims; i++) {
          if (i > 0)
            text[pos++] = ',';
          text[pos++] = (bits[i / 8] & (1 << (7 - (i % 8)))) ? '1' : '0';
        }
        text[pos++] = ']';
        text[pos] = '\0';
        sqlite3_result_text(ctx, text, pos, sqlite3_free);
      } else {
        sqlite3_result_null(ctx);
      }
    } else {
      /* VEC_STORAGE_F32 — original behavior */
      const float *fblob = (const float *)blob;
      int dims = bytes / (int)sizeof(float);
      char *text = vec_format(fblob, dims);
      if (text)
        sqlite3_result_text(ctx, text, -1, sqlite3_free);
      else
        sqlite3_result_null(ctx);
    }
  } else {
    sqlite3_result_null(ctx);
  }
  vec0_stmt_rewind(p->sc_get_vec);
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
    vec0_stmt_rewind(p->sc_del_data);
    sqlite3_bind_int64(p->sc_del_data, 1, del_rowid);
    sqlite3_step(p->sc_del_data);

    /* Remove from HNSW graph */
    HnswCtx hctx;
    vec0_make_hctx(p, &hctx);
    int drc = hnsw_delete(&hctx, del_rowid);
    if (drc != SQLITE_OK) {
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return drc;
    }
    p->entry_point = hctx.entry_point;
    p->max_layer = hctx.max_layer;

    if (p->count > 0)
      p->count--;
    if (!p->in_txn) {
      char buf[32];
      sqlite3_snprintf((int)sizeof(buf), buf, "%lld", p->count);
      vec0_stmt_rewind(p->sc_upd_config);
      sqlite3_bind_text(p->sc_upd_config, 1, buf, -1, SQLITE_TRANSIENT);
      sqlite3_bind_text(p->sc_upd_config, 2, "count", -1, SQLITE_STATIC);
      sqlite3_step(p->sc_upd_config);
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
    vec0_stmt_rewind(p->sc_del_data);
    sqlite3_bind_int64(p->sc_del_data, 1, old_rowid);
    sqlite3_step(p->sc_del_data);

    HnswCtx del_ctx;
    vec0_make_hctx(p, &del_ctx);
    upd_rc = hnsw_delete(&del_ctx, old_rowid);
    if (upd_rc != SQLITE_OK) {
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

    /* Re-insert with same rowid, converting to native storage type */
    int upd_blob_bytes = 0;
    void *upd_storage = vec_convert_to_storage(
        upd_vec, upd_dims, p->storage_type, &upd_blob_bytes);
    const void *upd_bind = upd_storage ? upd_storage : (const void *)upd_vec;

    vec0_stmt_rewind(p->sc_ins_data_id);
    sqlite3_bind_int64(p->sc_ins_data_id, 1, old_rowid);
    sqlite3_bind_blob(p->sc_ins_data_id, 2, upd_bind, upd_blob_bytes,
                      SQLITE_STATIC);
    upd_rc = sqlite3_step(p->sc_ins_data_id);
    if (upd_rc != SQLITE_DONE) {
      p->count = pre_count;
      p->entry_point = pre_ep;
      p->max_layer = pre_ml;
      sqlite3_free(upd_storage);
      sqlite3_free(upd_vec);
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return SQLITE_ERROR;
    }
    p->count++;
    *pRowid = old_rowid;

    /* Re-insert into HNSW graph — pass storage-type blob */
    HnswCtx ins_ctx;
    vec0_make_hctx(p, &ins_ctx);
    upd_rc = hnsw_insert(&ins_ctx, old_rowid, upd_bind);
    sqlite3_free(upd_storage);
    sqlite3_free(upd_vec);
    if (upd_rc != SQLITE_OK) {
      p->count = pre_count;
      p->entry_point = pre_ep;
      p->max_layer = pre_ml;
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return upd_rc;
    }
    p->entry_point = ins_ctx.entry_point;
    p->max_layer = ins_ctx.max_layer;
    if (!p->in_txn) {
      char buf[32];
      sqlite3_snprintf((int)sizeof(buf), buf, "%lld", p->count);
      vec0_stmt_rewind(p->sc_upd_config);
      sqlite3_bind_text(p->sc_upd_config, 1, buf, -1, SQLITE_TRANSIENT);
      sqlite3_bind_text(p->sc_upd_config, 2, "count", -1, SQLITE_STATIC);
      sqlite3_step(p->sc_upd_config);
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

  /* Insert vector BLOB into _data, converting to native storage type.
   * SQLite's automatic statement savepoint ensures that if hnsw_insert later
   * fails and we return an error, the _data row and any partial graph writes
   * are rolled back automatically by SQLite. */
  int blob_bytes = 0;
  void *storage_vec =
      vec_convert_to_storage(vec, dims, p->storage_type, &blob_bytes);
  const void *bind_ptr = storage_vec ? storage_vec : (const void *)vec;

  vec0_stmt_rewind(p->sc_ins_data);
  sqlite3_bind_blob(p->sc_ins_data, 1, bind_ptr, blob_bytes, SQLITE_STATIC);
  rc = sqlite3_step(p->sc_ins_data);
  if (rc != SQLITE_DONE) {
    sqlite3_free(storage_vec);
    sqlite3_free(vec);
    pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return SQLITE_ERROR;
  }

  *pRowid = sqlite3_last_insert_rowid(p->db);
  p->count++;

  /* Insert into HNSW graph — pass the storage-type blob */
  HnswCtx hctx;
  vec0_make_hctx(p, &hctx);
  rc = hnsw_insert(&hctx, *pRowid, bind_ptr);
  sqlite3_free(storage_vec);
  sqlite3_free(vec);
  if (rc != SQLITE_OK) {
    p->count--;
    pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
    return rc;
  }
  p->entry_point = hctx.entry_point;
  p->max_layer = hctx.max_layer;

  /* Keep count in _config in sync (deferred to xCommit when in_txn) */
  if (!p->in_txn) {
    char buf[32];
    sqlite3_snprintf((int)sizeof(buf), buf, "%lld", p->count);
    vec0_stmt_rewind(p->sc_upd_config);
    sqlite3_bind_text(p->sc_upd_config, 1, buf, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(p->sc_upd_config, 2, "count", -1, SQLITE_STATIC);
    sqlite3_step(p->sc_upd_config);
  }
  return SQLITE_OK;
}

/* ── xFindFunction ───────────────────────────────────────────────────────────
 * Intercept vec_distance_* calls on a vtab column so SQLite routes them
 * through xBestIndex as index-accelerated kNN scans.
 * Returns a nonzero operator code (VEC0_OP_DISTANCE_*) that becomes
 * aConstraint[i].op in xBestIndex.  See the constant definitions near the
 * top of this file for the mapping.
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
    return VEC0_OP_DISTANCE_L2;
  }
  if (strcmp(zName, "vec_distance_cosine") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"cosine";
    return VEC0_OP_DISTANCE_COSINE;
  }
  if (strcmp(zName, "vec_distance_ip") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"ip";
    return VEC0_OP_DISTANCE_IP;
  }
  if (strcmp(zName, "vec_distance_l1") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"l1";
    return VEC0_OP_DISTANCE_L1;
  }
  if (strcmp(zName, "vec_distance_hamming") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"hamming";
    return VEC0_OP_DISTANCE_HAMMING;
  }
  if (strcmp(zName, "vec_distance_jaccard") == 0) {
    *pxFunc = op_dist_fn;
    *ppArg = (void *)"jaccard";
    return VEC0_OP_DISTANCE_JACCARD;
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

static void vec0_write_config(Vec0Table *p, const char *key,
                              sqlite3_int64 val) {
  char buf[32];
  sqlite3_snprintf((int)sizeof(buf), buf, "%lld", val);
  vec0_stmt_rewind(p->sc_upd_config);
  sqlite3_bind_text(p->sc_upd_config, 1, buf, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(p->sc_upd_config, 2, key, -1, SQLITE_STATIC);
  sqlite3_step(p->sc_upd_config);
}

static int vec0Commit(sqlite3_vtab *pVtab) {
  Vec0Table *p = (Vec0Table *)pVtab;
  vec0_write_config(p, "count", p->count);
  vec0_write_config(p, "entry_point", p->entry_point);
  vec0_write_config(p, "max_layer", (sqlite3_int64)p->max_layer);
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

static int vec0Rename(sqlite3_vtab *pVtab, const char *zNew) {
  Vec0Table *p = (Vec0Table *)pVtab;
  static const char *shadows[] = {"config", "data", "graph", "layers"};

  if (!zNew || !*zNew) {
    pVtab->zErrMsg = sqlite3_mprintf("vec0: invalid new table name");
    return SQLITE_ERROR;
  }

  for (int i = 0; i < 4; i++) {
    char *sql =
        sqlite3_mprintf("ALTER TABLE \"%w\".\"%w_%w\" RENAME TO \"%w_%w\"",
                        p->schema, p->name, shadows[i], zNew, shadows[i]);
    int rc = sqlite3_exec(p->db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
      pVtab->zErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return rc;
    }
  }

  sqlite3_free(p->name);
  p->name = sqlite3_mprintf("%s", zNew);
  if (!p->name) {
    pVtab->zErrMsg = sqlite3_mprintf("vec0: out of memory");
    return SQLITE_NOMEM;
  }

  vec0_finalize_stmts(p);
  {
    char *prep_err = NULL;
    int rc = vec0_prepare_stmts(p, &prep_err);
    if (rc != SQLITE_OK) {
      pVtab->zErrMsg =
          prep_err ? prep_err : sqlite3_mprintf("%s", sqlite3_errmsg(p->db));
      return rc;
    }
  }

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
    /* xRename     */ vec0Rename,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ vec0ShadowName,
    /* xIntegrity  */ NULL,
};
