#ifndef SQLITE_VECTOR_HNSW_H
#define SQLITE_VECTOR_HNSW_H

#include "distance.h"
#include <sqlite3ext.h>
#include <stdint.h>

SQLITE_EXTENSION_INIT3

/*
 * HnswCtx — all HNSW algorithm state needed for search/insert/delete.
 * Populated by vtab.c from Vec0Table before calling any hnsw_* function.
 * entry_point and max_layer may be updated in-place by hnsw_insert/delete.
 */
typedef struct HnswCtx {
  sqlite3 *db;
  const char *schema;        /* schema name (not owned) */
  const char *tbl_name;      /* virtual table name (not owned) */
  int dims;                  /* vector dimensionality */
  int m;                     /* HNSW M: max neighbours per node per layer */
  int ef_construction;       /* beam width during index build */
  int ef_search;             /* beam width during query */
  sqlite3_int64 entry_point; /* rowid of entry-point node; -1 when empty */
  int max_layer;             /* current top layer */
  dist_fn_t dist_fn;         /* distance kernel (set from metric string) */

  /* If non-zero, update_config() skips DB writes (deferred to xCommit). */
  int defer_config;

  /* Cached prepared statements — must all be populated before calling
   * hnsw_*.  Owned by the vtab; hnsw_* functions never finalize them. */
  sqlite3_stmt *sc_get_nbrs;       /* SELECT neighbor_id, distance FROM _graph
                                      WHERE layer=? AND node_id=? */
  sqlite3_stmt *sc_get_vec;        /* SELECT vector FROM _data WHERE id=? */
  sqlite3_stmt *sc_ins_edge;       /* INSERT OR REPLACE INTO _graph(...) */
  sqlite3_stmt *sc_del_edges;      /* DELETE FROM _graph WHERE layer=? AND
                                      node_id=? */
  sqlite3_stmt *sc_ins_layer;      /* INSERT OR REPLACE INTO _layers(...) */
  sqlite3_stmt *sc_nbr_count;      /* SELECT COUNT(*) FROM _graph WHERE layer=?
                                      AND node_id=? */
  sqlite3_stmt *sc_rev_nbrs;       /* SELECT DISTINCT node_id FROM _graph WHERE
                                      layer=? AND neighbor_id=? */
  sqlite3_stmt *sc_get_layer;      /* SELECT max_layer FROM _layers WHERE
                                      node_id=? */
  sqlite3_stmt *sc_del_layer;      /* DELETE FROM _layers WHERE node_id=? */
  sqlite3_stmt *sc_del_node_edges; /* DELETE FROM _graph WHERE node_id=? OR
                                      neighbor_id=? */
  sqlite3_stmt *sc_elect_ep;       /* SELECT node_id, max_layer FROM _layers
                                      ORDER BY max_layer DESC LIMIT 1 */
  sqlite3_stmt *sc_upd_config;     /* UPDATE _config SET value=? WHERE key=? */
} HnswCtx;

/*
 * HnswResult — one element of the kNN result array returned by hnsw_search.
 */
typedef struct HnswResult {
  sqlite3_int64 rowid;
  double dist;
} HnswResult;

/*
 * hnsw_search — find k nearest neighbours of query_vec.
 *
 * On success: returns SQLITE_OK, allocates *out_results (caller must
 *             sqlite3_free), sets *out_n to number of results.
 * Returns 0 results (not an error) when the graph is empty.
 */
int hnsw_search(HnswCtx *ctx, const float *query_vec, int k,
                HnswResult **out_results, int *out_n);

/*
 * hnsw_insert — insert a new vector into the HNSW graph.
 *
 * rowid must already exist in {tbl_name}_data.
 * Writes to {tbl_name}_layers and {tbl_name}_graph.
 * Updates ctx->entry_point and ctx->max_layer, and persists both to _config.
 */
int hnsw_insert(HnswCtx *ctx, sqlite3_int64 rowid, const float *vec);

/*
 * hnsw_delete — remove a node from the HNSW graph.
 *
 * Deletes rows from {tbl_name}_layers and all edges touching rowid in
 * {tbl_name}_graph. If the deleted node was the entry_point, a new one is
 * selected. Updates ctx->entry_point / ctx->max_layer and persists to _config.
 * v2: graph repair — former neighbours of the deleted node are re-wired by
 * running a search_layer beam search to find replacement connections.
 */
int hnsw_delete(HnswCtx *ctx, sqlite3_int64 rowid);

#endif /* SQLITE_VECTOR_HNSW_H */
