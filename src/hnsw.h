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
  const char *schema;   /* schema name (not owned) */
  const char *tbl_name; /* virtual table name (not owned) */
  int dims;             /* vector dimensionality */
  int m;                /* HNSW M: max neighbours per node per layer */
  int ef_construction;  /* beam width during index build */
  int ef_search;        /* beam width during query */
  sqlite3_int64 entry_point; /* rowid of entry-point node; -1 when empty */
  int max_layer;             /* current top layer */
  dist_fn_t dist_fn;         /* distance kernel (set from metric string) */
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
 * v1: no graph repair — edges just become sparser; traversal skips missing
 * nodes.
 */
int hnsw_delete(HnswCtx *ctx, sqlite3_int64 rowid);

#endif /* SQLITE_VECTOR_HNSW_H */
