/*
 * hnsw.c — SQL-native HNSW graph implementation.
 *
 * All graph state lives in SQLite shadow tables.  The C heap holds only the
 * candidate/result sets (O(ef) nodes); no full index copy is kept in RAM.
 *
 * References:
 *   Malkov & Yashunin 2018, "Efficient and robust approximate nearest neighbor
 *   search using Hierarchical Navigable Small World graphs."
 */

#include "hnsw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* ══════════════════════════════════════════════════════════════════════════
 * Binary heap (shared implementation; min or max behaviour via comparator)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
  sqlite3_int64 id;
  double dist;
} HNode;

typedef struct {
  HNode *data;
  int len;
  int cap;
} Heap;

static int heap_init(Heap *h) {
  h->cap = 16;
  h->len = 0;
  h->data = sqlite3_malloc(h->cap * (int)sizeof(HNode));
  return h->data ? SQLITE_OK : SQLITE_NOMEM;
}

static void heap_free(Heap *h) {
  sqlite3_free(h->data);
  h->data = NULL;
  h->len = h->cap = 0;
}

static int heap_grow(Heap *h) {
  int ncap = h->cap * 2;
  HNode *nd = sqlite3_realloc(h->data, ncap * (int)sizeof(HNode));
  if (!nd)
    return SQLITE_NOMEM;
  h->data = nd;
  h->cap = ncap;
  return SQLITE_OK;
}

/* ── Max-heap ────────────────────────────────────────────────────────────── */

static void maxheap_sift_up(Heap *h, int i) {
  while (i > 0) {
    int p = (i - 1) / 2;
    if (h->data[p].dist >= h->data[i].dist)
      break;
    HNode tmp = h->data[p];
    h->data[p] = h->data[i];
    h->data[i] = tmp;
    i = p;
  }
}

static void maxheap_sift_down(Heap *h, int i) {
  for (;;) {
    int l = 2 * i + 1, r = 2 * i + 2, m = i;
    if (l < h->len && h->data[l].dist > h->data[m].dist)
      m = l;
    if (r < h->len && h->data[r].dist > h->data[m].dist)
      m = r;
    if (m == i)
      break;
    HNode tmp = h->data[m];
    h->data[m] = h->data[i];
    h->data[i] = tmp;
    i = m;
  }
}

static int maxheap_push(Heap *h, sqlite3_int64 id, double dist) {
  if (h->len == h->cap) {
    int rc = heap_grow(h);
    if (rc != SQLITE_OK)
      return rc;
  }
  h->data[h->len].id = id;
  h->data[h->len].dist = dist;
  maxheap_sift_up(h, h->len++);
  return SQLITE_OK;
}

static void maxheap_pop(Heap *h) {
  if (h->len <= 0)
    return;
  h->data[0] = h->data[--h->len];
  if (h->len > 0)
    maxheap_sift_down(h, 0);
}

/* ── Min-heap ────────────────────────────────────────────────────────────── */

static void minheap_sift_up(Heap *h, int i) {
  while (i > 0) {
    int p = (i - 1) / 2;
    if (h->data[p].dist <= h->data[i].dist)
      break;
    HNode tmp = h->data[p];
    h->data[p] = h->data[i];
    h->data[i] = tmp;
    i = p;
  }
}

static void minheap_sift_down(Heap *h, int i) {
  for (;;) {
    int l = 2 * i + 1, r = 2 * i + 2, m = i;
    if (l < h->len && h->data[l].dist < h->data[m].dist)
      m = l;
    if (r < h->len && h->data[r].dist < h->data[m].dist)
      m = r;
    if (m == i)
      break;
    HNode tmp = h->data[m];
    h->data[m] = h->data[i];
    h->data[i] = tmp;
    i = m;
  }
}

static int minheap_push(Heap *h, sqlite3_int64 id, double dist) {
  if (h->len == h->cap) {
    int rc = heap_grow(h);
    if (rc != SQLITE_OK)
      return rc;
  }
  h->data[h->len].id = id;
  h->data[h->len].dist = dist;
  minheap_sift_up(h, h->len++);
  return SQLITE_OK;
}

static void minheap_pop(Heap *h) {
  if (h->len <= 0)
    return;
  h->data[0] = h->data[--h->len];
  if (h->len > 0)
    minheap_sift_down(h, 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Visited-node set — sorted int64 array with binary search.
 * Efficient for sets ≤ a few thousand elements (ef × M).
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
  sqlite3_int64 *ids;
  int len;
  int cap;
} VisitedSet;

static int visited_init(VisitedSet *v) {
  v->cap = 64;
  v->len = 0;
  v->ids = sqlite3_malloc(v->cap * (int)sizeof(sqlite3_int64));
  return v->ids ? SQLITE_OK : SQLITE_NOMEM;
}

static void visited_free(VisitedSet *v) {
  sqlite3_free(v->ids);
  v->ids = NULL;
}

static int visited_contains(const VisitedSet *v, sqlite3_int64 id) {
  int lo = 0, hi = v->len - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (v->ids[mid] == id)
      return 1;
    if (v->ids[mid] < id)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return 0;
}

static int visited_add(VisitedSet *v, sqlite3_int64 id) {
  /* binary search for insertion point */
  int lo = 0, hi = v->len;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (v->ids[mid] < id)
      lo = mid + 1;
    else
      hi = mid;
  }
  /* grow if needed */
  if (v->len == v->cap) {
    int ncap = v->cap * 2;
    sqlite3_int64 *nd =
        sqlite3_realloc(v->ids, ncap * (int)sizeof(sqlite3_int64));
    if (!nd)
      return SQLITE_NOMEM;
    v->ids = nd;
    v->cap = ncap;
  }
  memmove(v->ids + lo + 1, v->ids + lo,
          (v->len - lo) * (int)sizeof(sqlite3_int64));
  v->ids[lo] = id;
  v->len++;
  return SQLITE_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * SQL helpers
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * fetch_vector_stmt — bind and step stmt_get_vec for id.
 * *out_vec points into the stmt column buffer; valid until the next
 * sqlite3_reset / sqlite3_step on that statement.
 * Caller must NOT sqlite3_free *out_vec.
 */
static int fetch_vector_stmt(sqlite3_stmt *stmt_get_vec, sqlite3_int64 id,
                             const float **out_vec, int dims) {
  sqlite3_reset(stmt_get_vec);
  sqlite3_bind_int64(stmt_get_vec, 1, id);
  int rc = sqlite3_step(stmt_get_vec);
  if (rc != SQLITE_ROW)
    return SQLITE_ERROR;
  int bytes = sqlite3_column_bytes(stmt_get_vec, 0);
  if (bytes != dims * (int)sizeof(float))
    return SQLITE_ERROR;
  *out_vec = (const float *)sqlite3_column_blob(stmt_get_vec, 0);
  return SQLITE_OK;
}

/* Persist a single integer config value. */
static int update_config(HnswCtx *ctx, const char *key, sqlite3_int64 val) {
  char *sql = sqlite3_mprintf("UPDATE \"%w\".\"%w_config\" SET value='%lld' "
                              "WHERE key='%q'",
                              ctx->schema, ctx->tbl_name, val, key);
  int rc = sqlite3_exec(ctx->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  return rc;
}

/* ══════════════════════════════════════════════════════════════════════════
 * search_layer — Algorithm 2 from the HNSW paper.
 *
 * Inputs:
 *   ctx            — vtab context
 *   query          — query float vector (dims floats)
 *   ep_id/ep_dist  — entry-point node and its distance to query
 *   ef             — beam width (max size of W)
 *   lc             — layer to search
 *   stmt_get_nbrs  — "SELECT neighbor_id, distance FROM _graph WHERE layer=?
 *                       AND node_id=?"
 *   stmt_get_vec   — "SELECT vector FROM _data WHERE id=?"
 *   W              — pre-initialised empty max-heap; filled with ≤ef nearest
 *                    nodes found at this layer (caller frees)
 * ══════════════════════════════════════════════════════════════════════════ */
static int search_layer(HnswCtx *ctx, const float *query, sqlite3_int64 ep_id,
                        double ep_dist, int ef, int lc,
                        sqlite3_stmt *stmt_get_nbrs, sqlite3_stmt *stmt_get_vec,
                        Heap *W) {
  Heap C; /* min-heap: candidates ordered by distance (nearest first) */
  int rc = heap_init(&C);
  if (rc != SQLITE_OK)
    return rc;

  VisitedSet visited;
  rc = visited_init(&visited);
  if (rc != SQLITE_OK) {
    heap_free(&C);
    return rc;
  }

  rc = maxheap_push(W, ep_id, ep_dist);
  if (rc)
    goto done;
  rc = minheap_push(&C, ep_id, ep_dist);
  if (rc)
    goto done;
  rc = visited_add(&visited, ep_id);
  if (rc)
    goto done;

  while (C.len > 0) {
    HNode c = C.data[0]; /* nearest candidate */
    minheap_pop(&C);

    double f_dist = W->data[0].dist; /* furthest in W */
    if (c.dist > f_dist)
      break; /* all remaining candidates are farther than worst result */

    /* Expand c's neighbours at layer lc */
    sqlite3_reset(stmt_get_nbrs);
    sqlite3_bind_int(stmt_get_nbrs, 1, lc);
    sqlite3_bind_int64(stmt_get_nbrs, 2, c.id);

    while (sqlite3_step(stmt_get_nbrs) == SQLITE_ROW) {
      sqlite3_int64 e_id = sqlite3_column_int64(stmt_get_nbrs, 0);
      if (visited_contains(&visited, e_id))
        continue;
      rc = visited_add(&visited, e_id);
      if (rc)
        goto done;

      /* Fetch vector and compute distance to query */
      const float *e_vec = NULL;
      if (fetch_vector_stmt(stmt_get_vec, e_id, &e_vec, ctx->dims) != SQLITE_OK)
        continue; /* skip deleted/missing nodes */

      double e_dist = 0.0;
      ctx->dist_fn(query, e_vec, ctx->dims, &e_dist);

      double fw_dist = (W->len > 0) ? W->data[0].dist : 1e300;
      if (e_dist < fw_dist || W->len < ef) {
        rc = minheap_push(&C, e_id, e_dist);
        if (rc)
          goto done;
        rc = maxheap_push(W, e_id, e_dist);
        if (rc)
          goto done;
        if (W->len > ef)
          maxheap_pop(W); /* prune to ef */
      }
    }
  }

done:
  heap_free(&C);
  visited_free(&visited);
  return rc;
}

/* ══════════════════════════════════════════════════════════════════════════
 * hnsw_search
 * ══════════════════════════════════════════════════════════════════════════ */
int hnsw_search(HnswCtx *ctx, const float *query_vec, int k,
                HnswResult **out_results, int *out_n) {
  *out_results = NULL;
  *out_n = 0;

  if (ctx->entry_point < 0)
    return SQLITE_OK; /* empty graph */

  int local = (ctx->sc_get_nbrs == NULL);
  char *sql;
  int rc = SQLITE_OK;
  sqlite3_stmt *stmt_get_nbrs = NULL, *stmt_get_vec = NULL;

  if (local) {
    sql =
        sqlite3_mprintf("SELECT neighbor_id, distance FROM \"%w\".\"%w_graph\""
                        " WHERE layer=? AND node_id=?",
                        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_get_nbrs, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      return rc;

    sql = sqlite3_mprintf("SELECT vector FROM \"%w\".\"%w_data\" WHERE id=?",
                          ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_get_vec, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
      sqlite3_finalize(stmt_get_nbrs);
      return rc;
    }
  } else {
    stmt_get_nbrs = ctx->sc_get_nbrs;
    stmt_get_vec = ctx->sc_get_vec;
  }

  /* Distance from query to entry point */
  const float *ep_vec = NULL;
  rc = fetch_vector_stmt(stmt_get_vec, ctx->entry_point, &ep_vec, ctx->dims);
  if (rc != SQLITE_OK)
    goto cleanup;

  double ep_dist = 0.0;
  ctx->dist_fn(query_vec, ep_vec, ctx->dims, &ep_dist);

  sqlite3_int64 cur_ep = ctx->entry_point;
  double cur_ep_dist = ep_dist;

  /* Greedy descent: layers max_layer → 1 (ef=1) */
  for (int lc = ctx->max_layer; lc >= 1; lc--) {
    Heap W;
    rc = heap_init(&W);
    if (rc != SQLITE_OK)
      goto cleanup;
    rc = search_layer(ctx, query_vec, cur_ep, cur_ep_dist, 1, lc, stmt_get_nbrs,
                      stmt_get_vec, &W);
    if (rc == SQLITE_OK && W.len > 0) {
      /* find nearest in W (scan; W is tiny at ef=1) */
      HNode best = W.data[0];
      for (int i = 1; i < W.len; i++)
        if (W.data[i].dist < best.dist)
          best = W.data[i];
      cur_ep = best.id;
      cur_ep_dist = best.dist;
    }
    heap_free(&W);
    if (rc != SQLITE_OK)
      goto cleanup;
  }

  /* Full beam search at layer 0 */
  int ef = ctx->ef_search > k ? ctx->ef_search : k;
  Heap W0;
  rc = heap_init(&W0);
  if (rc != SQLITE_OK)
    goto cleanup;
  rc = search_layer(ctx, query_vec, cur_ep, cur_ep_dist, ef, 0, stmt_get_nbrs,
                    stmt_get_vec, &W0);
  if (rc != SQLITE_OK) {
    heap_free(&W0);
    goto cleanup;
  }

  /* Prune W0 to k nearest */
  while (W0.len > k)
    maxheap_pop(&W0);

  int n = W0.len;
  *out_n = n;
  if (n == 0) {
    heap_free(&W0);
    goto cleanup;
  }

  *out_results = sqlite3_malloc(n * (int)sizeof(HnswResult));
  if (!*out_results) {
    heap_free(&W0);
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  /*
   * Pop from max-heap into result array in reverse order so that
   * index 0 is the nearest (smallest distance).
   */
  for (int i = n - 1; i >= 0; i--) {
    (*out_results)[i].rowid = W0.data[0].id;
    (*out_results)[i].dist = W0.data[0].dist;
    maxheap_pop(&W0);
  }
  heap_free(&W0);

cleanup:
  if (local) {
    sqlite3_finalize(stmt_get_nbrs);
    sqlite3_finalize(stmt_get_vec);
  }
  return rc;
}

/* ══════════════════════════════════════════════════════════════════════════
 * set_connections — write the M nearest edges from a candidate heap for one
 * node at one layer.  Replaces all existing edges for that (layer, node_id).
 *
 * W is a max-heap of candidates (may have more than M entries).
 * stmt_del_edges: "DELETE FROM _graph WHERE layer=? AND node_id=?"
 * stmt_ins_edge:  "INSERT OR REPLACE INTO _graph(layer,node_id,neighbor_id,
 *                  distance) VALUES(?,?,?,?)"
 * ══════════════════════════════════════════════════════════════════════════ */
static int set_connections(HnswCtx *ctx, int lc, sqlite3_int64 node_id,
                           Heap *W, /* candidates max-heap */
                           int M, sqlite3_stmt *stmt_ins_edge,
                           sqlite3_stmt *stmt_del_edges) {
  (void)ctx;

  /* Delete existing edges for (lc, node_id) */
  sqlite3_reset(stmt_del_edges);
  sqlite3_bind_int(stmt_del_edges, 1, lc);
  sqlite3_bind_int64(stmt_del_edges, 2, node_id);
  sqlite3_step(stmt_del_edges);

  /* Copy W.data to a temp array and insertion-sort ascending by distance */
  int ntmp = W->len;
  if (ntmp == 0)
    return SQLITE_OK;
  HNode *tmp = sqlite3_malloc(ntmp * (int)sizeof(HNode));
  if (!tmp)
    return SQLITE_NOMEM;
  memcpy(tmp, W->data, ntmp * (int)sizeof(HNode));

  for (int i = 1; i < ntmp; i++) {
    HNode key = tmp[i];
    int j = i - 1;
    while (j >= 0 && tmp[j].dist > key.dist) {
      tmp[j + 1] = tmp[j];
      j--;
    }
    tmp[j + 1] = key;
  }

  int n = ntmp < M ? ntmp : M;
  for (int i = 0; i < n; i++) {
    sqlite3_reset(stmt_ins_edge);
    sqlite3_bind_int(stmt_ins_edge, 1, lc);
    sqlite3_bind_int64(stmt_ins_edge, 2, node_id);
    sqlite3_bind_int64(stmt_ins_edge, 3, tmp[i].id);
    sqlite3_bind_double(stmt_ins_edge, 4, tmp[i].dist);
    sqlite3_step(stmt_ins_edge);
  }

  sqlite3_free(tmp);
  return SQLITE_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * hnsw_insert — Algorithm 1 (insert) from the HNSW paper.
 * ══════════════════════════════════════════════════════════════════════════ */
int hnsw_insert(HnswCtx *ctx, sqlite3_int64 rowid, const float *vec) {
  int rc = SQLITE_OK;

  /* Sample random level: l = floor(-ln(U(0,1)) / ln(M)) */
  double r = (double)(rand() + 1) / ((double)RAND_MAX + 2.0);
  int node_level = (int)(-log(r) / log((double)(ctx->m > 1 ? ctx->m : 2)));
  if (node_level < 0)
    node_level = 0;

  int local = (ctx->sc_get_nbrs == NULL);
  char *sql;
  sqlite3_stmt *stmt_get_nbrs = NULL, *stmt_get_vec = NULL;
  sqlite3_stmt *stmt_ins_edge = NULL, *stmt_del_edges = NULL;
  sqlite3_stmt *stmt_ins_layer = NULL, *stmt_nbr_count = NULL;

  if (local) {
    sql =
        sqlite3_mprintf("SELECT neighbor_id, distance FROM \"%w\".\"%w_graph\""
                        " WHERE layer=? AND node_id=?",
                        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_get_nbrs, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      return rc;

    sql = sqlite3_mprintf("SELECT vector FROM \"%w\".\"%w_data\" WHERE id=?",
                          ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_get_vec, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;

    sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w\".\"%w_graph\""
        "(layer, node_id, neighbor_id, distance) VALUES (?,?,?,?)",
        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_ins_edge, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;

    sql = sqlite3_mprintf(
        "DELETE FROM \"%w\".\"%w_graph\" WHERE layer=? AND node_id=?",
        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_del_edges, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;

    sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w\".\"%w_layers\"(node_id, max_layer)"
        " VALUES (?,?)",
        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_ins_layer, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;

    sql = sqlite3_mprintf(
        "SELECT COUNT(*) FROM \"%w\".\"%w_graph\" WHERE layer=? AND node_id=?",
        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_nbr_count, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;
  } else {
    stmt_get_nbrs = ctx->sc_get_nbrs;
    stmt_get_vec = ctx->sc_get_vec;
    stmt_ins_edge = ctx->sc_ins_edge;
    stmt_del_edges = ctx->sc_del_edges;
    stmt_ins_layer = ctx->sc_ins_layer;
    stmt_nbr_count = ctx->sc_nbr_count;
  }

  /* Write this node's max layer to _layers */
  sqlite3_reset(stmt_ins_layer);
  sqlite3_bind_int64(stmt_ins_layer, 1, rowid);
  sqlite3_bind_int(stmt_ins_layer, 2, node_level);
  rc = sqlite3_step(stmt_ins_layer);
  rc = (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
  if (rc != SQLITE_OK)
    goto cleanup;

  /* If graph is empty, this node becomes the sole entry point */
  if (ctx->entry_point < 0) {
    ctx->entry_point = rowid;
    ctx->max_layer = node_level;
    update_config(ctx, "entry_point", rowid);
    update_config(ctx, "max_layer", node_level);
    goto cleanup;
  }

  /* Distance from new node to current entry point */
  const float *ep_vec = NULL;
  rc = fetch_vector_stmt(stmt_get_vec, ctx->entry_point, &ep_vec, ctx->dims);
  if (rc != SQLITE_OK)
    goto cleanup;

  double ep_dist = 0.0;
  ctx->dist_fn(vec, ep_vec, ctx->dims, &ep_dist);

  sqlite3_int64 cur_ep = ctx->entry_point;
  double cur_ep_dist = ep_dist;

  /* Greedy descent from max_layer → node_level+1 (ef=1) */
  for (int lc = ctx->max_layer; lc > node_level; lc--) {
    Heap W;
    rc = heap_init(&W);
    if (rc != SQLITE_OK)
      goto cleanup;
    rc = search_layer(ctx, vec, cur_ep, cur_ep_dist, 1, lc, stmt_get_nbrs,
                      stmt_get_vec, &W);
    if (rc == SQLITE_OK && W.len > 0) {
      HNode best = W.data[0];
      for (int i = 1; i < W.len; i++)
        if (W.data[i].dist < best.dist)
          best = W.data[i];
      cur_ep = best.id;
      cur_ep_dist = best.dist;
    }
    heap_free(&W);
    if (rc != SQLITE_OK)
      goto cleanup;
  }

  /* Build connections at each layer from min(max_layer, node_level) down to 0
   */
  int connect_top = ctx->max_layer < node_level ? ctx->max_layer : node_level;

  for (int lc = connect_top; lc >= 0; lc--) {
    int Mmax = (lc == 0) ? ctx->m * 2 : ctx->m;

    Heap W;
    rc = heap_init(&W);
    if (rc != SQLITE_OK)
      goto cleanup;
    rc = search_layer(ctx, vec, cur_ep, cur_ep_dist, ctx->ef_construction, lc,
                      stmt_get_nbrs, stmt_get_vec, &W);
    if (rc != SQLITE_OK) {
      heap_free(&W);
      goto cleanup;
    }

    if (W.len == 0) {
      heap_free(&W);
      continue;
    }

    /* Connect new node → its M nearest neighbours */
    rc = set_connections(ctx, lc, rowid, &W, Mmax, stmt_ins_edge,
                         stmt_del_edges);
    if (rc != SQLITE_OK) {
      heap_free(&W);
      goto cleanup;
    }

    /*
     * Add reverse edges: each selected neighbour → new node.
     * Then prune that neighbour's edges if it exceeds Mmax.
     */
    /* Sort W ascending to pick the M best we actually connected */
    int ntmp = W.len;
    HNode *tmp = sqlite3_malloc(ntmp * (int)sizeof(HNode));
    if (!tmp) {
      heap_free(&W);
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    memcpy(tmp, W.data, ntmp * (int)sizeof(HNode));
    for (int i = 1; i < ntmp; i++) {
      HNode key = tmp[i];
      int j = i - 1;
      while (j >= 0 && tmp[j].dist > key.dist) {
        tmp[j + 1] = tmp[j];
        j--;
      }
      tmp[j + 1] = key;
    }
    int nconn = ntmp < Mmax ? ntmp : Mmax;

    for (int i = 0; i < nconn; i++) {
      sqlite3_int64 nbr_id = tmp[i].id;
      double nbr_dist = tmp[i].dist;

      /* Insert reverse edge: nbr_id → rowid */
      sqlite3_reset(stmt_ins_edge);
      sqlite3_bind_int(stmt_ins_edge, 1, lc);
      sqlite3_bind_int64(stmt_ins_edge, 2, nbr_id);
      sqlite3_bind_int64(stmt_ins_edge, 3, rowid);
      sqlite3_bind_double(stmt_ins_edge, 4, nbr_dist);
      sqlite3_step(stmt_ins_edge);

      /* Count current out-edges of nbr_id at lc using cached/local stmt */
      sqlite3_reset(stmt_nbr_count);
      sqlite3_bind_int(stmt_nbr_count, 1, lc);
      sqlite3_bind_int64(stmt_nbr_count, 2, nbr_id);
      int nbr_conn = 0;
      if (sqlite3_step(stmt_nbr_count) == SQLITE_ROW)
        nbr_conn = sqlite3_column_int(stmt_nbr_count, 0);

      if (nbr_conn > Mmax) {
        /* Fetch all existing edges for nbr_id and prune to Mmax.
         * Reuse stmt_get_nbrs (search_layer has finished; safe to reset). */
        sqlite3_reset(stmt_get_nbrs);
        sqlite3_bind_int(stmt_get_nbrs, 1, lc);
        sqlite3_bind_int64(stmt_get_nbrs, 2, nbr_id);

        Heap nbr_W;
        heap_init(&nbr_W);
        while (sqlite3_step(stmt_get_nbrs) == SQLITE_ROW) {
          sqlite3_int64 nid = sqlite3_column_int64(stmt_get_nbrs, 0);
          double nd = sqlite3_column_double(stmt_get_nbrs, 1);
          maxheap_push(&nbr_W, nid, nd);
        }
        /* Reset so stmt_get_nbrs is clean for next search_layer call */
        sqlite3_reset(stmt_get_nbrs);

        set_connections(ctx, lc, nbr_id, &nbr_W, Mmax, stmt_ins_edge,
                        stmt_del_edges);
        heap_free(&nbr_W);
      }
    }
    sqlite3_free(tmp);

    /* Advance ep to nearest found in W for next layer */
    HNode best = W.data[0];
    for (int i = 1; i < W.len; i++)
      if (W.data[i].dist < best.dist)
        best = W.data[i];
    cur_ep = best.id;
    cur_ep_dist = best.dist;
    heap_free(&W);
  }

  /* Update entry point if this node's level exceeds the current maximum */
  if (node_level > ctx->max_layer) {
    ctx->entry_point = rowid;
    ctx->max_layer = node_level;
    update_config(ctx, "entry_point", rowid);
    update_config(ctx, "max_layer", node_level);
  }

cleanup:
  if (local) {
    sqlite3_finalize(stmt_get_nbrs);
    sqlite3_finalize(stmt_get_vec);
    sqlite3_finalize(stmt_ins_edge);
    sqlite3_finalize(stmt_del_edges);
    sqlite3_finalize(stmt_ins_layer);
    sqlite3_finalize(stmt_nbr_count);
  }
  return rc;
}

/* ══════════════════════════════════════════════════════════════════════════
 * hnsw_delete — v2: remove node + edges, then repair affected neighbours.
 *
 * Repair: For each node N that had a forward edge N→D at some layer lc,
 * we run a search_layer beam search to find replacement connections and
 * call set_connections to re-wire N at that layer.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Per-layer repair target: node N that lost an edge to deleted D. */
typedef struct {
  int lc;
  sqlite3_int64 nid;
} RepairTarget;

int hnsw_delete(HnswCtx *ctx, sqlite3_int64 rowid) {
  char *sql;
  int rc = SQLITE_OK;

  int local = (ctx->sc_get_nbrs == NULL);
  sqlite3_stmt *stmt_get_nbrs = NULL, *stmt_get_vec = NULL;
  sqlite3_stmt *stmt_ins_edge = NULL, *stmt_del_edges = NULL;
  sqlite3_stmt *stmt_rev_nbrs = NULL;

  if (local) {
    sql =
        sqlite3_mprintf("SELECT neighbor_id, distance FROM \"%w\".\"%w_graph\""
                        " WHERE layer=? AND node_id=?",
                        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_get_nbrs, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      return rc;

    sql = sqlite3_mprintf("SELECT vector FROM \"%w\".\"%w_data\" WHERE id=?",
                          ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_get_vec, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;

    sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w\".\"%w_graph\""
        "(layer, node_id, neighbor_id, distance) VALUES (?,?,?,?)",
        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_ins_edge, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;

    sql = sqlite3_mprintf(
        "DELETE FROM \"%w\".\"%w_graph\" WHERE layer=? AND node_id=?",
        ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_del_edges, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;

    sql = sqlite3_mprintf("SELECT DISTINCT node_id FROM \"%w\".\"%w_graph\""
                          " WHERE layer=? AND neighbor_id=?",
                          ctx->schema, ctx->tbl_name);
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &stmt_rev_nbrs, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
      goto cleanup;
  } else {
    stmt_get_nbrs = ctx->sc_get_nbrs;
    stmt_get_vec = ctx->sc_get_vec;
    stmt_ins_edge = ctx->sc_ins_edge;
    stmt_del_edges = ctx->sc_del_edges;
    stmt_rev_nbrs = ctx->sc_rev_nbrs;
  }

  /* ── Step 1: Get deleted node's max layer (before removing from _layers) ──
   */
  int d_max_layer = 0;
  {
    char *lsql = sqlite3_mprintf(
        "SELECT max_layer FROM \"%w\".\"%w_layers\" WHERE node_id=%lld",
        ctx->schema, ctx->tbl_name, rowid);
    sqlite3_stmt *lstmt = NULL;
    if (sqlite3_prepare_v2(ctx->db, lsql, -1, &lstmt, NULL) == SQLITE_OK) {
      if (sqlite3_step(lstmt) == SQLITE_ROW)
        d_max_layer = sqlite3_column_int(lstmt, 0);
      sqlite3_finalize(lstmt);
    }
    sqlite3_free(lsql);
  }

  /* ── Step 2: Collect repair targets before edge deletion ─────────────── */
  /* For each layer lc, collect nodes N that had D as a forward neighbor
   * (i.e., N→D edge exists): SELECT DISTINCT node_id WHERE neighbor_id=D.
   * These nodes lose a direct connection and will be re-wired. */
  RepairTarget *repairs = NULL;
  int n_repairs = 0;
  int cap_repairs = 0;

  for (int lc = 0; lc <= d_max_layer; lc++) {
    sqlite3_reset(stmt_rev_nbrs);
    sqlite3_bind_int(stmt_rev_nbrs, 1, lc);
    sqlite3_bind_int64(stmt_rev_nbrs, 2, rowid);
    while (sqlite3_step(stmt_rev_nbrs) == SQLITE_ROW) {
      sqlite3_int64 nid = sqlite3_column_int64(stmt_rev_nbrs, 0);
      if (nid == rowid)
        continue;
      if (n_repairs == cap_repairs) {
        int ncap = cap_repairs ? cap_repairs * 2 : 16;
        RepairTarget *nd =
            sqlite3_realloc(repairs, ncap * (int)sizeof(RepairTarget));
        if (!nd) {
          rc = SQLITE_NOMEM;
          goto cleanup;
        }
        repairs = nd;
        cap_repairs = ncap;
      }
      repairs[n_repairs].lc = lc;
      repairs[n_repairs].nid = nid;
      n_repairs++;
    }
  }

  /* ── Step 3: Delete from _layers ─────────────────────────────────────── */
  /* Reset stmt_rev_nbrs so it releases any shared read lock before we write */
  sqlite3_reset(stmt_rev_nbrs);
  sql = sqlite3_mprintf("DELETE FROM \"%w\".\"%w_layers\" WHERE node_id=%lld",
                        ctx->schema, ctx->tbl_name, rowid);
  rc = sqlite3_exec(ctx->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    goto cleanup;

  /* ── Step 4: Delete all graph edges touching this node ───────────────── */
  sql = sqlite3_mprintf("DELETE FROM \"%w\".\"%w_graph\""
                        " WHERE node_id=%lld OR neighbor_id=%lld",
                        ctx->schema, ctx->tbl_name, rowid, rowid);
  rc = sqlite3_exec(ctx->db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    goto cleanup;

  /* ── Step 5: Entry-point election ────────────────────────────────────── */
  if (ctx->entry_point == rowid) {
    sql = sqlite3_mprintf("SELECT node_id, max_layer FROM \"%w\".\"%w_layers\""
                          " ORDER BY max_layer DESC LIMIT 1",
                          ctx->schema, ctx->tbl_name);
    sqlite3_stmt *ep_stmt = NULL;
    rc = sqlite3_prepare_v2(ctx->db, sql, -1, &ep_stmt, NULL);
    sqlite3_free(sql);
    if (rc == SQLITE_OK) {
      if (sqlite3_step(ep_stmt) == SQLITE_ROW) {
        ctx->entry_point = sqlite3_column_int64(ep_stmt, 0);
        ctx->max_layer = sqlite3_column_int(ep_stmt, 1);
      } else {
        ctx->entry_point = -1;
        ctx->max_layer = 0;
      }
      sqlite3_finalize(ep_stmt);
    }
    update_config(ctx, "entry_point", ctx->entry_point);
    update_config(ctx, "max_layer", (sqlite3_int64)ctx->max_layer);
  }

  /* ── Step 6: Graph repair ─────────────────────────────────────────────── */
  if (ctx->entry_point >= 0 && n_repairs > 0) {
    float *n_vec_copy = sqlite3_malloc(ctx->dims * (int)sizeof(float));
    if (!n_vec_copy) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }

    for (int i = 0; i < n_repairs && rc == SQLITE_OK; i++) {
      int lc = repairs[i].lc;
      sqlite3_int64 nid = repairs[i].nid;

      /* Fetch N's vector */
      const float *n_raw = NULL;
      if (fetch_vector_stmt(stmt_get_vec, nid, &n_raw, ctx->dims) != SQLITE_OK)
        continue; /* node missing or wrong dims — skip */
      memcpy(n_vec_copy, n_raw, ctx->dims * (int)sizeof(float));

      /* Fetch entry point's vector to compute distance for search_layer */
      const float *ep_raw = NULL;
      if (fetch_vector_stmt(stmt_get_vec, ctx->entry_point, &ep_raw,
                            ctx->dims) != SQLITE_OK)
        continue;
      double ep_dist = 0.0;
      ctx->dist_fn(n_vec_copy, ep_raw, ctx->dims, &ep_dist);

      /* Search at layer lc for replacement neighbours */
      Heap W;
      rc = heap_init(&W);
      if (rc != SQLITE_OK)
        break;

      rc = search_layer(ctx, n_vec_copy, ctx->entry_point, ep_dist,
                        ctx->ef_construction, lc, stmt_get_nbrs, stmt_get_vec,
                        &W);

      if (rc == SQLITE_OK && W.len > 0) {
        int Mmax = (lc == 0) ? ctx->m * 2 : ctx->m;

        /* Re-wire N's forward edges at lc */
        rc = set_connections(ctx, lc, nid, &W, Mmax, stmt_ins_edge,
                             stmt_del_edges);

        /* Add reverse edges: each selected neighbour X → N */
        if (rc == SQLITE_OK) {
          int ntmp = W.len;
          HNode *tmp = sqlite3_malloc(ntmp * (int)sizeof(HNode));
          if (tmp) {
            memcpy(tmp, W.data, ntmp * (int)sizeof(HNode));
            /* Sort ascending by distance */
            for (int j = 1; j < ntmp; j++) {
              HNode key = tmp[j];
              int kk = j - 1;
              while (kk >= 0 && tmp[kk].dist > key.dist) {
                tmp[kk + 1] = tmp[kk];
                kk--;
              }
              tmp[kk + 1] = key;
            }
            int nconn = ntmp < Mmax ? ntmp : Mmax;
            for (int j = 0; j < nconn; j++) {
              sqlite3_reset(stmt_ins_edge);
              sqlite3_bind_int(stmt_ins_edge, 1, lc);
              sqlite3_bind_int64(stmt_ins_edge, 2, tmp[j].id);
              sqlite3_bind_int64(stmt_ins_edge, 3, nid);
              sqlite3_bind_double(stmt_ins_edge, 4, tmp[j].dist);
              sqlite3_step(stmt_ins_edge);
            }
            sqlite3_free(tmp);
          }
        }
      }
      heap_free(&W);
    }
    sqlite3_free(n_vec_copy);
  }

cleanup:
  sqlite3_free(repairs);
  if (local) {
    sqlite3_finalize(stmt_get_nbrs);
    sqlite3_finalize(stmt_get_vec);
    sqlite3_finalize(stmt_ins_edge);
    sqlite3_finalize(stmt_del_edges);
    sqlite3_finalize(stmt_rev_nbrs);
  }
  return rc;
}
