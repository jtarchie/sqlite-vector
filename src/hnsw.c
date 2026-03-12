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
                             const void **out_vec, int dims, int storage_type) {
  sqlite3_reset(stmt_get_vec);
  sqlite3_bind_int64(stmt_get_vec, 1, id);
  int rc = sqlite3_step(stmt_get_vec);
  if (rc != SQLITE_ROW)
    return SQLITE_ERROR;
  int bytes = sqlite3_column_bytes(stmt_get_vec, 0);
  if (bytes != vec_blob_bytes(storage_type, dims))
    return SQLITE_ERROR;
  *out_vec = sqlite3_column_blob(stmt_get_vec, 0);
  return SQLITE_OK;
}

/* Persist a single integer config value via cached statement.
 * When ctx->defer_config is set the DB write is suppressed (xCommit flushes).
 */
static int update_config(HnswCtx *ctx, const char *key, sqlite3_int64 val) {
  if (ctx->defer_config)
    return SQLITE_OK;
  char buf[32];
  sqlite3_snprintf((int)sizeof(buf), buf, "%lld", val);
  sqlite3_reset(ctx->sc_upd_config);
  sqlite3_bind_text(ctx->sc_upd_config, 1, buf, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(ctx->sc_upd_config, 2, key, -1, SQLITE_STATIC);
  int rc = sqlite3_step(ctx->sc_upd_config);
  return (rc == SQLITE_DONE) ? SQLITE_OK : rc;
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
static int search_layer(HnswCtx *ctx, const void *query, sqlite3_int64 ep_id,
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
      const void *e_vec = NULL;
      if (fetch_vector_stmt(stmt_get_vec, e_id, &e_vec, ctx->dims,
                            ctx->storage_type) != SQLITE_OK)
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
int hnsw_search(HnswCtx *ctx, const void *query_vec, int k,
                HnswResult **out_results, int *out_n) {
  *out_results = NULL;
  *out_n = 0;

  if (ctx->entry_point < 0)
    return SQLITE_OK; /* empty graph */

  int rc = SQLITE_OK;
  sqlite3_stmt *stmt_get_nbrs = ctx->sc_get_nbrs;
  sqlite3_stmt *stmt_get_vec = ctx->sc_get_vec;

  /* Distance from query to entry point */
  const void *ep_vec = NULL;
  rc = fetch_vector_stmt(stmt_get_vec, ctx->entry_point, &ep_vec, ctx->dims,
                         ctx->storage_type);
  if (rc != SQLITE_OK)
    return rc;

  double ep_dist = 0.0;
  ctx->dist_fn(query_vec, ep_vec, ctx->dims, &ep_dist);

  sqlite3_int64 cur_ep = ctx->entry_point;
  double cur_ep_dist = ep_dist;

  /* Greedy descent: layers max_layer → 1 (ef=1) */
  for (int lc = ctx->max_layer; lc >= 1; lc--) {
    Heap W;
    rc = heap_init(&W);
    if (rc != SQLITE_OK)
      return rc;
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
      return rc;
  }

  /* Full beam search at layer 0 */
  int ef = ctx->ef_search > k ? ctx->ef_search : k;
  Heap W0;
  rc = heap_init(&W0);
  if (rc != SQLITE_OK)
    return rc;
  rc = search_layer(ctx, query_vec, cur_ep, cur_ep_dist, ef, 0, stmt_get_nbrs,
                    stmt_get_vec, &W0);
  if (rc != SQLITE_OK) {
    heap_free(&W0);
    return rc;
  }

  /* Prune W0 to k nearest */
  while (W0.len > k)
    maxheap_pop(&W0);

  int n = W0.len;
  *out_n = n;
  if (n == 0) {
    heap_free(&W0);
    return rc;
  }

  *out_results = sqlite3_malloc(n * (int)sizeof(HnswResult));
  if (!*out_results) {
    heap_free(&W0);
    return SQLITE_NOMEM;
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

  int n = W->len;
  if (n == 0)
    return SQLITE_OK;

  /* Sort W->data in-place ascending by distance so callers can reuse */
  for (int i = 1; i < n; i++) {
    HNode key = W->data[i];
    int j = i - 1;
    while (j >= 0 && W->data[j].dist > key.dist) {
      W->data[j + 1] = W->data[j];
      j--;
    }
    W->data[j + 1] = key;
  }

  int nconn = n < M ? n : M;
  for (int i = 0; i < nconn; i++) {
    sqlite3_reset(stmt_ins_edge);
    sqlite3_bind_int(stmt_ins_edge, 1, lc);
    sqlite3_bind_int64(stmt_ins_edge, 2, node_id);
    sqlite3_bind_int64(stmt_ins_edge, 3, W->data[i].id);
    sqlite3_bind_double(stmt_ins_edge, 4, W->data[i].dist);
    sqlite3_step(stmt_ins_edge);
  }

  return SQLITE_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * hnsw_insert — Algorithm 1 (insert) from the HNSW paper.
 * ══════════════════════════════════════════════════════════════════════════ */
int hnsw_insert(HnswCtx *ctx, sqlite3_int64 rowid, const void *vec) {
  int rc = SQLITE_OK;

  /* Sample random level: l = floor(-ln(U(0,1)) / ln(M)) */
  double r = (double)(rand() + 1) / ((double)RAND_MAX + 2.0);
  int node_level = (int)(-log(r) / log((double)(ctx->m > 1 ? ctx->m : 2)));
  if (node_level < 0)
    node_level = 0;

  sqlite3_stmt *stmt_get_nbrs = ctx->sc_get_nbrs;
  sqlite3_stmt *stmt_get_vec = ctx->sc_get_vec;
  sqlite3_stmt *stmt_ins_edge = ctx->sc_ins_edge;
  sqlite3_stmt *stmt_del_edges = ctx->sc_del_edges;
  sqlite3_stmt *stmt_ins_layer = ctx->sc_ins_layer;
  sqlite3_stmt *stmt_nbr_count = ctx->sc_nbr_count;

  /* Write this node's max layer to _layers */
  sqlite3_reset(stmt_ins_layer);
  sqlite3_bind_int64(stmt_ins_layer, 1, rowid);
  sqlite3_bind_int(stmt_ins_layer, 2, node_level);
  rc = sqlite3_step(stmt_ins_layer);
  rc = (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
  if (rc != SQLITE_OK)
    return rc;

  /* If graph is empty, this node becomes the sole entry point */
  if (ctx->entry_point < 0) {
    ctx->entry_point = rowid;
    ctx->max_layer = node_level;
    update_config(ctx, "entry_point", rowid);
    update_config(ctx, "max_layer", node_level);
    return SQLITE_OK;
  }

  /* Distance from new node to current entry point */
  const void *ep_vec = NULL;
  rc = fetch_vector_stmt(stmt_get_vec, ctx->entry_point, &ep_vec, ctx->dims,
                         ctx->storage_type);
  if (rc != SQLITE_OK)
    return rc;

  double ep_dist = 0.0;
  ctx->dist_fn(vec, ep_vec, ctx->dims, &ep_dist);

  sqlite3_int64 cur_ep = ctx->entry_point;
  double cur_ep_dist = ep_dist;

  /* Greedy descent from max_layer → node_level+1 (ef=1) */
  for (int lc = ctx->max_layer; lc > node_level; lc--) {
    Heap W;
    rc = heap_init(&W);
    if (rc != SQLITE_OK)
      return rc;
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
      return rc;
  }

  /* Build connections at each layer from min(max_layer, node_level) down to 0
   */
  int connect_top = ctx->max_layer < node_level ? ctx->max_layer : node_level;

  for (int lc = connect_top; lc >= 0; lc--) {
    int Mmax = (lc == 0) ? ctx->m * 2 : ctx->m;

    Heap W;
    rc = heap_init(&W);
    if (rc != SQLITE_OK)
      return rc;
    rc = search_layer(ctx, vec, cur_ep, cur_ep_dist, ctx->ef_construction, lc,
                      stmt_get_nbrs, stmt_get_vec, &W);
    if (rc != SQLITE_OK) {
      heap_free(&W);
      return rc;
    }

    if (W.len == 0) {
      heap_free(&W);
      continue;
    }

    /* Connect new node → its M nearest neighbours.
     * set_connections sorts W.data in-place ascending by distance. */
    rc = set_connections(ctx, lc, rowid, &W, Mmax, stmt_ins_edge,
                         stmt_del_edges);
    if (rc != SQLITE_OK) {
      heap_free(&W);
      return rc;
    }

    /* Add reverse edges using the already-sorted W.data from set_connections */
    int nconn = W.len < Mmax ? W.len : Mmax;

    for (int i = 0; i < nconn; i++) {
      sqlite3_int64 nbr_id = W.data[i].id;
      double nbr_dist = W.data[i].dist;

      /* Insert reverse edge: nbr_id → rowid */
      sqlite3_reset(stmt_ins_edge);
      sqlite3_bind_int(stmt_ins_edge, 1, lc);
      sqlite3_bind_int64(stmt_ins_edge, 2, nbr_id);
      sqlite3_bind_int64(stmt_ins_edge, 3, rowid);
      sqlite3_bind_double(stmt_ins_edge, 4, nbr_dist);
      sqlite3_step(stmt_ins_edge);

      /* Count current out-edges of nbr_id at lc */
      sqlite3_reset(stmt_nbr_count);
      sqlite3_bind_int(stmt_nbr_count, 1, lc);
      sqlite3_bind_int64(stmt_nbr_count, 2, nbr_id);
      int nbr_conn = 0;
      if (sqlite3_step(stmt_nbr_count) == SQLITE_ROW)
        nbr_conn = sqlite3_column_int(stmt_nbr_count, 0);

      if (nbr_conn > Mmax) {
        /* Fetch all existing edges for nbr_id and prune to Mmax */
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
        sqlite3_reset(stmt_get_nbrs);

        set_connections(ctx, lc, nbr_id, &nbr_W, Mmax, stmt_ins_edge,
                        stmt_del_edges);
        heap_free(&nbr_W);
      }
    }

    /* Advance ep to nearest found in W for next layer */
    cur_ep = W.data[0].id;
    cur_ep_dist = W.data[0].dist;
    heap_free(&W);
  }

  /* Update entry point if this node's level exceeds the current maximum */
  if (node_level > ctx->max_layer) {
    ctx->entry_point = rowid;
    ctx->max_layer = node_level;
    update_config(ctx, "entry_point", rowid);
    update_config(ctx, "max_layer", node_level);
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
  int rc = SQLITE_OK;

  sqlite3_stmt *stmt_get_nbrs = ctx->sc_get_nbrs;
  sqlite3_stmt *stmt_get_vec = ctx->sc_get_vec;
  sqlite3_stmt *stmt_ins_edge = ctx->sc_ins_edge;
  sqlite3_stmt *stmt_del_edges = ctx->sc_del_edges;
  sqlite3_stmt *stmt_rev_nbrs = ctx->sc_rev_nbrs;

  /* ── Step 1: Get deleted node's max layer ───────────────────────────── */
  int d_max_layer = 0;
  sqlite3_reset(ctx->sc_get_layer);
  sqlite3_bind_int64(ctx->sc_get_layer, 1, rowid);
  if (sqlite3_step(ctx->sc_get_layer) == SQLITE_ROW)
    d_max_layer = sqlite3_column_int(ctx->sc_get_layer, 0);

  /* ── Step 2: Collect repair targets before edge deletion ─────────────── */
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
          sqlite3_free(repairs);
          return SQLITE_NOMEM;
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
  sqlite3_reset(stmt_rev_nbrs);
  sqlite3_reset(ctx->sc_del_layer);
  sqlite3_bind_int64(ctx->sc_del_layer, 1, rowid);
  rc = sqlite3_step(ctx->sc_del_layer);
  rc = (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
  if (rc != SQLITE_OK) {
    sqlite3_free(repairs);
    return rc;
  }

  /* ── Step 4: Delete all graph edges touching this node ───────────────── */
  sqlite3_reset(ctx->sc_del_node_edges);
  sqlite3_bind_int64(ctx->sc_del_node_edges, 1, rowid);
  sqlite3_bind_int64(ctx->sc_del_node_edges, 2, rowid);
  rc = sqlite3_step(ctx->sc_del_node_edges);
  rc = (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
  if (rc != SQLITE_OK) {
    sqlite3_free(repairs);
    return rc;
  }

  /* ── Step 5: Entry-point election ────────────────────────────────────── */
  if (ctx->entry_point == rowid) {
    sqlite3_reset(ctx->sc_elect_ep);
    if (sqlite3_step(ctx->sc_elect_ep) == SQLITE_ROW) {
      ctx->entry_point = sqlite3_column_int64(ctx->sc_elect_ep, 0);
      ctx->max_layer = sqlite3_column_int(ctx->sc_elect_ep, 1);
    } else {
      ctx->entry_point = -1;
      ctx->max_layer = 0;
    }
    update_config(ctx, "entry_point", ctx->entry_point);
    update_config(ctx, "max_layer", (sqlite3_int64)ctx->max_layer);
  }

  /* ── Step 6: Graph repair ─────────────────────────────────────────────── */
  if (ctx->entry_point >= 0 && n_repairs > 0) {
    int vec_bytes = vec_blob_bytes(ctx->storage_type, ctx->dims);
    void *n_vec_copy = sqlite3_malloc(vec_bytes);
    if (!n_vec_copy) {
      sqlite3_free(repairs);
      return SQLITE_NOMEM;
    }

    for (int i = 0; i < n_repairs && rc == SQLITE_OK; i++) {
      int lc = repairs[i].lc;
      sqlite3_int64 nid = repairs[i].nid;

      const void *n_raw = NULL;
      if (fetch_vector_stmt(stmt_get_vec, nid, &n_raw, ctx->dims,
                            ctx->storage_type) != SQLITE_OK)
        continue;
      memcpy(n_vec_copy, n_raw, vec_bytes);

      const void *ep_raw = NULL;
      if (fetch_vector_stmt(stmt_get_vec, ctx->entry_point, &ep_raw, ctx->dims,
                            ctx->storage_type) != SQLITE_OK)
        continue;
      double ep_dist = 0.0;
      ctx->dist_fn(n_vec_copy, ep_raw, ctx->dims, &ep_dist);

      Heap W;
      rc = heap_init(&W);
      if (rc != SQLITE_OK)
        break;

      rc = search_layer(ctx, n_vec_copy, ctx->entry_point, ep_dist,
                        ctx->ef_construction, lc, stmt_get_nbrs, stmt_get_vec,
                        &W);

      /* Remove N itself from the candidate set to avoid self-edges */
      if (rc == SQLITE_OK) {
        int dst = 0;
        for (int j = 0; j < W.len; j++) {
          if (W.data[j].id != nid)
            W.data[dst++] = W.data[j];
        }
        W.len = dst;
      }

      if (rc == SQLITE_OK && W.len > 0) {
        int Mmax = (lc == 0) ? ctx->m * 2 : ctx->m;

        /* Re-wire N's edges; set_connections sorts W in-place */
        rc = set_connections(ctx, lc, nid, &W, Mmax, stmt_ins_edge,
                             stmt_del_edges);

        /* Add reverse edges using sorted W.data */
        if (rc == SQLITE_OK) {
          int nconn = W.len < Mmax ? W.len : Mmax;
          for (int j = 0; j < nconn; j++) {
            sqlite3_int64 nbr_id = W.data[j].id;
            sqlite3_reset(stmt_ins_edge);
            sqlite3_bind_int(stmt_ins_edge, 1, lc);
            sqlite3_bind_int64(stmt_ins_edge, 2, nbr_id);
            sqlite3_bind_int64(stmt_ins_edge, 3, nid);
            sqlite3_bind_double(stmt_ins_edge, 4, W.data[j].dist);
            sqlite3_step(stmt_ins_edge);

            /* Prune if neighbor now exceeds Mmax edges */
            sqlite3_reset(ctx->sc_nbr_count);
            sqlite3_bind_int(ctx->sc_nbr_count, 1, lc);
            sqlite3_bind_int64(ctx->sc_nbr_count, 2, nbr_id);
            int nbr_conn = 0;
            if (sqlite3_step(ctx->sc_nbr_count) == SQLITE_ROW)
              nbr_conn = sqlite3_column_int(ctx->sc_nbr_count, 0);

            if (nbr_conn > Mmax) {
              sqlite3_reset(stmt_get_nbrs);
              sqlite3_bind_int(stmt_get_nbrs, 1, lc);
              sqlite3_bind_int64(stmt_get_nbrs, 2, nbr_id);

              Heap nbr_W;
              heap_init(&nbr_W);
              while (sqlite3_step(stmt_get_nbrs) == SQLITE_ROW) {
                sqlite3_int64 n2 = sqlite3_column_int64(stmt_get_nbrs, 0);
                double nd = sqlite3_column_double(stmt_get_nbrs, 1);
                maxheap_push(&nbr_W, n2, nd);
              }
              sqlite3_reset(stmt_get_nbrs);

              set_connections(ctx, lc, nbr_id, &nbr_W, Mmax, stmt_ins_edge,
                              stmt_del_edges);
              heap_free(&nbr_W);
            }
          }
        }
      }
      heap_free(&W);
    }
    sqlite3_free(n_vec_copy);
  }

  sqlite3_free(repairs);
  return rc;
}
