/*
 * distance.c — SimSIMD-backed distance kernels + SQL scalar functions.
 *
 * Each kernel wraps one SimSIMD runtime-dispatch function.  The runtime
 * dispatcher selects the best SIMD back-end (NEON, SVE, AVX2, …) at the
 * first call using CPUID so there is no per-call overhead.
 *
 * SQL functions registered:
 *   vec_distance_l2(a, b)       → sqrt(L2sq)   ORDER BY ASC → nearest first
 *   vec_distance_cosine(a, b)   → 1 − cos sim  ORDER BY ASC → nearest first
 *   vec_distance_ip(a, b)       → −dot(a,b)    ORDER BY ASC → largest IP first
 *   vec_distance_l1(a, b)       → L1 / taxicab
 *   vec_distance_hamming(a, b)  → bit-level Hamming (bytes = dims × 4)
 *   vec_distance_jaccard(a, b)  → Jaccard distance on bit strings
 */

#include "distance.h"
#include "vec_parse.h"

#include <math.h>
#include <sqlite3ext.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* Pull in SimSIMD's header-only implementation. */
#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include <simsimd/simsimd.h>

/* -------------------------------------------------------------------------- */
/* Kernel wrappers */
/* -------------------------------------------------------------------------- */

static void kernel_l2(const float *a, const float *b, int dims, double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_l2sq_f32(a, b, (simsimd_size_t)dims, &d);
  *out = sqrt(d);
}

static void kernel_cosine(const float *a, const float *b, int dims,
                          double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_cos_f32(a, b, (simsimd_size_t)dims, &d);
  *out = d;
}

static void kernel_ip(const float *a, const float *b, int dims, double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_dot_f32(a, b, (simsimd_size_t)dims, &d);
  /* Negate: ORDER BY ASC gives the highest inner product first, matching
   * pgvector's <#> convention. */
  *out = -d;
}

static void kernel_l1(const float *a, const float *b, int dims, double *out) {
  /* SimSIMD does not expose a top-level l1 dispatch yet; fall through to a
   * simple serial loop that the compiler can auto-vectorise. */
  double acc = 0.0;
  for (int i = 0; i < dims; i++) {
    double diff = (double)a[i] - (double)b[i];
    acc += diff < 0.0 ? -diff : diff;
  }
  *out = acc;
}

static void kernel_hamming(const float *a, const float *b, int dims,
                           double *out) {
  /* Treat the float arrays as packed byte strings.
   * dims floats × 4 bytes = total byte count passed to the b8 kernel. */
  simsimd_distance_t d = 0.0;
  simsimd_hamming_b8((const simsimd_b8_t *)a, (const simsimd_b8_t *)b,
                     (simsimd_size_t)(dims * (int)sizeof(float)), &d);
  *out = d;
}

static void kernel_jaccard(const float *a, const float *b, int dims,
                           double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_jaccard_b8((const simsimd_b8_t *)a, (const simsimd_b8_t *)b,
                     (simsimd_size_t)(dims * (int)sizeof(float)), &d);
  *out = d;
}

/* -------------------------------------------------------------------------- */
/* distance_for_metric */
/* -------------------------------------------------------------------------- */

dist_fn_t distance_for_metric(const char *metric) {
  if (!metric)
    return NULL;
  if (strcmp(metric, "l2") == 0 || strcmp(metric, "l2sq") == 0 ||
      strcmp(metric, "euclidean") == 0)
    return kernel_l2;
  if (strcmp(metric, "cosine") == 0 || strcmp(metric, "cos") == 0)
    return kernel_cosine;
  if (strcmp(metric, "ip") == 0 || strcmp(metric, "dot") == 0 ||
      strcmp(metric, "inner_product") == 0)
    return kernel_ip;
  if (strcmp(metric, "l1") == 0 || strcmp(metric, "taxicab") == 0 ||
      strcmp(metric, "manhattan") == 0)
    return kernel_l1;
  if (strcmp(metric, "hamming") == 0)
    return kernel_hamming;
  if (strcmp(metric, "jaccard") == 0)
    return kernel_jaccard;
  return NULL;
}

/* -------------------------------------------------------------------------- */
/* Helper used by all SQL functions */
/* -------------------------------------------------------------------------- */

/*
 * sql_distance_fn — generic body for each vec_distance_* SQL function.
 * pApp is cast to dist_fn_t.
 */
static void sql_distance_fn(sqlite3_context *ctx, int argc,
                            sqlite3_value **argv) {
  (void)argc;
  dist_fn_t fn = (dist_fn_t)(uintptr_t)sqlite3_user_data(ctx);

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }

  const char *text_a = (const char *)sqlite3_value_text(argv[0]);
  const char *text_b = (const char *)sqlite3_value_text(argv[1]);

  float *va = NULL, *vb = NULL;
  int dims_a = 0, dims_b = 0;
  char *err = NULL;

  if (vec_parse(text_a, &va, &dims_a, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : "vec_distance: invalid first vector",
                         -1);
    sqlite3_free(err);
    return;
  }
  if (vec_parse(text_b, &vb, &dims_b, &err) != SQLITE_OK) {
    sqlite3_free(va);
    sqlite3_result_error(ctx, err ? err : "vec_distance: invalid second vector",
                         -1);
    sqlite3_free(err);
    return;
  }
  if (dims_a != dims_b) {
    sqlite3_free(va);
    sqlite3_free(vb);
    sqlite3_result_error(ctx, "vec_distance: dimension mismatch", -1);
    return;
  }

  double result = 0.0;
  fn(va, vb, dims_a, &result);

  sqlite3_free(va);
  sqlite3_free(vb);
  sqlite3_result_double(ctx, result);
}

/* -------------------------------------------------------------------------- */
/* distance_register_functions */
/* -------------------------------------------------------------------------- */

int distance_register_functions(sqlite3 *db) {
  static const struct {
    const char *name;
    dist_fn_t fn;
  } funcs[] = {
      {"vec_distance_l2", kernel_l2},
      {"vec_distance_cosine", kernel_cosine},
      {"vec_distance_ip", kernel_ip},
      {"vec_distance_l1", kernel_l1},
      {"vec_distance_hamming", kernel_hamming},
      {"vec_distance_jaccard", kernel_jaccard},
  };

  int nfuncs = (int)(sizeof(funcs) / sizeof(funcs[0]));
  for (int i = 0; i < nfuncs; i++) {
    int rc = sqlite3_create_function(
        db, funcs[i].name, 2, /* nArg */
        SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS,
        (void *)(uintptr_t)funcs[i].fn, /* pApp */
        sql_distance_fn, NULL,          /* xStep */
        NULL                            /* xFinal */
    );
    if (rc != SQLITE_OK)
      return rc;
  }
  return SQLITE_OK;
}
