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
#include "vtab.h"

#include <math.h>
#include <sqlite3ext.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* Pull in SimSIMD's header-only implementation. */
#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include <simsimd/simsimd.h>

/* -------------------------------------------------------------------------- */
/* Float32 kernel wrappers */
/* -------------------------------------------------------------------------- */

static void kernel_l2(const void *a, const void *b, int dims, double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_l2sq_f32((const simsimd_f32_t *)a, (const simsimd_f32_t *)b,
                   (simsimd_size_t)dims, &d);
  *out = sqrt(d);
}

static void kernel_cosine(const void *a, const void *b, int dims, double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_cos_f32((const simsimd_f32_t *)a, (const simsimd_f32_t *)b,
                  (simsimd_size_t)dims, &d);
  *out = d;
}

static void kernel_ip(const void *a, const void *b, int dims, double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_dot_f32((const simsimd_f32_t *)a, (const simsimd_f32_t *)b,
                  (simsimd_size_t)dims, &d);
  *out = -d;
}

static void kernel_l1(const void *a, const void *b, int dims, double *out) {
  const float *fa = (const float *)a;
  const float *fb = (const float *)b;
  double acc = 0.0;
  for (int i = 0; i < dims; i++) {
    double diff = (double)fa[i] - (double)fb[i];
    acc += diff < 0.0 ? -diff : diff;
  }
  *out = acc;
}

static void kernel_hamming(const void *a, const void *b, int dims,
                           double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_hamming_b8((const simsimd_b8_t *)a, (const simsimd_b8_t *)b,
                     (simsimd_size_t)(dims * (int)sizeof(float)), &d);
  *out = d;
}

static void kernel_jaccard(const void *a, const void *b, int dims,
                           double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_jaccard_b8((const simsimd_b8_t *)a, (const simsimd_b8_t *)b,
                     (simsimd_size_t)(dims * (int)sizeof(float)), &d);
  *out = d;
}

/* -------------------------------------------------------------------------- */
/* Int8 kernel wrappers */
/* -------------------------------------------------------------------------- */

static void kernel_l2_i8(const void *a, const void *b, int dims, double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_l2sq_i8((const simsimd_i8_t *)a, (const simsimd_i8_t *)b,
                  (simsimd_size_t)dims, &d);
  *out = sqrt(d);
}

static void kernel_cosine_i8(const void *a, const void *b, int dims,
                             double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_cos_i8((const simsimd_i8_t *)a, (const simsimd_i8_t *)b,
                 (simsimd_size_t)dims, &d);
  *out = d;
}

static void kernel_ip_i8(const void *a, const void *b, int dims, double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_dot_i8((const simsimd_i8_t *)a, (const simsimd_i8_t *)b,
                 (simsimd_size_t)dims, &d);
  *out = -d;
}

static void kernel_l1_i8(const void *a, const void *b, int dims, double *out) {
  const int8_t *ia = (const int8_t *)a;
  const int8_t *ib = (const int8_t *)b;
  double acc = 0.0;
  for (int i = 0; i < dims; i++) {
    int diff = (int)ia[i] - (int)ib[i];
    acc += diff < 0 ? -diff : diff;
  }
  *out = acc;
}

/* -------------------------------------------------------------------------- */
/* Binary (bit-packed) kernel wrappers — dims = number of bits */
/* -------------------------------------------------------------------------- */

static void kernel_hamming_bit(const void *a, const void *b, int dims,
                               double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_hamming_b8((const simsimd_b8_t *)a, (const simsimd_b8_t *)b,
                     (simsimd_size_t)((dims + 7) / 8), &d);
  *out = d;
}

static void kernel_jaccard_bit(const void *a, const void *b, int dims,
                               double *out) {
  simsimd_distance_t d = 0.0;
  simsimd_jaccard_b8((const simsimd_b8_t *)a, (const simsimd_b8_t *)b,
                     (simsimd_size_t)((dims + 7) / 8), &d);
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
/* distance_for_metric_typed — resolve kernel for metric + storage type */
/* -------------------------------------------------------------------------- */

dist_fn_t distance_for_metric_typed(const char *metric, int storage_type) {
  if (!metric)
    return NULL;

  if (storage_type == VEC_STORAGE_INT8) {
    if (strcmp(metric, "l2") == 0 || strcmp(metric, "l2sq") == 0 ||
        strcmp(metric, "euclidean") == 0)
      return kernel_l2_i8;
    if (strcmp(metric, "cosine") == 0 || strcmp(metric, "cos") == 0)
      return kernel_cosine_i8;
    if (strcmp(metric, "ip") == 0 || strcmp(metric, "dot") == 0 ||
        strcmp(metric, "inner_product") == 0)
      return kernel_ip_i8;
    if (strcmp(metric, "l1") == 0 || strcmp(metric, "taxicab") == 0 ||
        strcmp(metric, "manhattan") == 0)
      return kernel_l1_i8;
    return NULL; /* int8 does not support hamming/jaccard */
  }

  if (storage_type == VEC_STORAGE_BIT) {
    if (strcmp(metric, "hamming") == 0)
      return kernel_hamming_bit;
    if (strcmp(metric, "jaccard") == 0)
      return kernel_jaccard_bit;
    return NULL; /* binary only supports hamming/jaccard */
  }

  /* VEC_STORAGE_F32 — default */
  return distance_for_metric(metric);
}

/* -------------------------------------------------------------------------- */
/* Helper used by all SQL functions */
/* -------------------------------------------------------------------------- */

/*
 * sql_distance_fn — generic body for each vec_distance_* SQL function.
 * pApp is cast to dist_fn_t.  Parses text → float32.
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

/*
 * sql_distance_fn_i8 — generic body for vec_distance_*_i8 functions.
 * Parses text → int8.
 */
static void sql_distance_fn_i8(sqlite3_context *ctx, int argc,
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

  int8_t *va = NULL, *vb = NULL;
  int dims_a = 0, dims_b = 0;
  char *err = NULL;

  if (vec_parse_int8(text_a, &va, &dims_a, &err) != SQLITE_OK) {
    sqlite3_result_error(
        ctx, err ? err : "vec_distance_i8: invalid first vector", -1);
    sqlite3_free(err);
    return;
  }
  if (vec_parse_int8(text_b, &vb, &dims_b, &err) != SQLITE_OK) {
    sqlite3_free(va);
    sqlite3_result_error(
        ctx, err ? err : "vec_distance_i8: invalid second vector", -1);
    sqlite3_free(err);
    return;
  }
  if (dims_a != dims_b) {
    sqlite3_free(va);
    sqlite3_free(vb);
    sqlite3_result_error(ctx, "vec_distance_i8: dimension mismatch", -1);
    return;
  }

  double result = 0.0;
  fn(va, vb, dims_a, &result);

  sqlite3_free(va);
  sqlite3_free(vb);
  sqlite3_result_double(ctx, result);
}

/*
 * sql_distance_fn_bit — generic body for vec_distance_*_bit functions.
 * Both inputs must be BLOBs of equal length.
 */
static void sql_distance_fn_bit(sqlite3_context *ctx, int argc,
                                sqlite3_value **argv) {
  (void)argc;
  dist_fn_t fn = (dist_fn_t)(uintptr_t)sqlite3_user_data(ctx);

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
      sqlite3_value_type(argv[1]) != SQLITE_BLOB) {
    sqlite3_result_error(ctx, "vec_distance_bit: inputs must be BLOBs", -1);
    return;
  }

  int bytes_a = sqlite3_value_bytes(argv[0]);
  int bytes_b = sqlite3_value_bytes(argv[1]);
  const void *blob_a = sqlite3_value_blob(argv[0]);
  const void *blob_b = sqlite3_value_blob(argv[1]);

  if (bytes_a <= 0 || bytes_b <= 0) {
    sqlite3_result_error(ctx, "vec_distance_bit: empty blob not allowed", -1);
    return;
  }
  if (bytes_a != bytes_b) {
    sqlite3_result_error(ctx, "vec_distance_bit: blob size mismatch", -1);
    return;
  }

  int bits = bytes_a * 8;
  double result = 0.0;
  fn(blob_a, blob_b, bits, &result);

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

  /* Int8 typed distance functions */
  static const struct {
    const char *name;
    dist_fn_t fn;
  } i8_funcs[] = {
      {"vec_distance_l2_i8", kernel_l2_i8},
      {"vec_distance_cosine_i8", kernel_cosine_i8},
      {"vec_distance_ip_i8", kernel_ip_i8},
      {"vec_distance_l1_i8", kernel_l1_i8},
  };

  int ni8 = (int)(sizeof(i8_funcs) / sizeof(i8_funcs[0]));
  for (int i = 0; i < ni8; i++) {
    int rc = sqlite3_create_function(
        db, i8_funcs[i].name, 2,
        SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS,
        (void *)(uintptr_t)i8_funcs[i].fn, sql_distance_fn_i8, NULL, NULL);
    if (rc != SQLITE_OK)
      return rc;
  }

  /* Binary typed distance functions */
  static const struct {
    const char *name;
    dist_fn_t fn;
  } bit_funcs[] = {
      {"vec_distance_hamming_bit", kernel_hamming_bit},
      {"vec_distance_jaccard_bit", kernel_jaccard_bit},
  };

  int nbit = (int)(sizeof(bit_funcs) / sizeof(bit_funcs[0]));
  for (int i = 0; i < nbit; i++) {
    int rc = sqlite3_create_function(
        db, bit_funcs[i].name, 2,
        SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS,
        (void *)(uintptr_t)bit_funcs[i].fn, sql_distance_fn_bit, NULL, NULL);
    if (rc != SQLITE_OK)
      return rc;
  }

  return SQLITE_OK;
}
