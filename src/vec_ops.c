/*
 * vec_ops.c — Element-wise vector operations and utility functions.
 *
 * SQL functions registered:
 *   vec_add(a, b)          → element-wise sum (requires same dims)
 *   vec_sub(a, b)          → element-wise difference (requires same dims)
 *   vec_normalize(v)       → unit vector (divide by L2 norm)
 *   vec_slice(v, start, end) → subvector extraction (0-indexed, exclusive end)
 *   vec_f32(v)             → float32 BLOB with subtype 223
 *   vec_int8(v)            → int8 BLOB with subtype 225
 *   vec_bit(v)             → bit-packed BLOB with subtype 224
 *   vec_type(v)            → type name string
 */

#include "vec_ops.h"
#include "vec_parse.h"

#include <math.h>
#include <sqlite3ext.h>
#include <stdint.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* -------------------------------------------------------------------------- */
/* vec_add: element-wise sum */
/* -------------------------------------------------------------------------- */

static void sql_vec_add(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;

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
    sqlite3_result_error(ctx, err ? err : "vec_add: invalid first vector", -1);
    sqlite3_free(err);
    return;
  }
  if (vec_parse(text_b, &vb, &dims_b, &err) != SQLITE_OK) {
    sqlite3_free(va);
    sqlite3_result_error(ctx, err ? err : "vec_add: invalid second vector", -1);
    sqlite3_free(err);
    return;
  }
  if (dims_a != dims_b) {
    sqlite3_free(va);
    sqlite3_free(vb);
    sqlite3_result_error(ctx, "vec_add: dimension mismatch", -1);
    return;
  }

  float *result = sqlite3_malloc64(dims_a * sizeof(float));
  if (!result) {
    sqlite3_free(va);
    sqlite3_free(vb);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  for (int i = 0; i < dims_a; i++) {
    result[i] = va[i] + vb[i];
  }

  char *result_text = vec_format(result, dims_a);
  sqlite3_free(va);
  sqlite3_free(vb);
  sqlite3_free(result);

  if (!result_text) {
    sqlite3_result_error_nomem(ctx);
    return;
  }

  sqlite3_result_text(ctx, result_text, -1, sqlite3_free);
}

/* -------------------------------------------------------------------------- */
/* vec_sub: element-wise difference */
/* -------------------------------------------------------------------------- */

static void sql_vec_sub(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;

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
    sqlite3_result_error(ctx, err ? err : "vec_sub: invalid first vector", -1);
    sqlite3_free(err);
    return;
  }
  if (vec_parse(text_b, &vb, &dims_b, &err) != SQLITE_OK) {
    sqlite3_free(va);
    sqlite3_result_error(ctx, err ? err : "vec_sub: invalid second vector", -1);
    sqlite3_free(err);
    return;
  }
  if (dims_a != dims_b) {
    sqlite3_free(va);
    sqlite3_free(vb);
    sqlite3_result_error(ctx, "vec_sub: dimension mismatch", -1);
    return;
  }

  float *result = sqlite3_malloc64(dims_a * sizeof(float));
  if (!result) {
    sqlite3_free(va);
    sqlite3_free(vb);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  for (int i = 0; i < dims_a; i++) {
    result[i] = va[i] - vb[i];
  }

  char *result_text = vec_format(result, dims_a);
  sqlite3_free(va);
  sqlite3_free(vb);
  sqlite3_free(result);

  if (!result_text) {
    sqlite3_result_error_nomem(ctx);
    return;
  }

  sqlite3_result_text(ctx, result_text, -1, sqlite3_free);
}

/* -------------------------------------------------------------------------- */
/* vec_normalize: divide by L2 norm to create unit vector */
/* -------------------------------------------------------------------------- */

static void sql_vec_normalize(sqlite3_context *ctx, int argc,
                              sqlite3_value **argv) {
  (void)argc;

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }

  const char *text = (const char *)sqlite3_value_text(argv[0]);
  float *vec = NULL;
  int dims = 0;
  char *err = NULL;

  if (vec_parse(text, &vec, &dims, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : "vec_normalize: invalid vector", -1);
    sqlite3_free(err);
    return;
  }

  /* Compute L2 norm */
  double norm_sq = 0.0;
  for (int i = 0; i < dims; i++) {
    norm_sq += (double)vec[i] * (double)vec[i];
  }
  double norm = sqrt(norm_sq);

  /* Handle zero vector */
  if (norm < 1e-10) {
    sqlite3_free(vec);
    sqlite3_result_error(ctx, "vec_normalize: zero vector has no unit form",
                         -1);
    return;
  }

  /* Normalize in place */
  for (int i = 0; i < dims; i++) {
    vec[i] = (float)((double)vec[i] / norm);
  }

  char *result_text = vec_format(vec, dims);
  sqlite3_free(vec);

  if (!result_text) {
    sqlite3_result_error_nomem(ctx);
    return;
  }

  sqlite3_result_text(ctx, result_text, -1, sqlite3_free);
}

/* -------------------------------------------------------------------------- */
/* vec_slice: extract subvector [start, end) (0-indexed, exclusive end) */
/* -------------------------------------------------------------------------- */

static void sql_vec_slice(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  (void)argc;

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }

  const char *text = (const char *)sqlite3_value_text(argv[0]);
  int start = sqlite3_value_int(argv[1]);
  int end = sqlite3_value_int(argv[2]);

  float *vec = NULL;
  int dims = 0;
  char *err = NULL;

  if (vec_parse(text, &vec, &dims, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : "vec_slice: invalid vector", -1);
    sqlite3_free(err);
    return;
  }

  /* Validate indices */
  if (start < 0 || end < 0 || start > dims || end > dims || start > end) {
    sqlite3_free(vec);
    sqlite3_result_error(ctx, "vec_slice: invalid slice indices", -1);
    return;
  }

  int slice_len = end - start;
  if (slice_len == 0) {
    sqlite3_free(vec);
    sqlite3_result_error(ctx, "vec_slice: empty slice not allowed", -1);
    return;
  }

  /* Create subvector by copying the slice */
  float *sub = sqlite3_malloc64(slice_len * sizeof(float));
  if (!sub) {
    sqlite3_free(vec);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  memcpy(sub, vec + start, slice_len * sizeof(float));

  char *result_text = vec_format(sub, slice_len);
  sqlite3_free(vec);
  sqlite3_free(sub);

  if (!result_text) {
    sqlite3_result_error_nomem(ctx);
    return;
  }

  sqlite3_result_text(ctx, result_text, -1, sqlite3_free);
}

/* -------------------------------------------------------------------------- */
/* Typed vector constructors: vec_f32, vec_int8, vec_bit, vec_type */
/* -------------------------------------------------------------------------- */

#define VEC_SUBTYPE_F32 223
#define VEC_SUBTYPE_BIT 224
#define VEC_SUBTYPE_INT8 225

/* vec_f32: convert text or blob to float32 BLOB with subtype 223 */
static void sql_vec_f32(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }

  int input_type = sqlite3_value_type(argv[0]);
  if (input_type == SQLITE_BLOB) {
    /* Already a blob, just set subtype and return */
    int bytes = sqlite3_value_bytes(argv[0]);
    const void *blob = sqlite3_value_blob(argv[0]);
    sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
    sqlite3_result_subtype(ctx, VEC_SUBTYPE_F32);
    return;
  }

  /* Parse from text */
  const char *text = (const char *)sqlite3_value_text(argv[0]);
  float *vec = NULL;
  int dims = 0;
  char *err = NULL;

  if (vec_parse(text, &vec, &dims, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : "vec_f32: invalid vector", -1);
    sqlite3_free(err);
    return;
  }

  /* Return as raw float32 blob */
  sqlite3_result_blob(ctx, vec, dims * sizeof(float), sqlite3_free);
  sqlite3_result_subtype(ctx, VEC_SUBTYPE_F32);
}

/* vec_int8: convert text or blob to int8 BLOB with subtype 225 */
static void sql_vec_int8(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }

  int input_type = sqlite3_value_type(argv[0]);
  if (input_type == SQLITE_BLOB) {
    /* Already a blob, just set subtype and return */
    int bytes = sqlite3_value_bytes(argv[0]);
    const void *blob = sqlite3_value_blob(argv[0]);
    sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
    sqlite3_result_subtype(ctx, VEC_SUBTYPE_INT8);
    return;
  }

  /* Parse from text as floats, then convert to int8 */
  const char *text = (const char *)sqlite3_value_text(argv[0]);
  float *vec = NULL;
  int dims = 0;
  char *err = NULL;

  if (vec_parse(text, &vec, &dims, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : "vec_int8: invalid vector", -1);
    sqlite3_free(err);
    return;
  }

  /* Convert to int8 */
  int8_t *int8_vec = sqlite3_malloc64(dims);
  if (!int8_vec) {
    sqlite3_free(vec);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  for (int i = 0; i < dims; i++) {
    /* Clamp to int8 range [-128, 127] */
    float v = vec[i];
    if (v < -128.0f)
      v = -128.0f;
    if (v > 127.0f)
      v = 127.0f;
    int8_vec[i] = (int8_t)v;
  }

  sqlite3_free(vec);
  sqlite3_result_blob(ctx, int8_vec, dims, sqlite3_free);
  sqlite3_result_subtype(ctx, VEC_SUBTYPE_INT8);
}

/* vec_bit: validate and tag a bit-packed blob with subtype 224 */
static void sql_vec_bit(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_error(ctx, "vec_bit: input must be a BLOB", -1);
    return;
  }

  int bytes = sqlite3_value_bytes(argv[0]);
  const void *blob = sqlite3_value_blob(argv[0]);

  /* Just validate it's a blob and tag it */
  sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
  sqlite3_result_subtype(ctx, VEC_SUBTYPE_BIT);
}

/* vec_type: return the type name for a typed vector */
static void sql_vec_type(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }

  unsigned int subtype = sqlite3_value_subtype(argv[0]);
  const char *type_name = NULL;

  switch (subtype) {
  case VEC_SUBTYPE_F32:
    type_name = "float32";
    break;
  case VEC_SUBTYPE_BIT:
    type_name = "bit";
    break;
  case VEC_SUBTYPE_INT8:
    type_name = "int8";
    break;
  case 0:
    /* No subtype - could be text vector */
    if (sqlite3_value_type(argv[0]) == SQLITE_TEXT) {
      type_name = "text";
    } else {
      type_name = "unknown";
    }
    break;
  default:
    type_name = "unknown";
    break;
  }

  sqlite3_result_text(ctx, type_name, -1, SQLITE_STATIC);
}

/* -------------------------------------------------------------------------- */
/* vec_ops_register_functions */
/* -------------------------------------------------------------------------- */

int vec_ops_register_functions(sqlite3 *db) {
  int rc;

  rc = sqlite3_create_function(
      db, "vec_add", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS,
      NULL, sql_vec_add, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_function(
      db, "vec_sub", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS,
      NULL, sql_vec_sub, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_function(db, "vec_normalize", 1,
                               SQLITE_UTF8 | SQLITE_DETERMINISTIC |
                                   SQLITE_INNOCUOUS,
                               NULL, sql_vec_normalize, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_function(
      db, "vec_slice", 3, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS,
      NULL, sql_vec_slice, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_function(db, "vec_f32", 1,
                               SQLITE_UTF8 | SQLITE_DETERMINISTIC |
                                   SQLITE_INNOCUOUS | SQLITE_RESULT_SUBTYPE,
                               NULL, sql_vec_f32, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_function(db, "vec_int8", 1,
                               SQLITE_UTF8 | SQLITE_DETERMINISTIC |
                                   SQLITE_INNOCUOUS | SQLITE_RESULT_SUBTYPE,
                               NULL, sql_vec_int8, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_function(db, "vec_bit", 1,
                               SQLITE_UTF8 | SQLITE_DETERMINISTIC |
                                   SQLITE_INNOCUOUS | SQLITE_RESULT_SUBTYPE,
                               NULL, sql_vec_bit, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_function(db, "vec_type", 1,
                               SQLITE_UTF8 | SQLITE_DETERMINISTIC |
                                   SQLITE_INNOCUOUS | SQLITE_SUBTYPE,
                               NULL, sql_vec_type, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  return SQLITE_OK;
}
