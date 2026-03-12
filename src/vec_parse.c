#include "vec_parse.h"

#include <math.h>
#include <sqlite3ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

SQLITE_EXTENSION_INIT3

/* ── Parsing ─────────────────────────────────────────────────────────────────
 */

int vec_parse(const char *text, float **out_vec, int *out_dims, char **pzErr) {
  const char *p = text;

  /* Skip leading whitespace */
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
    p++;

  if (*p != '[') {
    *pzErr = sqlite3_mprintf("vec: expected '[' at start, got '%.10s'", p);
    return SQLITE_ERROR;
  }
  p++; /* consume '[' */

  int capacity = 16;
  int ndims = 0;
  float *vec = sqlite3_malloc64(capacity * sizeof(float));
  if (!vec)
    return SQLITE_NOMEM;

  while (1) {
    /* Skip whitespace and commas between values */
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
      p++;

    if (*p == ']') {
      p++; /* consume ']' */
      break;
    }

    if (*p == '\0') {
      sqlite3_free(vec);
      *pzErr = sqlite3_mprintf("vec: unterminated vector, expected ']'");
      return SQLITE_ERROR;
    }

    /* Parse one float */
    char *end;
    float val = strtof(p, &end);
    if (end == p) {
      sqlite3_free(vec);
      *pzErr = sqlite3_mprintf("vec: invalid float value '%.20s'", p);
      return SQLITE_ERROR;
    }
    p = end;

    /* Grow the array if needed */
    if (ndims == capacity) {
      capacity *= 2;
      float *tmp = sqlite3_realloc64(vec, capacity * sizeof(float));
      if (!tmp) {
        sqlite3_free(vec);
        return SQLITE_NOMEM;
      }
      vec = tmp;
    }
    vec[ndims++] = val;
  }

  if (ndims == 0) {
    sqlite3_free(vec);
    *pzErr = sqlite3_mprintf("vec: empty vector not allowed");
    return SQLITE_ERROR;
  }

  *out_vec = vec;
  *out_dims = ndims;
  return SQLITE_OK;
}

/* ── Formatting ──────────────────────────────────────────────────────────────
 */

char *vec_format(const float *vec, int dims) {
  /* Upper-bound estimate: '[' + dims * (15 chars for float + 1 comma) + ']' */
  int bufsize = 2 + dims * 16 + 1;
  char *buf = sqlite3_malloc(bufsize);
  if (!buf)
    return NULL;

  int pos = 0;
  buf[pos++] = '[';
  for (int i = 0; i < dims; i++) {
    if (i > 0)
      buf[pos++] = ',';
    /* %.7g gives 7 significant digits — matches float32 precision */
    int n = snprintf(buf + pos, bufsize - pos, "%.7g", (double)vec[i]);
    if (n < 0 || pos + n >= bufsize) {
      /* Shouldn't happen with our estimate, but handle gracefully */
      sqlite3_free(buf);
      return NULL;
    }
    pos += n;
  }
  buf[pos++] = ']';
  buf[pos] = '\0';
  return buf;
}

/* ── Int8 Parsing ────────────────────────────────────────────────────────────
 */

int vec_parse_int8(const char *text, int8_t **out_vec, int *out_dims,
                   char **pzErr) {
  const char *p = text;

  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
    p++;

  if (*p != '[') {
    *pzErr = sqlite3_mprintf("vec_int8: expected '[' at start, got '%.10s'", p);
    return SQLITE_ERROR;
  }
  p++;

  int capacity = 16;
  int ndims = 0;
  int8_t *vec = sqlite3_malloc64(capacity);
  if (!vec)
    return SQLITE_NOMEM;

  while (1) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
      p++;

    if (*p == ']') {
      p++;
      break;
    }

    if (*p == '\0') {
      sqlite3_free(vec);
      *pzErr = sqlite3_mprintf("vec_int8: unterminated vector, expected ']'");
      return SQLITE_ERROR;
    }

    char *end;
    long val = strtol(p, &end, 10);
    if (end == p) {
      sqlite3_free(vec);
      *pzErr = sqlite3_mprintf("vec_int8: invalid integer value '%.20s'", p);
      return SQLITE_ERROR;
    }
    if (val < -128 || val > 127) {
      sqlite3_free(vec);
      *pzErr =
          sqlite3_mprintf("vec_int8: value %ld out of range [-128, 127]", val);
      return SQLITE_ERROR;
    }
    p = end;

    if (ndims == capacity) {
      capacity *= 2;
      int8_t *tmp = sqlite3_realloc64(vec, capacity);
      if (!tmp) {
        sqlite3_free(vec);
        return SQLITE_NOMEM;
      }
      vec = tmp;
    }
    vec[ndims++] = (int8_t)val;
  }

  if (ndims == 0) {
    sqlite3_free(vec);
    *pzErr = sqlite3_mprintf("vec_int8: empty vector not allowed");
    return SQLITE_ERROR;
  }

  *out_vec = vec;
  *out_dims = ndims;
  return SQLITE_OK;
}

/* ── Bit-packed Validation ───────────────────────────────────────────────────
 */

int vec_parse_bit(const void *blob, int bytes, int *out_bits, char **pzErr) {
  if (!blob || bytes <= 0) {
    *pzErr = sqlite3_mprintf("vec_bit: empty blob not allowed");
    return SQLITE_ERROR;
  }
  *out_bits = bytes * 8;
  return SQLITE_OK;
}

/* ── SQL scalar functions ────────────────────────────────────────────────────
 */

/* vec(text) → TEXT  Validates and normalises a vector literal. */
static void sqlVecFunc(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;
  const char *text = (const char *)sqlite3_value_text(argv[0]);
  if (!text) {
    sqlite3_result_null(ctx);
    return;
  }

  float *vec = NULL;
  int dims = 0;
  char *err = NULL;

  if (vec_parse(text, &vec, &dims, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err, -1);
    sqlite3_free(err);
    return;
  }

  char *out = vec_format(vec, dims);
  sqlite3_free(vec);
  if (!out) {
    sqlite3_result_error_nomem(ctx);
    return;
  }

  sqlite3_result_text(ctx, out, -1, sqlite3_free);
}

/* vec_dims(text) → INTEGER  Returns the number of dimensions. */
static void sqlVecDimsFunc(sqlite3_context *ctx, int argc,
                           sqlite3_value **argv) {
  (void)argc;
  const char *text = (const char *)sqlite3_value_text(argv[0]);
  if (!text) {
    sqlite3_result_null(ctx);
    return;
  }

  float *vec = NULL;
  int dims = 0;
  char *err = NULL;

  if (vec_parse(text, &vec, &dims, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err, -1);
    sqlite3_free(err);
    return;
  }

  sqlite3_free(vec);
  sqlite3_result_int(ctx, dims);
}

/* vec_norm(text) → REAL  Returns the L2 (Euclidean) norm ‖v‖₂. */
static void sqlVecNormFunc(sqlite3_context *ctx, int argc,
                           sqlite3_value **argv) {
  (void)argc;
  const char *text = (const char *)sqlite3_value_text(argv[0]);
  if (!text) {
    sqlite3_result_null(ctx);
    return;
  }

  float *vec = NULL;
  int dims = 0;
  char *err = NULL;

  if (vec_parse(text, &vec, &dims, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err, -1);
    sqlite3_free(err);
    return;
  }

  double sum = 0.0;
  for (int i = 0; i < dims; i++)
    sum += (double)vec[i] * (double)vec[i];
  sqlite3_free(vec);
  sqlite3_result_double(ctx, sqrt(sum));
}

/* ── Registration ────────────────────────────────────────────────────────────
 */

int vec_register_functions(sqlite3 *db) {
  static const struct {
    const char *name;
    void (*fn)(sqlite3_context *, int, sqlite3_value **);
  } funcs[] = {
      {"vec", sqlVecFunc},
      {"vec_dims", sqlVecDimsFunc},
      {"vec_norm", sqlVecNormFunc},
  };

  int flags = SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS;
  for (int i = 0; i < (int)(sizeof(funcs) / sizeof(funcs[0])); i++) {
    int rc = sqlite3_create_function_v2(db, funcs[i].name, 1, flags, NULL,
                                        funcs[i].fn, NULL, NULL, NULL);
    if (rc != SQLITE_OK)
      return rc;
  }
  return SQLITE_OK;
}
