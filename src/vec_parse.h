#ifndef SQLITE_VECTOR_VEC_PARSE_H
#define SQLITE_VECTOR_VEC_PARSE_H

#include <sqlite3ext.h>

SQLITE_EXTENSION_INIT3

/*
 * Parse pgvector text '[x,y,z]' into a malloc'd float array.
 *
 * On success: returns SQLITE_OK, sets *out_vec (sqlite3_malloc'd, caller frees)
 *             and *out_dims.
 * On error:   returns SQLITE_ERROR, sets *pzErr (sqlite3_mprintf'd, caller
 *             frees with sqlite3_free).  *out_vec and *out_dims are unset.
 * On OOM:     returns SQLITE_NOMEM.
 */
int vec_parse(const char *text, float **out_vec, int *out_dims, char **pzErr);

/*
 * Format a float array into pgvector text '[x,y,z]'.
 * Returns a sqlite3_malloc'd string; caller must sqlite3_free().
 * Returns NULL on OOM.
 */
char *vec_format(const float *vec, int dims);

/*
 * Register vec(), vec_dims(), vec_norm() scalar SQL functions on db.
 */
int vec_register_functions(sqlite3 *db);

#endif /* SQLITE_VECTOR_VEC_PARSE_H */
