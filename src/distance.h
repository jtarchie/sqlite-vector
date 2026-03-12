#ifndef SQLITE_VECTOR_DISTANCE_H
#define SQLITE_VECTOR_DISTANCE_H

#include <sqlite3ext.h>

/*
 * dist_fn_t — common signature for all distance kernels.
 *
 * a, b : input vectors of length dims (type depends on storage format)
 * dims : number of elements (for binary: number of bits)
 * out  : receives the scalar distance (written as double)
 */
typedef void (*dist_fn_t)(const void *a, const void *b, int dims, double *out);

/* distance_for_metric
 * Returns the dist_fn_t matching the named metric string, or NULL if the
 * name is not recognised.  Returns the float32 kernel.
 * Accepted names:
 *   "l2", "l2sq", "euclidean",
 *   "cosine", "cos",
 *   "ip", "dot", "inner_product",
 *   "l1", "taxicab", "manhattan",
 *   "hamming",
 *   "jaccard"
 */
dist_fn_t distance_for_metric(const char *metric);

/* distance_for_metric_typed
 * Returns the dist_fn_t for the given metric + storage type combination.
 * storage_type: VEC_STORAGE_F32, VEC_STORAGE_INT8, or VEC_STORAGE_BIT
 *   (enum values from vtab.h).
 * Returns NULL for unsupported metric+type combinations.
 */
dist_fn_t distance_for_metric_typed(const char *metric, int storage_type);

/* distance_register_functions
 * Registers the following SQL scalar functions on db:
 *   vec_distance_l2(vec_text, vec_text)       → REAL  (sqrt of L2sq)
 *   vec_distance_cosine(vec_text, vec_text)   → REAL  (1 − cos similarity)
 *   vec_distance_ip(vec_text, vec_text)       → REAL  (negated dot product)
 *   vec_distance_l1(vec_text, vec_text)       → REAL
 *   vec_distance_hamming(vec_text, vec_text)  → REAL
 *   vec_distance_jaccard(vec_text, vec_text)  → REAL
 * Returns SQLITE_OK on success.
 */
int distance_register_functions(sqlite3 *db);

#endif /* SQLITE_VECTOR_DISTANCE_H */
