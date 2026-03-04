#ifndef SQLITE_VECTOR_DISTANCE_H
#define SQLITE_VECTOR_DISTANCE_H

#include <sqlite3ext.h>

/*
 * dist_fn_t — common signature for all distance kernels.
 *
 * a, b : input float vectors of length dims
 * dims : number of elements
 * out  : receives the scalar distance (written as double)
 *
 * Hamming / Jaccard kernels treat the float arrays as packed bit-vectors:
 * each 4-byte float element is one 32-bit word of bits, so dims must be a
 * multiple of 8 for those metrics (SimSIMD's b8 kernels operate on bytes).
 */
typedef void (*dist_fn_t)(const float *a, const float *b, int dims,
                          double *out);

/* distance_for_metric
 * Returns the dist_fn_t matching the named metric string, or NULL if the
 * name is not recognised.  Accepted names match the PLAN.md list:
 *   "l2", "l2sq", "euclidean",
 *   "cosine", "cos",
 *   "ip", "dot", "inner_product",
 *   "l1", "taxicab", "manhattan",
 *   "hamming",
 *   "jaccard"
 */
dist_fn_t distance_for_metric(const char *metric);

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
