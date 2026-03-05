#ifndef SQLITE_VECTOR_VEC_OPS_H
#define SQLITE_VECTOR_VEC_OPS_H

#include <sqlite3ext.h>

/*
 * vec_ops_register_functions
 * Registers the following SQL scalar functions on db:
 *   vec_add(vec_text, vec_text)       → TEXT  (element-wise sum)
 *   vec_sub(vec_text, vec_text)       → TEXT  (element-wise difference)
 *   vec_normalize(vec_text)           → TEXT  (unit vector via L2 norm)
 *   vec_slice(vec_text, start, end)   → TEXT  (subvector extraction)
 *   vec_f32(text|blob)                → BLOB  (float32 with subtype 223)
 *   vec_int8(text|blob)               → BLOB  (int8 with subtype 225)
 *   vec_bit(blob)                     → BLOB  (bitvector with subtype 224)
 *   vec_type(v)                       → TEXT  (type name: float32/int8/bit/text)
 * Returns SQLITE_OK on success.
 */
int vec_ops_register_functions(sqlite3 *db);

#endif /* SQLITE_VECTOR_VEC_OPS_H */
