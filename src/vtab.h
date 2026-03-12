#ifndef SQLITE_VECTOR_VTAB_H
#define SQLITE_VECTOR_VTAB_H

#include <sqlite3ext.h>

SQLITE_EXTENSION_INIT3

/* The name of the virtual table module registered with SQLite. */
#define VEC0_MODULE_NAME "vec0"

/* Storage types for vectors in the vec0 virtual table. */
enum {
  VEC_STORAGE_F32 = 0,  /* 4 bytes per dimension (default) */
  VEC_STORAGE_INT8 = 1, /* 1 byte per dimension */
  VEC_STORAGE_BIT = 2   /* 1 bit per dimension, packed */
};

/* Return the number of bytes needed to store a vector of the given type. */
static inline int vec_blob_bytes(int storage_type, int dims) {
  switch (storage_type) {
  case VEC_STORAGE_INT8:
    return dims;
  case VEC_STORAGE_BIT:
    return (dims + 7) / 8;
  default: /* VEC_STORAGE_F32 */
    return dims * (int)sizeof(float);
  }
}

/* sqlite3_module for the vec0 virtual table. Defined in vtab.c. */
extern sqlite3_module vec0Module;

#endif /* SQLITE_VECTOR_VTAB_H */
