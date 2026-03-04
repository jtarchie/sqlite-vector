#ifndef SQLITE_VECTOR_VTAB_H
#define SQLITE_VECTOR_VTAB_H

#include <sqlite3ext.h>

SQLITE_EXTENSION_INIT3

/* The name of the virtual table module registered with SQLite. */
#define VEC0_MODULE_NAME "vec0"

/* sqlite3_module for the vec0 virtual table. Defined in vtab.c. */
extern sqlite3_module vec0Module;

#endif /* SQLITE_VECTOR_VTAB_H */
