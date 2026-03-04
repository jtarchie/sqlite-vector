#include <sqlite3ext.h>
#include <stddef.h>

SQLITE_EXTENSION_INIT1

#include "distance.h"
#include "vec_parse.h"
#include "vtab.h"

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_sqlitevector_init(sqlite3 *db, char **pzErrMsg,
                              const sqlite3_api_routines *pApi) {
  int rc = SQLITE_OK;
  SQLITE_EXTENSION_INIT2(pApi);
  (void)pzErrMsg;

  rc = sqlite3_create_module_v2(db, VEC0_MODULE_NAME, &vec0Module, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = vec_register_functions(db);
  if (rc != SQLITE_OK)
    return rc;

  rc = distance_register_functions(db);
  return rc;
}
