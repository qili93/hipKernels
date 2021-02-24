include(ExternalProject)

set(CUB_PREFIX_DIR ${CMAKE_BINARY_DIR}/cub)
set(CUB_SOURCE_DIR ${CMAKE_BINARY_DIR}/cub/src/extern_cub)
set(CUB_REPOSITORY https://github.com/NVlabs/cub.git)
set(CUB_TAG        1.8.0)

SET(CUB_INCLUDE_DIR   ${CUB_SOURCE_DIR})
include_directories(${CUB_INCLUDE_DIR})

# external dependencies log output
SET(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD    0     # Wrap download in script to log output
    LOG_UPDATE      1     # Wrap update in script to log output
    LOG_CONFIGURE   1     # Wrap configure in script to log output
    LOG_BUILD       0     # Wrap build in script to log output
    LOG_TEST        1     # Wrap test in script to log output
    LOG_INSTALL     0     # Wrap install in script to log output
)

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_SHALLOW     TRUE
  GIT_REPOSITORY  https://github.com/NVlabs/cub.git
  GIT_TAG         1.8.0
  PREFIX          ${CUB_PREFIX_DIR}
  SOURCE_DIR      ${CUB_SOURCE_DIR}
  UPDATE_COMMAND    ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

add_library(cub INTERFACE)
add_dependencies(cub extern_cub)
