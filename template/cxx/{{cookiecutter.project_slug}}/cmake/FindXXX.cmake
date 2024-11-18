# FindXXX.cmake - Locate XXX library and headers
#
# This module defines the following variables:
#   XXX_FOUND        - True if the XXX library and headers are found
#   XXX_INCLUDE_DIRS - The include directories for XXX
#   XXX_LIBRARIES    - The libraries to link against for XXX
#   XXX_VERSION      - The version string of XXX (if available)

# Define search paths
set(XXX_SEARCH_PATHS
        ${CMAKE_PREFIX_PATH}
        /usr/local/xxx
        /usr/local
        /usr
        /opt/xxx
        /opt
)

# Locate the header files
find_path(XXX_INCLUDE_DIR NAMES xxx.h
        HINTS
        ${XXX_ROOT}
        $ENV{XXX_ROOT}
        PATHS
        ${XXX_SEARCH_PATHS}
        PATH_SUFFIXES include
)

# Define library components
set(XXX_LIB_COMPONENTS component1 component2 component3)

# Locate the libraries
set(XXX_LIBRARIES)
foreach(_comp ${XXX_LIB_COMPONENTS})
    find_library(XXX_${_comp}_LIBRARY
            NAMES ${_comp}
            HINTS
            ${XXX_ROOT}
            $ENV{XXX_ROOT}
            PATHS
            ${XXX_SEARCH_PATHS}
            PATH_SUFFIXES lib lib64
    )
    if(XXX_${_comp}_LIBRARY)
        list(APPEND XXX_LIBRARIES ${XXX_${_comp}_LIBRARY})
    endif()
endforeach()

# Set include directories
set(XXX_INCLUDE_DIRS ${XXX_INCLUDE_DIR})

# Version check (adapted it with your own library structure)
if(XXX_INCLUDE_DIR AND EXISTS "${XXX_INCLUDE_DIR}/../release-xxx.txt")
    get_filename_component(XXX_PARENT_DIR ${XXX_INCLUDE_DIR} DIRECTORY)
    set(RELEASE_FILE "${XXX_PARENT_DIR}/release-xxx.txt")
    if(EXISTS "${RELEASE_FILE}")
        file(READ "${RELEASE_FILE}" XXX_VERSION_CONTENTS LIMIT_COUNT 1)
        string(REGEX MATCH "XXX SDK ([0-9]+\\.[0-9]+\\.[0-9]+)" XXX_VERSION_MATCH "${XXX_VERSION_CONTENTS}")
        if(XXX_VERSION_MATCH)
            string(REGEX REPLACE "XXX SDK ([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" XXX_VERSION "${XXX_VERSION_MATCH}")
        else()
            message(WARNING "Failed to detect XXX version from ${RELEASE_FILE}")
        endif()
    endif()
endif()

# Handle the QUIETLY and REQUIRED arguments, set XXX_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XXX
        REQUIRED_VARS XXX_LIBRARIES XXX_INCLUDE_DIRS
        VERSION_VAR XXX_VERSION
)

# Mark variables as advanced
mark_as_advanced(XXX_INCLUDE_DIR)
foreach(_comp ${XXX_LIB_COMPONENTS})
    mark_as_advanced(XXX_${_comp}_LIBRARY)
endforeach()
