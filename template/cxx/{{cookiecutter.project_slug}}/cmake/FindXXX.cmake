# FindXXX.cmake - Locate XXX library and headers
#
# This module defines the following variables:
#   XXX_FOUND          - True if the XXX library and headers are found
#   XXX_INCLUDE_DIRS   - The include directories for XXX
#   XXX_LIBRARIES      - The libraries to link against for XXX
#   XXX_VERSION        - The version string of XXX (if available)

# Early return if target is already defined
if(TARGET XXX::XXX)
    return()
endif()

# Define search paths
set(XXX_SEARCH_PATHS
        ${CMAKE_PREFIX_PATH}
        /usr/local/xxx
        /usr/local
        /usr
        /opt/xxx
        /opt
        # Add other common installation paths here
)

# Locate the header files
find_path(XXX_INCLUDE_DIR
        NAMES xxx.h
        HINTS ${XXX_ROOT} $ENV{XXX_ROOT}
        PATHS ${XXX_SEARCH_PATHS}
        PATH_SUFFIXES include
        DOC "XXX include directory"
)

# Define library components
set(XXX_LIB_COMPONENTS xxx)  # Add more components if needed

# Locate the libraries
set(XXX_LIBRARIES)
foreach(_comp ${XXX_LIB_COMPONENTS})
    find_library(XXX_${_comp}_LIBRARY
            NAMES ${_comp}
            HINTS ${XXX_ROOT} $ENV{XXX_ROOT}
            PATHS ${XXX_SEARCH_PATHS}
            PATH_SUFFIXES lib lib64
            DOC "XXX ${_comp} library"
    )
    if(XXX_${_comp}_LIBRARY)
        list(APPEND XXX_LIBRARIES ${XXX_${_comp}_LIBRARY})
    endif()
endforeach()

# Set include directories
set(XXX_INCLUDE_DIRS ${XXX_INCLUDE_DIR})

# Version detection (customize based on your library's version format)
if(XXX_INCLUDE_DIR AND EXISTS "${XXX_INCLUDE_DIR}/xxx_version.h")
    file(STRINGS "${XXX_INCLUDE_DIR}/xxx_version.h" XXX_VERSION_MAJOR_LINE REGEX "^#define[ \t]+XXX_VERSION_MAJOR[ \t]+[0-9]+")
    file(STRINGS "${XXX_INCLUDE_DIR}/xxx_version.h" XXX_VERSION_MINOR_LINE REGEX "^#define[ \t]+XXX_VERSION_MINOR[ \t]+[0-9]+")
    file(STRINGS "${XXX_INCLUDE_DIR}/xxx_version.h" XXX_VERSION_PATCH_LINE REGEX "^#define[ \t]+XXX_VERSION_PATCH[ \t]+[0-9]+")

    string(REGEX REPLACE "^#define[ \t]+XXX_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" XXX_VERSION_MAJOR "${XXX_VERSION_MAJOR_LINE}")
    string(REGEX REPLACE "^#define[ \t]+XXX_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" XXX_VERSION_MINOR "${XXX_VERSION_MINOR_LINE}")
    string(REGEX REPLACE "^#define[ \t]+XXX_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" XXX_VERSION_PATCH "${XXX_VERSION_PATCH_LINE}")

    set(XXX_VERSION "${XXX_VERSION_MAJOR}.${XXX_VERSION_MINOR}.${XXX_VERSION_PATCH}")
endif()

# Handle the QUIETLY and REQUIRED arguments, set XXX_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XXX
        REQUIRED_VARS XXX_LIBRARIES XXX_INCLUDE_DIRS
        VERSION_VAR XXX_VERSION
)

# Create imported target
if(XXX_FOUND AND NOT TARGET XXX::XXX)
    add_library(XXX::XXX UNKNOWN IMPORTED)
    set_target_properties(XXX::XXX PROPERTIES
            IMPORTED_LOCATION "${XXX_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${XXX_INCLUDE_DIRS}"
    )

    # Optional: Add more properties if needed
    # set_property(TARGET XXX::XXX PROPERTY
    #     INTERFACE_COMPILE_DEFINITIONS XXX_SOME_DEFINITION
    # )
endif()

# Mark variables as advanced
mark_as_advanced(XXX_INCLUDE_DIR)
foreach(_comp ${XXX_LIB_COMPONENTS})
    mark_as_advanced(XXX_${_comp}_LIBRARY)
endforeach()
