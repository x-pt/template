-- Common configuration for third-party libraries

-- Target: httplib
target("httplib")
    set_kind("headeronly")                          -- Define it as a header-only library.
    add_includedirs("httplib")                      -- Fallback to direct folder.
    add_headerfiles("httplib/**/*.h")               -- Ensure all headers are tracked for IDEs.
    add_headerfiles("httplib/**/*.hpp")

-- Target: cxxopts
target("cxxopts")
    set_kind("headeronly")                          -- Define it as a header-only library.
    add_includedirs("cxxopts")                      -- Default include path.
    add_headerfiles("cxxopts/**/*.h")               -- Track all headers.
    add_headerfiles("cxxopts/**/*.hpp")
