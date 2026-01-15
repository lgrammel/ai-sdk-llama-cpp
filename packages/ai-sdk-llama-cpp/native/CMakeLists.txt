cmake_minimum_required(VERSION 3.15)
project(llama_binding)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build llama.cpp as a static library
set(LLAMA_STATIC ON CACHE BOOL "Build llama.cpp as static library" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama.cpp server" FORCE)

# Enable Metal on macOS
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "Enable Metal support" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal library" FORCE)
endif()

# Add llama.cpp subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

# Find Node.js and node-addon-api
include_directories(${CMAKE_JS_INC})

# Find node-addon-api
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
include_directories(${NODE_ADDON_API_DIR})

# Include llama.cpp headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/include)

# Build the native addon
add_library(${PROJECT_NAME} SHARED
    binding.cpp
    llama-wrapper.cpp
)

# Set output name and extension
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    OUTPUT_NAME "llama_binding"
)

# Link against llama.cpp
target_link_libraries(${PROJECT_NAME} PRIVATE
    llama
    ggml
    ${CMAKE_JS_LIB}
)

# Define NAPI_VERSION
target_compile_definitions(${PROJECT_NAME} PRIVATE
    NAPI_VERSION=8
    NAPI_CPP_EXCEPTIONS
)

# Platform-specific settings
if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
endif()

