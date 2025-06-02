cmake_minimum_required(VERSION 3.18)
project(ParticleSim LANGUAGES C CXX CUDA)  # Añade C como lenguaje para GLAD

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)  # Necesario para GLAD

# Configuración CUDA
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "61")

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

# Separa GLAD como biblioteca primero
add_library(glad STATIC external/glad.c)
target_include_directories(glad PRIVATE include)

# Configura propiedades para GLAD (importante para CUDA)
set_target_properties(glad PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(ParticleSim src/main.cu)
target_include_directories(ParticleSim PRIVATE
    include
    ${GLFW_INCLUDE_DIRS}
)

# Vinculación en el orden CORRECTO
target_link_libraries(ParticleSim PRIVATE
    glad       # GLAD primero
    ${GLFW_LIBRARIES}
    OpenGL::GL # OpenGL oficial (moderno)
    ${CUDA_cudart_static_LIBRARY}
    ${CUDA_cuda_gl_interop_LIBRARY}
)

# Para versiones antiguas de CMake sin OpenGL::GL target
if(NOT TARGET OpenGL::GL)
    find_package(OpenGL REQUIRED)
    target_link_libraries(ParticleSim PRIVATE ${OPENGL_LIBRARIES})
endif()


