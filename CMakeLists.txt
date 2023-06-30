cmake_minimum_required(VERSION 3.25)

project(k-means-in-C LANGUAGES C)

set(CMAKE_C_STANDARD 90)

add_executable(k-means-in-C  src/main.c src/help.h src/help.c src/kmeans.h src/kmeans.c)

target_link_libraries(k-means-in-C m)

install(TARGETS k-means-in-C
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})