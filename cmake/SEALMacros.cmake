# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Set the C++ language version
macro(seal_set_language target)
    if(SEAL_USE_CXX17)
        target_compile_features(${target} PUBLIC cxx_std_17)
    else()
        target_compile_features(${target} PUBLIC cxx_std_14)
    endif()
endmacro()

# Set the VERSION property
macro(seal_set_version target)
    set_target_properties(${target} PROPERTIES VERSION ${SEAL_VERSION})
endmacro()

# Set the library filename to reflect version
macro(seal_set_version_filename target)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${target}-${SEAL_VERSION_MAJOR}.${SEAL_VERSION_MINOR})
endmacro()

# Set the SOVERSION property
macro(seal_set_soversion target)
    set_target_properties(${target} PROPERTIES
        SOVERSION ${SEAL_VERSION_MAJOR}.${SEAL_VERSION_MINOR})
endmacro()

# Set include directories for build and install interfaces
macro(seal_set_include_directories target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${SEAL_INCLUDES_BUILD_DIR}>
        $<INSTALL_INTERFACE:${SEAL_INCLUDES_INSTALL_DIR}>)
endmacro()

# Link a thread library
macro(seal_link_threads target)
    # Require thread library
    if(NOT TARGET Threads::Threads)
        set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
        set(THREADS_PREFER_PTHREAD_FLAG TRUE)
        find_package(Threads REQUIRED)
    endif()

    # Link Threads
    target_link_libraries(${target} PUBLIC Threads::Threads)
endmacro()

# Include target to given export
macro(seal_install_target target export)
    install(TARGETS ${target} EXPORT ${export}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endmacro()

# Manually combine archives, using ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} to keep temporary files.
macro(seal_combine_archives target dependency)
    if(MSVC)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND lib.exe /OUT:$<TARGET_FILE:${target}> $<TARGET_FILE:${target}> $<TARGET_FILE:${dependency}>
            DEPENDS $<TARGET_FILE:${target}> $<TARGET_FILE:${dependency}>
            WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    else()
        if(CMAKE_HOST_WIN32)
            get_filename_component(CXX_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
            set(AR_CMD_PATH "${CXX_DIR}/llvm-ar.exe")
            file(TO_NATIVE_PATH "${AR_CMD_PATH}" AR_CMD_PATH)
            set(DEL_CMD "del")
            set(DEL_CMD_OPTS "")
        else()
            set(AR_CMD_PATH "ar")
            set(DEL_CMD "rm")
            set(DEL_CMD_OPTS "-rf")
        endif()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${AR_CMD_PATH}" x $<TARGET_FILE:${target}>
            COMMAND "${AR_CMD_PATH}" x $<TARGET_FILE:${dependency}>
            COMMAND "${AR_CMD_PATH}" rcs $<TARGET_FILE:${target}> *.o
            COMMAND ${DEL_CMD} ${DEL_CMD_OPTS} *.o
            WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    endif()
endmacro()