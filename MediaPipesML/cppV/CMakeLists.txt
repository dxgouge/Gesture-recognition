cmake_minimum_required(VERSION 3.16)
project(rps_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

# =============================================================================
# OPENCV
# =============================================================================
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# =============================================================================
# LIGHTGBM
# =============================================================================
find_package(LightGBM QUIET)
if(NOT LightGBM_FOUND)
    if(EXISTS "/opt/homebrew/opt/lightgbm/include/LightGBM")
        set(LIGHTGBM_INCLUDE_DIR "/opt/homebrew/opt/lightgbm/include")
        set(LIGHTGBM_LIB_DIR     "/opt/homebrew/opt/lightgbm/lib")
    elseif(EXISTS "/usr/local/opt/lightgbm/include/LightGBM")
        set(LIGHTGBM_INCLUDE_DIR "/usr/local/opt/lightgbm/include")
        set(LIGHTGBM_LIB_DIR     "/usr/local/opt/lightgbm/lib")
    else()
        message(FATAL_ERROR "Could not find LightGBM. Run: brew install lightgbm")
    endif()
endif()

# =============================================================================
# TFLITE PATHS
# =============================================================================
set(TFLITE_ROOT  "$ENV{HOME}/tensorflow")
set(TFLITE_BUILD "${TFLITE_ROOT}/tflite_build_gpu")
set(DEPS         "${TFLITE_BUILD}/_deps")

set(TFLITE_INCLUDE_DIRS
    ${TFLITE_ROOT}
    ${TFLITE_BUILD}
    ${TFLITE_BUILD}/flatbuffers/include
    ${DEPS}/abseil-cpp-src
    ${DEPS}/flatbuffers-src/include
)

# =============================================================================
# TFLITE LIBRARIES (explicit, no globs)
# =============================================================================

# Core
set(TFLITE_LIB ${TFLITE_BUILD}/libtensorflow-lite.a)

# XNNPACK
set(XNNPACK_LIB ${DEPS}/xnnpack-build/libXNNPACK.a)

# Ruy
set(RUY_LIBS
    ${DEPS}/ruy-build/ruy/libruy_frontend.a
    ${DEPS}/ruy-build/ruy/libruy_context.a
    ${DEPS}/ruy-build/ruy/libruy_context_get_ctx.a
    ${DEPS}/ruy-build/ruy/libruy_ctx.a
    ${DEPS}/ruy-build/ruy/libruy_trmul.a
    ${DEPS}/ruy-build/ruy/libruy_block_map.a
    ${DEPS}/ruy-build/ruy/libruy_kernel_arm.a
    ${DEPS}/ruy-build/ruy/libruy_pack_arm.a
    ${DEPS}/ruy-build/ruy/libruy_apply_multiplier.a
    ${DEPS}/ruy-build/ruy/libruy_allocator.a
    ${DEPS}/ruy-build/ruy/libruy_system_aligned_alloc.a
    ${DEPS}/ruy-build/ruy/libruy_prepacked_cache.a
    ${DEPS}/ruy-build/ruy/libruy_prepare_packed_matrices.a
    ${DEPS}/ruy-build/ruy/libruy_tune.a
    ${DEPS}/ruy-build/ruy/libruy_cpuinfo.a
    ${DEPS}/ruy-build/ruy/libruy_denormal.a
    ${DEPS}/ruy-build/ruy/libruy_thread_pool.a
    ${DEPS}/ruy-build/ruy/libruy_blocking_counter.a
    ${DEPS}/ruy-build/ruy/libruy_wait.a
    ${DEPS}/ruy-build/ruy/libruy_have_built_path_for_avx.a
    ${DEPS}/ruy-build/ruy/libruy_have_built_path_for_avx2_fma.a
    ${DEPS}/ruy-build/ruy/libruy_have_built_path_for_avx512.a
    ${DEPS}/ruy-build/ruy/libruy_kernel_avx.a
    ${DEPS}/ruy-build/ruy/libruy_kernel_avx2_fma.a
    ${DEPS}/ruy-build/ruy/libruy_kernel_avx512.a
    ${DEPS}/ruy-build/ruy/libruy_pack_avx.a
    ${DEPS}/ruy-build/ruy/libruy_pack_avx2_fma.a
    ${DEPS}/ruy-build/ruy/libruy_pack_avx512.a
    ${DEPS}/ruy-build/ruy/profiler/libruy_profiler_instrumentation.a
    ${DEPS}/ruy-build/ruy/profiler/libruy_profiler_profiler.a
)

# Abseil
set(ABSL_LIBS
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_base.a
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_log_severity.a
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_malloc_internal.a
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_raw_logging_internal.a
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_scoped_set_env.a
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_spinlock_wait.a
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_strerror.a
    ${DEPS}/abseil-cpp-build/absl/base/libabsl_throw_delegate.a
    ${DEPS}/abseil-cpp-build/absl/container/libabsl_hashtablez_sampler.a
    ${DEPS}/abseil-cpp-build/absl/container/libabsl_raw_hash_set.a
    ${DEPS}/abseil-cpp-build/absl/crc/libabsl_crc_cord_state.a
    ${DEPS}/abseil-cpp-build/absl/crc/libabsl_crc_cpu_detect.a
    ${DEPS}/abseil-cpp-build/absl/crc/libabsl_crc_internal.a
    ${DEPS}/abseil-cpp-build/absl/crc/libabsl_crc32c.a
    ${DEPS}/abseil-cpp-build/absl/debugging/libabsl_debugging_internal.a
    ${DEPS}/abseil-cpp-build/absl/debugging/libabsl_demangle_internal.a
    ${DEPS}/abseil-cpp-build/absl/debugging/libabsl_examine_stack.a
    ${DEPS}/abseil-cpp-build/absl/debugging/libabsl_failure_signal_handler.a
    ${DEPS}/abseil-cpp-build/absl/debugging/libabsl_leak_check.a
    ${DEPS}/abseil-cpp-build/absl/debugging/libabsl_stacktrace.a
    ${DEPS}/abseil-cpp-build/absl/debugging/libabsl_symbolize.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_commandlineflag.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_commandlineflag_internal.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_config.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_internal.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_marshalling.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_parse.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_private_handle_accessor.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_program_name.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_reflection.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_usage.a
    ${DEPS}/abseil-cpp-build/absl/flags/libabsl_flags_usage_internal.a
    ${DEPS}/abseil-cpp-build/absl/hash/libabsl_city.a
    ${DEPS}/abseil-cpp-build/absl/hash/libabsl_hash.a
    ${DEPS}/abseil-cpp-build/absl/hash/libabsl_low_level_hash.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_die_if_null.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_entry.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_flags.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_globals.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_initialize.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_check_op.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_conditions.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_format.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_globals.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_log_sink_set.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_message.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_nullguard.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_internal_proto.a
    ${DEPS}/abseil-cpp-build/absl/log/libabsl_log_sink.a
    ${DEPS}/abseil-cpp-build/absl/numeric/libabsl_int128.a
    ${DEPS}/abseil-cpp-build/absl/profiling/libabsl_exponential_biased.a
    ${DEPS}/abseil-cpp-build/absl/profiling/libabsl_periodic_sampler.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_distributions.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_distribution_test_util.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_platform.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_pool_urbg.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_randen.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_randen_hwaes.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_randen_hwaes_impl.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_randen_slow.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_internal_seed_material.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_seed_gen_exception.a
    ${DEPS}/abseil-cpp-build/absl/random/libabsl_random_seed_sequences.a
    ${DEPS}/abseil-cpp-build/absl/status/libabsl_status.a
    ${DEPS}/abseil-cpp-build/absl/status/libabsl_statusor.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_cord.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_cord_internal.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_cordz_functions.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_cordz_handle.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_cordz_info.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_cordz_sample_token.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_str_format_internal.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_string_view.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_strings.a
    ${DEPS}/abseil-cpp-build/absl/strings/libabsl_strings_internal.a
    ${DEPS}/abseil-cpp-build/absl/synchronization/libabsl_graphcycles_internal.a
    ${DEPS}/abseil-cpp-build/absl/synchronization/libabsl_kernel_timeout_internal.a
    ${DEPS}/abseil-cpp-build/absl/synchronization/libabsl_synchronization.a
    ${DEPS}/abseil-cpp-build/absl/time/libabsl_civil_time.a
    ${DEPS}/abseil-cpp-build/absl/time/libabsl_time.a
    ${DEPS}/abseil-cpp-build/absl/time/libabsl_time_zone.a
    ${DEPS}/abseil-cpp-build/absl/types/libabsl_bad_any_cast_impl.a
    ${DEPS}/abseil-cpp-build/absl/types/libabsl_bad_optional_access.a
    ${DEPS}/abseil-cpp-build/absl/types/libabsl_bad_variant_access.a
)

# FFT2D
set(FFT2D_LIBS
    ${DEPS}/fft2d-build/libfft2d_fftsg.a
    ${DEPS}/fft2d-build/libfft2d_fftsg2d.a
    ${DEPS}/fft2d-build/libfft2d_fftsg3d.a
    ${DEPS}/fft2d-build/libfft2d_fft4f2d.a
    ${DEPS}/fft2d-build/libfft2d_alloc.a
    ${DEPS}/fft2d-build/libfft2d_shrtdct.a
)

# Misc
set(FARMHASH_LIB  ${DEPS}/farmhash-build/libfarmhash.a)
set(FLATBUF_LIB   ${DEPS}/flatbuffers-build/libflatbuffers.a)
set(PTHREADPOOL_LIB ${TFLITE_BUILD}/pthreadpool/libpthreadpool.a)
set(CPUINFO_LIBS
    ${DEPS}/cpuinfo-build/libcpuinfo.a
    ${DEPS}/cpuinfo-build/libcpuinfo_internals.a
)

find_library(ACCELERATE_FRAMEWORK Accelerate)
find_library(FOUNDATION_FRAMEWORK  Foundation)
find_library(METAL_FRAMEWORK Metal)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics)

# =============================================================================
# EXECUTABLE
# =============================================================================
add_executable(rps_cpp
    src/main.cpp
    src/features.cpp
    src/inference.cpp
    src/capture.cpp
    src/palm_detector.cpp
    src/hand_landmarker.cpp
)

target_include_directories(rps_cpp PRIVATE
    src
    ${OpenCV_INCLUDE_DIRS}
    ${LIGHTGBM_INCLUDE_DIR}
    ${TFLITE_INCLUDE_DIRS}
    ${TFLITE_ROOT}/tensorflow/lite/delegates/gpu
)

target_link_directories(rps_cpp PRIVATE
    ${LIGHTGBM_LIB_DIR}
    
    
)

target_link_libraries(rps_cpp PRIVATE
    ${OpenCV_LIBS}
    _lightgbm
    ${TFLITE_LIB}
    ${XNNPACK_LIB}
    ${RUY_LIBS}
    ${ABSL_LIBS}
    ${FFT2D_LIBS}
    ${FARMHASH_LIB}
    ${FLATBUF_LIB}
    ${PTHREADPOOL_LIB}
    ${CPUINFO_LIBS}
    ${ACCELERATE_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${TFLITE_BUILD}/libLiteRtMetalAccelerator.dylib
    ${METAL_FRAMEWORK}
    ${COREGRAPHICS_FRAMEWORK}
)
