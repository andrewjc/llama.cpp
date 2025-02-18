#include <stdint.h>
#include <stdio.h>
#include <hip/hip_fp16.h>
#include <atomic>
#include "ggml-rocm.h"

#define hipHalf __fp16

typedef uint16_t ggml_fp16_t;
static_assert(sizeof(hipHalf) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define QK4_0 32
typedef struct {
    float   d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    float   d;              // delta
    float   m;              // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(float) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK4_2 16
typedef struct {
    hipHalf  d;              // delta
    uint8_t qs[QK4_2 / 2];   // nibbles / quants
} block_q4_2;
static_assert(sizeof(block_q4_2) == sizeof(ggml_fp16_t) + QK4_2 / 2, "wrong q4_2 block size/padding");

#define QK4_3 16
typedef struct {
    hipHalf  d;              // delta
    hipHalf  m;              // min
    uint8_t qs[QK4_3 / 2];   // nibbles / quants
} block_q4_3;
static_assert(sizeof(block_q4_3) == 2 * sizeof(ggml_fp16_t) + QK4_3 / 2, "wrong q4_3 block size/padding");

#define QK8_0 32
typedef struct {
    float   d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(float) + QK8_0, "wrong q8_0 block size/padding");

static __global__ void dequantize_block_q4_0(const void * vx, float * y) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int i = hipBlockIdx_x;

    const float d = x[i].d;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_0; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = (vi0 - 8)*d;
        const float v1 = (vi1 - 8)*d;

        y[i*QK4_0 + l + 0] = v0;
        y[i*QK4_0 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q4_1(const void * vx, float * y) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int i = hipBlockIdx_x;

    const float d = x[i].d;
    const float m = x[i].m;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_1; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = vi0*d + m;
        const float v1 = vi1*d + m;

        y[i*QK4_1 + l + 0] = v0;
        y[i*QK4_1 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q4_2(const void * vx, float * y) {
    const block_q4_2 * x = (const block_q4_2 *) vx;

    const int i = hipBlockIdx_x;

    const float d = x[i].d;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_2; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = (vi0 - 8)*d;
        const float v1 = (vi1 - 8)*d;

        y[i*QK4_2 + l + 0] = v0;
        y[i*QK4_2 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q4_3(const void * vx, float * y) {
    const block_q4_3 * x = (const block_q4_3 *) vx;

    const int i = hipBlockIdx_x;

    const float d = x[i].d;
    const float m = x[i].m;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_3; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = vi0*d + m;
        const float v1 = vi1*d + m;

        y[i*QK4_3 + l + 0] = v0;
        y[i*QK4_3 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q8_0(const void * vx, float * y) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int i = hipBlockIdx_x;

    const float d = x[i].d;

    const int8_t * pp = x[i].qs;

    for (int l = 0; l < QK8_0; l++) {
        const int8_t vi = pp[l];

        y[i*QK8_0 + l] = vi*d;
    }
}

void dequantize_row_q4_0_hip(const void * vx, float * y, int k, hipStream_t stream) {
    const int nb = k / QK4_0;
    hipLaunchKernelGGL(dequantize_block_q4_0, dim3(nb), dim3(1), 0, stream, vx, y);
}

void dequantize_row_q4_1_hip(const void * vx, float * y, int k, hipStream_t stream) {
    const int nb = k / QK4_1;
    hipLaunchKernelGGL(dequantize_block_q4_1, dim3(nb), dim3(1), 0, stream, vx, y);
}

void dequantize_row_q4_2_hip(const void * vx, float * y, int k, hipStream_t stream) {
    const int nb = k / QK4_2;
    hipLaunchKernelGGL(dequantize_block_q4_2, dim3(nb), dim3(1), 0, stream, vx, y);
}

void dequantize_row_q4_3_hip(const void * vx, float * y, int k, hipStream_t stream) {
    const int nb = k / QK4_3;
    hipLaunchKernelGGL(dequantize_block_q4_3, dim3(nb), dim3(1), 0, stream, vx, y);
}

void dequantize_row_q8_0_hip(const void * vx, float * y, int k, hipStream_t stream) {
    const int nb = k / QK8_0;
    hipLaunchKernelGGL(dequantize_block_q8_0, dim3(nb), dim3(1), 0, stream, vx, y);
}

// buffer pool for HIP
#define MAX_HIP_BUFFERS 16

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct hip_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static hip_buffer g_hip_buffer_pool[MAX_HIP_BUFFERS];
static std::atomic_flag g_hip_pool_lock = ATOMIC_FLAG_INIT;

void * ggml_hip_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_hip_pool_lock);

    for (int i = 0; i < MAX_HIP_BUFFERS; ++i) {
        hip_buffer& b = g_hip_buffer_pool[i];
        if (b.size >= size && b.ptr != nullptr) {
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
    }
    void * ptr;
    HIP_CHECK(hipMalloc((void **) &ptr, size));
    *actual_size = size;
    return ptr;
}

void ggml_hip_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_hip_pool_lock);

    for (int i = 0; i < MAX_HIP_BUFFERS; ++i) {
        hip_buffer& b = g_hip_buffer_pool[i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: hip buffer pool full, increase MAX_HIP_BUFFERS\n");
    HIP_CHECK(hipFree(ptr));
}

hipblasHandle_t g_hipblasH = NULL;
hipStream_t g_hipStream = NULL;

void ggml_init_hipblas(void) {
    if (g_hipblasH == NULL) {
        // create hipblas handle, bind a stream
        HIPBLAS_CHECK(hipblasCreate(&g_hipblasH));

        HIP_CHECK(hipStreamCreateWithFlags(&g_hipStream, hipStreamNonBlocking));

        HIPBLAS_CHECK(hipblasSetStream(g_hipblasH, g_hipStream));

        // configure logging to stdout
        // HIPBLAS_CHECK(hipblasLoggerConfigure(1, 1, 0, NULL));
    }
}

