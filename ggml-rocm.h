#include <hipblas.h>
#include <rocblas.h>
#include <hip/hip_runtime.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define HIP_CHECK(err)                                                                 \
    do {                                                                                \
        hipError_t err_ = (err);                                                        \
        if (err_ != hipSuccess) {                                                       \
            fprintf(stderr, "HIP error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                hipGetErrorString(err_));                                               \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#define HIPBLAS_CHECK(err)                                                              \
    do {                                                                                \
        hipblasStatus_t err_ = (err);                                                   \
        if (err_ != HIPBLAS_STATUS_SUCCESS) {                                           \
            fprintf(stderr, "hipBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);   \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

extern hipblasHandle_t g_hipblasH;
extern hipStream_t   g_hipStream;

void   ggml_init_hipblas(void);
void * ggml_hip_pool_malloc(size_t size, size_t * actual_size);
void   ggml_hip_pool_free(void * ptr, size_t size);

void dequantize_row_q4_0_hip(const void * vx, float * y, int k, hipStream_t stream);
void dequantize_row_q4_1_hip(const void * vx, float * y, int k, hipStream_t stream);
void dequantize_row_q4_2_hip(const void * vx, float * y, int k, hipStream_t stream);
void dequantize_row_q4_3_hip(const void * vx, float * y, int k, hipStream_t stream);
void dequantize_row_q8_0_hip(const void * vx, float * y, int k, hipStream_t stream);

#ifdef  __cplusplus
}
#endif
