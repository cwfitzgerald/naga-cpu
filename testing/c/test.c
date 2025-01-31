// https://godbolt.org/z/K3jejh6s3
#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <string.h>

#include <immintrin.h>

// Force inlining helper
#if defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#endif

#define WORKGROUP_SIZE (64)
#define SUBGROUP_SIZE (64)
#define SUBGROUPS_PER_WORKGROUP (WORKGROUP_SIZE / SUBGROUP_SIZE)

typedef struct RegisterF32 {
    alignas(32) float data[SUBGROUP_SIZE];
} RegisterF32;


FORCE_INLINE void mul_register_f32(RegisterF32* restrict result, RegisterF32 left, RegisterF32 right) {
    for (size_t i = 0; i < (SUBGROUP_SIZE / 8); i++) {
        __m256 left_reg = _mm256_load_ps(&left.data[i * 8]);
        __m256 right_reg = _mm256_load_ps(&right.data[i * 8]);
        __m256 result_reg = _mm256_mul_ps(left_reg, right_reg);
        _mm256_store_ps(&result->data[i * 8], result_reg);
    }
}

FORCE_INLINE void power(RegisterF32* restrict result, RegisterF32 input, RegisterF32 multiplier, uint32_t count) {
    for (size_t i = 0; i < count; i++) {
        mul_register_f32(&input, input, multiplier);
    }
    *result = input;
}

void power_for_workgroup(float* restrict result, float* restrict input, float* restrict multiplier, uint32_t count) {
    // For each subgroup, calculate the power, then write the result to the output buffer
    for (size_t i = 0; i < SUBGROUPS_PER_WORKGROUP; i++) {
        RegisterF32 input_reg;
        RegisterF32 multiplier_reg;
        RegisterF32 result_reg;

        // Load the input and multiplier into registers
        memcpy(input_reg.data, &input[i * SUBGROUP_SIZE], sizeof(float) * SUBGROUP_SIZE);
        memcpy(multiplier_reg.data, &multiplier[i * SUBGROUP_SIZE], sizeof(float) * SUBGROUP_SIZE);

        // Calculate the power
        power(&result_reg, input_reg, multiplier_reg, count);

        // Write the result to the output buffer
        memcpy(&result[i * SUBGROUP_SIZE], result_reg.data, sizeof(float) * SUBGROUP_SIZE);
    }
}
