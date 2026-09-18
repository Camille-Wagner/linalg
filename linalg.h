#ifndef LINALG_H
#define LINALG_H

#define LINALG_NO_SIMD

#define EPSILON 0.000000000000001
#define STRASSEN_THRESHHOLD 128

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef LINALG_USE_NEON
#undef LINALG_NO_SIMD
#include <arm_neon.h>
#endif

#ifdef LINALG_COMPLEX
#include <complex.h>
#define scalar_t complex 
#else
#define scalar_t float 
#endif

// ************************************************************************************
//
// Library Structures
//
// ************************************************************************************

typedef struct {
    size_t num_rows;
    size_t num_cols;
    float *data;
} mat;

typedef struct {
    float *base;
    size_t capacity;
    size_t offset;
} arena_t;

float *_arena_alloc_unsafe(arena_t *arena, size_t count) {
    float *p = arena->base + arena->offset;
    arena->offset += count;
    return p;
}

// ************************************************************************************
//
// Error Handling
//
// ************************************************************************************

typedef enum {
    OK,
    NULL_ARGUMENT,
    ALLOCATION_FAILED,
    OUT_OF_BOUNDS,
    DIMENSION_MISMATCH,
} err;

const char *err_string(const err *error) {
    switch(*error) {
        case OK:
            return "No error";
        case NULL_ARGUMENT:
            return "Null argument";
        case ALLOCATION_FAILED:
            return "Memory allocation failed";
        case OUT_OF_BOUNDS:
            return "Index out of bounds";
        case DIMENSION_MISMATCH:
            return "Dimensions do not match";
        default:
            return "Unkown error";
    }
}

void err_print_fmt(const char *err_str, const char *d_fmt) {
    if (err_str == NULL) {
        fprintf(stdout, "Error string is NULL \n");
        return;
    }

    fprintf(stderr, d_fmt, err_str);
}

void err_print(const char *err_str) {
    err_print_fmt(err_str, "%s\n");
}

// ************************************************************************************
//
// Matrix Printing
//
// ************************************************************************************

void mat_print_fmt(const mat *matrix, const char *d_fmt) {
    if (matrix == NULL) {
        fprintf(stdout, "Matrix is NULL \n");
        return;
    }

    fprintf(stdout, "\n");
    for (size_t i = 0; i < matrix->num_rows; ++i) {
        for (size_t j = 0; j < matrix->num_cols; ++j) {
            fprintf(stdout, d_fmt, matrix->data[i * matrix->num_cols + j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

void mat_print(const mat *matrix) {
    mat_print_fmt(matrix, "%f\t");
}

// ************************************************************************************
//
// Matrix Allocation and Destruction
//
// ************************************************************************************

mat *mat_new(size_t num_rows, size_t num_cols) {
    if (num_rows == 0 || num_cols == 0) {
        return NULL;
    }

    if (num_rows > SIZE_MAX / num_cols) {
        return NULL;
    }

    mat *m = calloc(1, sizeof(*m));

    if (m == NULL) {
        return NULL;
    }

    m->num_rows = num_rows;
    m->num_cols = num_cols;
    m->data = calloc(m->num_rows * m->num_cols, sizeof(*m->data));

    if (m->data == NULL) {
        free(m);
        return NULL;
    }

    return m;
}

void mat_free(mat *matrix) {
    if (matrix == NULL) {
        return;
    }
    free(matrix->data);
    free(matrix);
}

mat *mat_id(size_t size) {
    mat *id = mat_new(size, size);

    if (id == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < size; ++i) {
        id->data[i * size + i] = 1.0;
    }

    return id;
}

float rand_interval(float min, float max) {
    float d = (float) rand() / (float) RAND_MAX;
    return min + d * (max - min);
}

mat *mat_rnd(size_t num_rows, size_t num_cols, float min, float max) {
    mat *m = mat_new(num_rows, num_cols);

    if (m == NULL) {
        return NULL;
    }

    for (size_t idx = 0; idx < m->num_rows * m->num_cols; ++idx) {
        m->data[idx] = rand_interval(min, max);
    }

    return m;
}

mat *mat_cp_unsafe(const mat *matrix) {
    mat *new = mat_new(matrix->num_rows, matrix->num_cols);

    if (new == NULL) {
        return NULL;
    }

    memcpy(new->data, matrix->data, matrix->num_rows * matrix->num_cols * sizeof(*matrix->data));

    return new;
}

mat *mat_cp(const mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    mat *new = mat_new(matrix->num_rows, matrix->num_cols);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    memcpy(new->data, matrix->data, matrix->num_rows * matrix->num_cols * sizeof(*matrix->data));

    return new;
}

// ************************************************************************************
//
// Matrix Equality
//
// ************************************************************************************

bool mat_eq(const mat *left, const mat *right) {
    if (left == NULL || right == NULL) {
        return false;
    }

    if (left->num_rows != right->num_rows || left->num_cols != right->num_cols) {
        return false;
    }

    for (size_t idx = 0; idx < left->num_rows * left->num_cols; ++idx) {
        if (fabs(left->data[idx] - right->data[idx]) > EPSILON) {
            return false;
        }
    }

    return true;
}

// ************************************************************************************
//
// Accessing and Setting Matrix Elements
//
// ************************************************************************************

float mat_element_get(const mat *matrix, size_t i, size_t j, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return 0.0;
    };

    if (i >= matrix->num_rows || j >= matrix->num_cols) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return 0.0;
    };

    return matrix->data[i * matrix->num_cols + j];
}

void mat_element_set(mat *matrix, size_t i, size_t j, float val, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }

    if (i >= matrix->num_rows || j >= matrix->num_cols) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return;
    }

    matrix->data[i * matrix->num_cols + j] = val;

    if (error != NULL) {
        *error = OK;
    }
}

void mat_all_set(mat *matrix, float val, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }

    for (size_t idx = 0; idx < matrix->num_rows * matrix->num_cols; ++idx) {
        matrix->data[idx] = val;
    }

    if (error != NULL) {
        *error = OK;
    }
}

mat *mat_row_get(const mat *matrix, size_t i, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (i >= matrix->num_rows) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return NULL;
    };

    mat *row = mat_new(1, matrix->num_cols);
    
    if (row == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    memcpy(row->data, &matrix->data[i * matrix->num_cols], matrix->num_cols * sizeof(float));
    
    if (error != NULL) {
        *error = OK;
    }
    return row;
}

mat *mat_col_get(const mat *matrix, size_t j, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (j >= matrix->num_cols) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return NULL;
    };

    mat *col = mat_new(matrix->num_rows, 1);

    if (col == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    for (size_t i = 0; i < matrix->num_rows; ++i) {
        col->data[i] = matrix->data[i * matrix->num_cols + j];
    }

    if (error != NULL) {
        *error = OK;
    }
    return col;
}

// ************************************************************************************
//
// Matrix Concatenation
//
// ************************************************************************************

mat *mat_cat_v(size_t num, mat **mat_arr, err *error) {
    if (mat_arr[0] == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }
    size_t tot_rows = mat_arr[0]->num_rows;
    size_t target_cols = mat_arr[0]->num_cols;

    for (size_t i = 1; i < num; ++i) {
        if (mat_arr[i] == NULL) {
            if (error != NULL) {
                *error = NULL_ARGUMENT;
            }
            return NULL;
        }

        if (mat_arr[i]->num_cols != target_cols) {
            if (error != NULL) {
                *error = DIMENSION_MISMATCH;
            }
            return NULL;
        }
        
        tot_rows += mat_arr[i]->num_rows;
    }

    mat *new = mat_new(tot_rows, target_cols);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < num; ++i) {
        memcpy(&new->data[current_offset], mat_arr[i]->data, mat_arr[i]->num_rows * mat_arr[i]->num_cols * sizeof(float));
        current_offset += mat_arr[i]->num_rows * mat_arr[i]->num_cols;
    }

    if (error != NULL) {
        *error = OK;
    }
    return new;
}

mat *mat_cat_h(size_t num, mat **mat_arr, err *error) {
    if (mat_arr[0] == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }
    
    size_t target_rows = mat_arr[0]->num_rows;
    size_t tot_cols = mat_arr[0]->num_cols;

    for (size_t i = 1; i < num; ++i) {
        if (mat_arr[i] == NULL) {
            if (error != NULL) {
                *error = NULL_ARGUMENT;
            }
            return NULL;
        }

        if (mat_arr[i]->num_rows != target_rows) {
            if (error != NULL) {
                *error = DIMENSION_MISMATCH;
            }
            return NULL;
        }
        tot_cols += mat_arr[i]->num_cols;
    }

    mat *new = mat_new(target_rows, tot_cols);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }
    
    size_t current_offset = 0;
    for (size_t row = 0; row < target_rows; ++row) {
        for (size_t i = 0; i < num; ++i) {
            size_t curr_cols = mat_arr[i]->num_cols;
            memcpy(&new->data[row * tot_cols + current_offset], &mat_arr[i]->data[row * curr_cols], mat_arr[i]->num_cols * sizeof(float));
            current_offset += curr_cols;
        }
        current_offset = 0;
    }

    if (error != NULL) {
        *error = OK;
    }
    return new;
}

// ************************************************************************************
//
// Matrix Operations
//
// ************************************************************************************

mat *mat_add(const mat *left, const mat *right, err *error) {
    if (left == NULL || right == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (left->num_rows != right->num_rows || left->num_cols != right->num_cols) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return NULL;
    };

    #ifdef LINALG_USE_NEON
    
    size_t n = left->num_rows;
    size_t m = left->num_cols;

    mat *C = mat_new(n, m);

    if (C == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    size_t i = 0;
    for (; i + 4 <= n * m; i += 4) {
        float32x4_t a_seg = vld1q_f32(left->data + i);
        float32x4_t b_seg = vld1q_f32(right->data + i);
        float32x4_t res = vaddq_f32(a_seg, b_seg);
        vst1q_f32(C->data + i, res);
    }
    for(; i < n * m; ++i) {
        C->data[i] = left->data[i] + right->data[i];
    }

    if (error != NULL) {
        *error = OK;
    }

    return C;

    #endif

    #ifdef LINALG_NO_SIMD

    mat *res = mat_new(left->num_rows, left->num_cols);

    if (res == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    for (size_t idx = 0; idx < res->num_rows * res->num_cols; ++idx) {
        res->data[idx] = left->data[idx] + right->data[idx];
    }

    if (error != NULL) {
        *error = OK;
    }
    return res;
    #endif
}

void mat_add_r(mat *left, const mat *right, err *error) {
    if (left == NULL || right == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }

    if (left->num_rows != right->num_rows || left->num_cols != right->num_cols) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return;
    };

    #ifdef LINALG_USE_NEON
    size_t n = left->num_rows;
    size_t m = left->num_cols;

    size_t i = 0;
    for(; i + 4 <= n * m; i += 4) {
        float32x4_t a_seg = vld1q_f32(left->data + i);
        float32x4_t b_seg = vld1q_f32(right->data + i);
        a_seg = vaddq_f32(a_seg, b_seg);
        vst1q_f32(left->data + i, a_seg);
    }

    for (; i < n * m; ++i) {
        left->data[i] += right->data[i];
    }

    #endif

    #ifdef LINALG_NO_SIMD
    for (size_t idx = 0; idx < left->num_rows * left->num_cols; ++idx) {
        left->data[idx] += right->data[idx];
    }
    #endif

    if (error != NULL) {
        *error = OK;
    }
}

mat *mat_sub(const mat *left, const mat *right, err *error) {
    if (left == NULL || right == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (left->num_rows != right->num_rows || left->num_cols != right->num_cols) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return NULL;
    };

    #ifdef LINALG_USE_NEON

    size_t n = left->num_rows;
    size_t m = left->num_cols;

    mat *C = mat_new(n, m);

    if (C == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    size_t i = 0;
    for (; i + 4 <= n * m; i += 4) {
        float32x4_t a_seg = vld1q_f32(left->data + i);
        float32x4_t b_seg = vld1q_f32(right->data + i);
        float32x4_t res = vsubq_f32(a_seg, b_seg);
        vst1q_f32(C->data + i, res);
    }
    for(; i < n * m; ++i) {
        C->data[i] = left->data[i] - right->data[i];
    }

    if (error != NULL) {
        *error = OK;
    }

    return C;

    #endif

    #ifdef LINALG_NO_SIMD
    mat *res = mat_new(left->num_rows, left->num_cols);

    if (res == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    for (size_t idx = 0; idx < left->num_rows * left->num_cols; ++idx) {
        res->data[idx] = left->data[idx] - right->data[idx];
    }

    if (error != NULL) {
        *error = OK;
    }
    return res;

    #endif
}

void mat_sub_r(mat *left, const mat *right, err *error) {
    if (left == NULL || right == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }

    if (left->num_rows != right->num_rows || left->num_cols != right->num_cols) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return;
    };

    #ifdef LINALG_USE_NEON
    size_t n = left->num_rows;
    size_t m = left->num_cols;

    size_t i = 0;
    for(; i + 4 <= n * m; i += 4) {
        float32x4_t a_seg = vld1q_f32(left->data + i);
        float32x4_t b_seg = vld1q_f32(right->data + i);
        a_seg = vsubq_f32(a_seg, b_seg);
        vst1q_f32(left->data + i, a_seg);
    }

    for (; i < n * m; ++i) {
        left->data[i] -= right->data[i];
    }

    #endif

    #ifdef LINALG_NO_SIMD
    for (size_t idx = 0; idx < left->num_rows * left->num_cols; ++idx) {
        left->data[idx] -= right->data[idx];
    }  

    if (error != NULL) {
        *error = OK;
    }
    #endif
}

mat *mat_scalar_mult(const mat *matrix, float scalar, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    mat *res = mat_new(matrix->num_rows, matrix->num_cols);

    if (res == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    for (size_t idx = 0; idx < res->num_rows * res->num_cols; ++idx) {
        res->data[idx] = scalar * matrix->data[idx];
    }

    if (error != NULL) {
        *error = OK;
    }
    return res;
}

void mat_scalar_mult_r(mat *matrix, float scalar, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }

    for (size_t idx = 0; idx < matrix->num_rows * matrix->num_cols; ++idx) {
        matrix->data[idx] *= scalar;
    }

    if (error != NULL) {
        *error = OK;
    }
}

mat *mat_mult(const mat *left, const mat *right, err *error) {
    if (left == NULL || right == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (left->num_cols != right->num_rows) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return NULL;
    };
    
    #ifdef LINALG_USE_NEON

    size_t n = left->num_rows; 
    size_t k = left->num_cols; 
    size_t m = right->num_cols;

    mat *C = mat_new(n, m);  
    if (C == NULL) {
        if (error != NULL) *error = ALLOCATION_FAILED;
        return NULL;
    }

    size_t i = 0;
    for (; i +4 <= n; i += 4) {
        size_t j = 0;
        for (; j + 4 <= m; j += 4) {
            float32x4_t acc0 = vmovq_n_f32(0.0f); // will hold C[i,   j:j+3]
            float32x4_t acc1 = vmovq_n_f32(0.0f); // will hold C[i+1, j:j+3]
            float32x4_t acc2 = vmovq_n_f32(0.0f); // will hold C[i+2, j:j+3]
            float32x4_t acc3 = vmovq_n_f32(0.0f); // will hold C[i+3, j:j+3]

            for (size_t p = 0; p < k; ++p) {
                // one B row-slice - B[p, j:j+3] 
                float32x4_t b_row = vld1q_f32(right->data + p * m + j); 

                acc0 = vfmaq_n_f32(acc0, b_row, left->data[i * k + p]);        // acc0 += b_row * A[i][p]
                acc1 = vfmaq_n_f32(acc1, b_row, left->data[(i + 1) * k + p]);  // acc1 += b_row * A[i+1][p]
                acc2 = vfmaq_n_f32(acc2, b_row, left->data[(i + 2) * k + p]);  // acc2 += b_row * A[i+2][p]
                acc3 = vfmaq_n_f32(acc3, b_row, left->data[(i + 3) * k + p]);  // acc3 += b_row * A[i+3][p]

            }
            vst1q_f32(C->data + i * m + j, acc0);
            vst1q_f32(C->data + (i + 1) * m + j, acc1);
            vst1q_f32(C->data + (i + 2) * m + j, acc2);
            vst1q_f32(C->data + (i + 3) * m + j, acc3);
        }
        // leftover columns (m % 4 != 0) for this row-block: scalar, one element at a time
        for (; j < m; ++j) {
            for (size_t r = 0; r < k; ++r) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += left->data[(i + r) * k + p] * right->data[p * m + j];
                }
                C->data[(i + r) * m + j] = sum;
            }
        }
    }
    // leftover rows (n % 4 != 0): fall back to the single-direction (1-row) routine from before
    for (; i < n; ++i) {
        size_t j = 0;
        for (; j + 4 <= m; j += 4) {
            float32x4_t acc = vmovq_n_f32(0.0f);

            for (size_t p = 0; p < k; ++p) {
                float32x4_t b_row = vld1q_f32(right->data + p * m + j);
                acc = vfmaq_n_f32(acc, b_row, left->data[i * k + p]);
            }
            vst1q_f32(C->data + i * m + j, acc);
        }
        for (; j < m; ++j) {
            float sum = 0.0f;

            for (size_t p = 0; p < k; ++p) {
                sum += left->data[i * k + p] * right->data[p * m + j];
            }
            C->data[i * m + j] = sum;
        }
    }

    if (error != NULL) {
        *error = OK;
    }
    return C;

    #endif

    #ifdef LINALG_NO_SIMD

    mat *res = mat_new(left->num_rows, right->num_cols);

    if (res == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    for (size_t i = 0; i < res->num_rows; ++i) {
        for (size_t j = 0; j < res->num_cols; ++j) {
            float sum = 0.0;
            for (size_t k = 0; k < left->num_cols; ++k) {
                sum += left->data[i * left->num_cols + k] * right->data[k * right->num_cols + j];
            }
            res->data[i * res->num_cols + j] = sum;
        }
    }


    if (error != NULL) {
        *error = OK;
    }
    return res;
    #endif
}

void _strassen_arena(const float *restrict A, size_t lda, const float *restrict B, size_t ldb, float *restrict C, size_t ldc, arena_t *arena, size_t n) {
    if (n <= STRASSEN_THRESHHOLD) {
        for (size_t i = 0; i < n; ++i) {
            size_t k = 0;
            float a = A[i * lda + k];
            for (size_t j = 0; j < n; ++j) {
                    C[i * ldc + j] = a * B[k * ldb + j];
                }
            for (size_t k = 1; k < n; ++ k) {
                float a = A[i * lda + k];
                for (size_t j = 0; j < n; ++j) {
                    C[i * ldc + j] += a * B[k * ldb + j];
                }
            }
        }
    }

    else {
        size_t k = n/2;
        size_t save_offset = arena->offset;

        float *P1 = _arena_alloc_unsafe(arena, k * k);
        float *P2 = _arena_alloc_unsafe(arena, k * k);
        float *P3 = _arena_alloc_unsafe(arena, k * k);
        float *P4 = _arena_alloc_unsafe(arena, k * k);
        float *P5 = _arena_alloc_unsafe(arena, k * k);
        float *P6 = _arena_alloc_unsafe(arena, k * k);
        float *P7 = _arena_alloc_unsafe(arena, k * k);

        float *S1 = _arena_alloc_unsafe(arena, k * k);
        float *S2 = _arena_alloc_unsafe(arena, k * k);
        float *S3 = _arena_alloc_unsafe(arena, k * k);
        float *S4 = _arena_alloc_unsafe(arena, k * k);
        float *S5 = _arena_alloc_unsafe(arena, k * k);
        float *S6 = _arena_alloc_unsafe(arena, k * k);
        float *S7 = _arena_alloc_unsafe(arena, k * k);
        float *S8 = _arena_alloc_unsafe(arena, k * k);
        float *S9 = _arena_alloc_unsafe(arena, k * k);
        float *S10 = _arena_alloc_unsafe(arena, k * k);

        // Pointers to the start of their respective slice in their parent buffer
        const float *A11 = A;             
        const float *A12 = A + k;
        const float *A21 = A + k * lda;
        const float *A22 = A + k * lda + k;

        const float *B11 = B;              
        const float *B12 = B + k;
        const float *B21 = B + k * ldb;
        const float *B22 = B + k * ldb + k;

        float *C11 = C;              
        float *C12 = C + k;
        float *C21 = C + k * ldc;    
        float *C22 = C + k * ldc + k;

        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                S1[i * k + j]  = A11[i * lda + j] + A21[i * lda + j]; // A11 + A21
                S3[i * k + j]  = A12[i * lda + j] + A22[i * lda + j]; // A12 + A22
                S5[i * k + j]  = A11[i * lda + j] - A22[i * lda + j]; // A11 - A22
                S8[i * k + j]  = A21[i * lda + j] + A22[i * lda + j]; // A21 + A22
                S9[i * k + j]  = A11[i * lda + j] + A12[i * lda + j]; // A11 + A12

                S2[i * k + j]  = B11[i * ldb + j] + B12[i * ldb + j]; // B11 + B12
                S4[i * k + j]  = B21[i * ldb + j] + B22[i * ldb + j]; // B21 + B22
                S6[i * k + j]  = B11[i * ldb + j] + B22[i * ldb + j]; // B11 + B22
                S7[i * k + j]  = B12[i * ldb + j] - B22[i * ldb + j]; // B12 - B22
                S10[i * k + j] = B21[i * ldb + j] - B11[i * ldb + j]; // B21 - B11
            }
        }

        _strassen_arena(S1, k, S2, k, P1, k, arena, k);     // (A11 + A21) * (B11 + B12)
        _strassen_arena(S3, k, S4, k, P2, k, arena, k);     // (A12 + A22) * (B21 + B22)
        _strassen_arena(S5, k, S6, k, P3, k, arena, k);     // (A11 - A22) * (B11 + B22)
        _strassen_arena(A11, lda, S7, k, P4, k, arena, k);  // A11 * (B12 - B22)
        _strassen_arena(S8, k, B11, ldb, P5, k, arena, k);  // (A21 + A22) * B11
        _strassen_arena(S9, k, B22, ldb, P6, k, arena, k);  // (A11 + A12) * B22
        _strassen_arena(A22, lda, S10, k, P7, k, arena, k); // A22 * (B21 - B11)

       for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                C11[i * ldc + j] = P2[i * k + j] + P3[i * k + j] - P6[i * k + j] - P7[i * k + j]; // C11 = P2 + P3 - P6 - P7
                C12[i * ldc + j] = P4[i * k + j] + P6[i * k + j];                                 // C12 = P4 + P6
                C21[i * ldc + j] = P5[i * k + j] + P7[i * k + j];                                 // C21 = P5 + P7
                C22[i * ldc + j] = P1[i * k + j] - P3[i * k + j] - P4[i * k + j] - P5[i * k + j]; // C22 = P1 - P3 - P4 - P5
            }
       }

        arena->offset = save_offset;
    }
}

size_t _strassen_arena_size(size_t n) {
    if (n < STRASSEN_THRESHHOLD) {
        return 0;
    }
    size_t total = 0;
    while (n > STRASSEN_THRESHHOLD) {
        size_t k = n / 2;
        total += 17 * k * k;
        n = k;
    }
    return total;
}

mat *mat_mult_strassen_arena(const mat *left, const mat *right, err *error){
    if (left == NULL || right == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (left->num_cols != right->num_rows) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return NULL;
    };

    mat *C = mat_new(left->num_rows, left->num_rows);

    if (C == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    size_t n = left->num_rows; // assume A and B are square matrices with a power of two dimensions
    size_t arena_capacity = _strassen_arena_size(n);

    arena_t arena = { .base = malloc(arena_capacity * sizeof(float)), .capacity = arena_capacity, .offset = 0};

    if (arena.base == NULL) {
        return NULL;
    }

    _strassen_arena(left->data, n, right->data, n, C->data, n, &arena, n);

    return C;
}

void _strassen(const float *restrict A, const float *restrict B, float *restrict C, size_t n) {
    // Base case
    if (n <= STRASSEN_THRESHHOLD) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < n; ++ k) {
                float a = A[i * n + k];
                for (size_t j = 0; j < n; ++j) {
                    C[i * n + j] += a * B[k * n + j];
                }
            }
        }
    }

    // Recursion step
    else {
        size_t k = n/2;

        float *P1 = malloc(k * k * sizeof(float));
        float *P2 = malloc(k * k * sizeof(float));
        float *P3 = malloc(k * k * sizeof(float));
        float *P4 = malloc(k * k * sizeof(float));
        float *P5 = malloc(k * k * sizeof(float));
        float *P6 = malloc(k * k * sizeof(float));
        float *P7 = malloc(k * k * sizeof(float));

        float *S1 = malloc(k * k * sizeof(float));
        float *S2 = malloc(k * k * sizeof(float));
        float *S3 = malloc(k * k * sizeof(float));
        float *S4 = malloc(k * k * sizeof(float));
        float *S5 = malloc(k * k * sizeof(float));
        float *S6 = malloc(k * k * sizeof(float));
        float *S7 = malloc(k * k * sizeof(float));
        float *S8 = malloc(k * k * sizeof(float));
        float *S9 = malloc(k * k * sizeof(float));
        float *S10 = malloc(k * k * sizeof(float));

        float *A11 = malloc(k * k * sizeof(float));
        float *A22 = malloc(k * k * sizeof(float));
        float *B11 = malloc(k * k * sizeof(float));
        float *B22 = malloc(k * k * sizeof(float));

        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                S1[i * k + j]  = A[i * n + j] + A[(i + k) * n + j];             // A11 + A21
                S3[i * k + j]  = A[i * n + (j + k)] + A[(i + k) * n + (j + k)]; // A12 + A22
                S5[i * k + j]  = A[i * n + j] - A[(i + k) * n + (j + k)];       // A11 - A22
                S8[i * k + j]  = A[(i + k) * n + j] + A[(i + k) * n + (j + k)]; // A21 + A22
                S9[i * k + j]  = A[i * n + j] + A[i * n + (j + k)];             // A11 + A12

                S2[i * k + j]  = B[i * n + j] + B[i * n + (j + k)];             // B11 + B12
                S4[i * k + j]  = B[(i + k) * n + j] + B[(i + k) * n + (j + k)]; // B21 + B22
                S6[i * k + j]  = B[i * n + j] + B[(i + k) * n + (j + k)];       // B11 + B22
                S7[i * k + j]  = B[i * n + (j + k)] - B[(i + k) * n + (j + k)]; // B12 - B22
                S10[i * k + j] = B[(i + k) * n + j] - B[i * n + j];             // B21 - B11

                A11[i * k + j] = A[i * n + j];
                A22[i * k + j] = A[(i + k) * n + (j + k)];


                B11[i * k + j] = B[i * n + j];
                B22[i * k + j] = B[(i + k) * n + (j + k)];
            }
        }

        _strassen(S1, S2, P1, k);   // (A11 + A21) * (B11 + B12)
        _strassen(S3, S4, P2, k);   // (A12 + A22) * (B21 + B22)
        _strassen(S5, S6, P3, k);   // (A11 - A22) * (B11 + B22)
        _strassen(A11, S7, P4, k);  // A11 * (B12 - B22)
        _strassen(S8, B11, P5, k);  // (A21 + A22) * B11
        _strassen(S9, B22, P6, k);  // (A11 + A12) * B22
        _strassen(A22, S10, P7, k); // A22 * (B21 - B11)

        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                C[i * n + j] = P2[i * k + j] + P3[i * k + j] - P6[i * k + j] - P7[i * k + j];             // C11 = P2 + P3 - P6 - P7
                C[i * n + (j + k)] = P4[i * k + j] + P6[i * k + j];                                       // C12 = P4 + P6
                C[(i + k) * n + j] = P5[i * k + j] + P7[i * k + j];                                       // C21 = P5 + P7
                C[(i + k) * n + (j + k)] = P1[i * k + j] - P3[i * k + j] - P4[i * k + j] - P5[i * k + j]; // C22 = P1 - P3 - P4 - P5
            }
        }

        free(P1);
        free(P2);
        free(P3);
        free(P4);
        free(P5);
        free(P6);
        free(P7);

        free(S1);
        free(S2);
        free(S3);
        free(S4);
        free(S5);
        free(S6);
        free(S7);
        free(S8);
        free(S9);
        free(S10);

        free(A11);
        free(A22);
        free(B11);
        free(B22);
    }
}

mat *mat_mult_strassen(const mat *left, const mat *right, err *error) {
    if (left == NULL || right == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (left->num_cols != right->num_rows) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return NULL;
    };

    mat *C = mat_new(left->num_rows, right->num_cols);

    if (C == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    _strassen(left->data, right->data, C->data, left->num_rows);

    if (error != NULL) {
        *error = OK;
    }
    return C;
}

size_t transposition_permutation(size_t idx, size_t num_rows, size_t num_cols) {
    if (idx == num_rows * num_cols - 1) {
        return idx;
    }
    else {
        return (num_rows * idx) % (num_rows * num_cols - 1);
    }
} 

mat *mat_transpose(const mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    mat *res = mat_new(matrix->num_cols, matrix->num_rows);

    if (res == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    for (size_t idx = 0; idx < res->num_rows * res->num_cols; ++idx) {
        res->data[transposition_permutation(idx, matrix->num_rows, matrix->num_cols)] = matrix->data[idx];
    }

    if (error != NULL) {
        *error = OK;
    }
    return res;
}

void mat_transpose_sqr_r(mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }

    if (matrix->num_rows != matrix->num_cols) {
        if (error != NULL) {
            *error = DIMENSION_MISMATCH;
        }
        return;
    }

    size_t dim = matrix->num_rows;

    for (size_t i = 0; i < dim - 1; ++i) {
        for (size_t j = i + 1; j < dim; ++j) {
            float tmp = matrix->data[i * dim + j];
            matrix->data[i * dim + j] = matrix->data[j * dim + i];
            matrix->data[j * dim + i] = tmp;
        }
    }

    if (error != NULL) {
        *error = OK;
    }
    return;
}

// ************************************************************************************
//
// Elementary Row Operations
//
// ************************************************************************************

void mat_row_swap_r_unsafe(mat *matrix, size_t row_1, size_t row_2) {
    float tmp[matrix->num_cols]; 
    memcpy(tmp, &matrix->data[row_1 * matrix->num_cols], matrix->num_cols * sizeof(float));
    memcpy(&matrix->data[row_1 * matrix->num_cols], &matrix->data[row_2 * matrix->num_cols], matrix->num_cols * sizeof(float));
    memcpy(&matrix->data[row_2 * matrix->num_cols], tmp, matrix->num_cols * sizeof(float));
    return;
}

void mat_row_scalar_mult_r_unsafe(mat *matrix, size_t row, float scalar) {
    for (size_t j = 0; j < matrix->num_cols; ++j) {
        matrix->data[row * matrix->num_cols + j] *= scalar;
    }
    return;
}

void mat_row_addrow_r_unsafe(mat *matrix, size_t target_row, size_t row, float scalar) {
    for (size_t j = 0; j < matrix->num_cols; ++j) {
        matrix->data[target_row * matrix->num_cols + j] += scalar * matrix->data[row * matrix->num_cols + j];
    }
    return;
}

void mat_row_swap_r(mat *matrix, size_t row_1, size_t row_2, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }

    if (row_1 >= matrix->num_rows || row_2 >= matrix->num_rows) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return;
    }
    
    if (row_1 == row_2) {
        return;
    }

    float tmp[matrix->num_cols]; 
    memcpy(tmp, &matrix->data[row_1 * matrix->num_cols], matrix->num_cols * sizeof(float));
    memcpy(&matrix->data[row_1 * matrix->num_cols], &matrix->data[row_2 * matrix->num_cols], matrix->num_cols * sizeof(float));
    memcpy(&matrix->data[row_2 * matrix->num_cols], tmp, matrix->num_cols * sizeof(float));

    if (error != NULL) {
        *error = OK;
    }
    return;
}

mat *mat_row_swap(const mat *matrix, size_t row_1, size_t row_2, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }

    if (row_1 >= matrix->num_rows || row_2 >= matrix->num_rows) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return NULL;
    }

    mat *new = mat_cp_unsafe(matrix);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    if (row_1 == row_2) {
        return new;
    }

    mat_row_swap_r_unsafe(new, row_1, row_2);

    if (error != NULL) {
        *error = OK;
    }
    return new;
}

void mat_row_scalar_mult_r(mat *matrix, size_t row, float scalar, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }  

    if (row >= matrix->num_rows) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return;
    }

    for (size_t j = 0; j < matrix->num_cols; ++j) {
        matrix->data[row * matrix->num_cols + j] *= scalar;
    }

    if (error != NULL) {
        *error = OK;
    }
    return;
}

mat *mat_row_scalar_mult(const mat *matrix, size_t row, float scalar, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }  

    if (row >= matrix->num_rows) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return NULL;
    }

    mat *new = mat_cp_unsafe(matrix);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    mat_row_scalar_mult_r_unsafe(new, row, scalar);

    if (error != NULL) {
        *error = OK;
    }
    return new;
}

void mat_row_addrow_r(mat *matrix, size_t target_row, size_t row, float scalar, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }  

    if (row >= matrix->num_rows || target_row >= matrix->num_rows) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return;
    }

    for (size_t j = 0; j < matrix->num_cols; ++j) {
        matrix->data[target_row * matrix->num_cols + j] += scalar * matrix->data[row * matrix->num_cols + j];
    }

    if (error != NULL) {
        *error = OK;
    }
    return;
}

mat *mat_row_addrow(const mat *matrix, size_t target_row, size_t row, float scalar, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }  

    if (row >= matrix->num_rows || target_row >= matrix->num_rows) {
        if (error != NULL) {
            *error = OUT_OF_BOUNDS;
        }
        return NULL;
    }

    mat *new = mat_cp_unsafe(matrix);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    mat_row_addrow_r_unsafe(new, target_row, row, scalar);

    if (error != NULL) {
        *error = OK;
    }
    return new;
}

// ************************************************************************************
//
// Row-Echelon Form
//
// ************************************************************************************

int _mat_find_pivot_row(const mat *matrix, size_t row, size_t col) {
    int max_i = row;
    float max_val = fabs(matrix->data[row * matrix->num_cols + col]);

    for (size_t i = row + 1; i < matrix->num_rows; ++i) {
        float current_val = fabs(matrix->data[i * matrix->num_cols + col]);
        if (current_val > max_val) {
            max_i = i;
            max_val = current_val;
        }
    }

    return (max_val > EPSILON) ? max_i : -1;
}

void mat_ref_r(mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }
    
    size_t h = 0; // Current pivot row
    size_t k = 0; // Current pivot column

    while (h < matrix->num_rows && k < matrix->num_cols) {
        int i_max = _mat_find_pivot_row(matrix, h, k); // Find the pivot in the k-th column

        if (i_max < 0) {
            ++k; // No pivot in current column, pass to next column
        }

        else {
            size_t pivot = (size_t)i_max;
            mat_row_swap_r_unsafe(matrix, h, pivot);

            // Repeat for all rows below the pivot
            for (size_t i = h + 1; i < matrix->num_rows; ++i) {
                float f = matrix->data[i * matrix->num_cols + k] / matrix->data[h * matrix->num_cols + k];
                matrix->data[i * matrix->num_cols + k] = 0.0; // Set element in pivot column to 0 (equivalent to including j = k in subsequent for loop)
                
                for (size_t j = k + 1; j < matrix->num_cols; ++j) {
                    matrix->data[i * matrix->num_cols + j] -= f * matrix->data[h * matrix->num_cols + j];
                }
            }
        }

        ++h;
        ++k;
    }

    if (error != NULL) {
        *error = OK;
    }
    return;
}

mat *mat_ref(const mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }
    
    mat *new = mat_cp_unsafe(matrix);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    mat_ref_r(new, NULL);

    if (error != NULL) {
        *error = OK;
    }
    return new;
}

int _find_pivot_col(mat *matrix, size_t row) {
    for (size_t col = 0; col < matrix->num_cols; ++col) {
        if (fabs(matrix->data[row * matrix->num_cols + col]) > EPSILON) {
            return (int)col;
        }
    }
    return -1;
}

void mat_rref_r(mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }
    
    size_t h = 0; // Current pivot row
    size_t k = 0; // Current pivot column

    // 1. Forward Pass
    while (h < matrix->num_rows && k < matrix->num_cols) {
        int i_max = _mat_find_pivot_row(matrix, h, k); // Find the pivot in the k-th column

        if (i_max < 0) {
            ++k; // No pivot in current column, pass to next column
        }

        else {
            size_t pivot = (size_t)i_max;
            mat_row_swap_r_unsafe(matrix, h, pivot);

             // Repeat for all rows below the pivot
            for (size_t i = h + 1; i < matrix->num_rows; ++i) {
                float f = matrix->data[i * matrix->num_cols + k] / matrix->data[h * matrix->num_cols + k];
                matrix->data[i * matrix->num_cols + k] = 0.0; // Set element in pivot column to 0 (equivalent to including j = k in subsequent for loop)
                
                for (size_t j = k + 1; j < matrix->num_cols; ++j) {
                    matrix->data[i * matrix->num_cols + j] -= f * matrix->data[h * matrix->num_cols + j];
                }
            }
        }

        ++h;
        ++k;
    }

    // 2. Backward Pass
    for (int i = h - 1; i >= 0; --i) {
        int p = _find_pivot_col(matrix, i);
        if (p < 0) {
            continue; // No pivot in current row
        }
        size_t pivot_col = (size_t)p;

        // Set pivot in current row to one
        float f = matrix->data[i * matrix->num_cols + pivot_col];
        matrix->data[i * matrix->num_cols + pivot_col] = 1.0;

        for (size_t j = pivot_col + 1; j < matrix->num_cols; ++j) {
            matrix->data[i * matrix->num_cols + j] /= f; 
        }

        for (size_t row_above = i - 1; row_above < matrix->num_rows; --row_above) {
            float ff = matrix->data[row_above * matrix->num_cols + pivot_col];
            mat_row_addrow_r_unsafe(matrix, row_above, i, -ff);
        }
    }

    if (error != NULL) {
        *error = OK;
    }
    return;
}

mat *mat_rref(const mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return NULL;
    }
    
    mat *new = mat_cp_unsafe(matrix);

    if (new == NULL) {
        if (error != NULL) {
            *error = ALLOCATION_FAILED;
        }
        return NULL;
    }

    mat_rref_r(new, NULL);

    if (error != NULL) {
        *error = OK;
    }
    return new;
}

// ************************************************************************************
//
// LU Decomposition
//
// ************************************************************************************


// ************************************************************************************
//
// deprecated
//
// ************************************************************************************

[[deprecated]] void mat_ref_explicit_r(mat *matrix, err *error) {
    if (matrix == NULL) {
        if (error != NULL) {
            *error = NULL_ARGUMENT;
        }
        return;
    }
    
    size_t h = 0;
    size_t k = 0;

    while (h < matrix->num_rows && k < matrix->num_cols) {
        int i_max = _mat_find_pivot_row(matrix, h, k);

        if (i_max < 0) {
            ++k;
        }

        else {
            size_t pivot = (size_t)i_max;
            mat_row_swap_r_unsafe(matrix, h, pivot);

            for (size_t i = h + 1; i < matrix->num_rows; ++i) {
                float f = matrix->data[i * matrix->num_cols + k] / matrix->data[h * matrix->num_cols + k];
                
                mat_row_addrow_r_unsafe(matrix, i, h, -f);
            }

            ++h;
            ++k;
        }
    }

    if (error != NULL) {
        *error = OK;
    }
    return;
}

#endif 

/*
#TODO : SIMD for matrix addition (in place), matrix subtraction (in place)               - DONE
#TODO : SIMD for Gauss elimination -- probably very hard, study multilication case more
#TODO : Support for complex numbers: use vcmlaq_f32() for complex multiply-accumulate 
#TODO : Implement Strassen's algorithm for matrix multiplication                         - DONE
#TODO : Decompositions (LU, )

*/