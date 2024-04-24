#include "neural_network.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* RANDOM */
float randn() {
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    return sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
}
/* END RANDOM */



/* VECTOR FUNCTIONS */
Vector *vector_init(unsigned int rows, float *values) {
    Vector *vector;
    vector = malloc((sizeof (Vector)));
    if (vector == NULL) {
        return NULL;
    }

    vector->rows = rows;
    vector->values = values;

    return vector;
}


void vector_free(Vector *vector) {
    free(vector->values);
    free(vector);
}


void vector_print(Vector *vector) {
    printf("Vector([");
    unsigned int i;
    for (i = 0; i < vector->rows - 1; i++) {
        printf("%.2f ", vector->values[i]);
    }
    printf("%.2f])\n", vector->values[vector->rows - 1]);
}
/* END VECTOR FUNCTIONS */




/* MATRIX FUNCTIONS */
Matrix *matrix_init(unsigned int rows, unsigned int cols) {
    Matrix *matrix;
    matrix = malloc(sizeof (Matrix));
    if (matrix == NULL) {
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->values = malloc(rows * sizeof(float *));
    if (matrix == NULL) {
        free(matrix);
        return NULL;
    }

    unsigned int row;
    unsigned int col;
    for (row = 0; row < rows; row++) {
        matrix->values[row] = malloc(cols * sizeof(float));
        if (matrix->values[row] == NULL) {
            for (int i = 0; i < row; i++) {
                free(matrix->values[i]);
                free(matrix);
            }
            return NULL;
        }

        for (col = 0; col < cols; col++) {
            matrix->values[row][col] = randn();
        }
    }

    return matrix;
}


void matrix_free(Matrix *matrix) {
    unsigned int row;
    for (row = 0; row < matrix->rows; row++) {
        free(matrix->values[row]);
    }
    free(matrix->values);
    free(matrix);
}


void matrix_print(Matrix *matrix) {
    unsigned int row;
    unsigned int col;
    char *padding = "";
    printf("Matrix([");
    for (row = 0; row < matrix->rows - 1; row++) {
        printf("%s[", padding);
        padding = "        ";
        for (col = 0; col < matrix->cols - 1; col++) {
            printf("%.2f ", matrix->values[row][col]);
        }
        printf("%.2f],\n", matrix->values[row][col]);
    }

    printf("%s[", padding);
    for (col = 0; col < matrix->cols - 1; col++) {
        printf("%.2f ", matrix->values[row][col]);
    }
    printf("%.2f]])\n", matrix->values[row][col]);
}


Vector *matrix_times_vector(Matrix *matrix, Vector *vector) {
    if (matrix->cols != vector->rows) {
        printf("Error: matrix cols (%d) != vector rows (%d)\n", matrix->cols, vector->rows);
        return NULL;
    }

    float *vector_out = malloc(matrix->rows * sizeof(float));

    float dot_product;
    unsigned int mat_row;
    unsigned int mat_col;
    for (mat_row = 0; mat_row < matrix->rows; mat_row++) {
        dot_product = 0;
        for (mat_col = 0; mat_col < matrix->cols; mat_col++) {
            dot_product += matrix->values[mat_row][mat_col] * vector->values[mat_col];
        }
        vector_out[mat_row] = dot_product;
    }

    return vector_init(matrix->rows, vector_out);
}
/* END MATRIX FUNCTIONS */