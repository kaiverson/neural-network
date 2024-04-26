#include "neural_network.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* RANDOM */
double randn() {
    double u1 = rand() / (double)RAND_MAX;
    double u2 = rand() / (double)RAND_MAX;
    double randn = sqrtf(-2 * logf(u1)) * cosf(2 * 3.14159 * u2);

    if (randn > 1000 | randn < -1000) {
        printf("VERY LARGE RANDOM NUMBER!\n\tu1: %f\n\tu2: %f\n\t randn: %f\n");
    }

    return randn;
}
/* END RANDOM */



/* VECTOR FUNCTIONS */
Vector *vector_init(unsigned int rows, double *values) {
    Vector *vector;
    vector = vector_init_empty(rows);
    if (vector == NULL) {
        printf("vector_init error\n");
        return NULL;
    }

    for (unsigned int row = 0; row < rows; row++) {
        vector->values[row] = values[row];
    }

    return vector;
}


Vector *vector_init_randn(unsigned int rows) {
    Vector *vector;
    vector = vector_init_empty(rows);
    if (vector == NULL) {
        printf("vector_init_randn error\n");
        return NULL;
    }

    for (unsigned int row = 0; row < rows; row++) {
        vector->values[row] = randn();
    }

    return vector;
}


Vector *vector_init_empty(unsigned int rows) {
    Vector *vector;
    vector = malloc((sizeof (Vector)));
    if (vector == NULL) {
        return NULL;
    }


    vector->values = malloc((rows) * sizeof(double));
    if (vector->values == NULL) {
        return NULL;
    }

    vector->rows = rows;

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


bool vector_equal(Vector *a, Vector *b) {
    if (a->rows != b->rows) {
        return false;
    }

    for (unsigned int row = 0; row < a->rows; row++) {

        if (a->values[row] != b->values[row]) {
            return false;
        }

    }

    return true;
}
/* END VECTOR FUNCTIONS */




/* MATRIX FUNCTIONS */
Matrix *matrix_init(unsigned int rows, unsigned int cols, double **values);


Matrix *matrix_init_randn(unsigned int rows, unsigned int cols) {
    Matrix *matrix;
    matrix = matrix_init_empty(rows, cols);
    if (matrix == NULL) {
        printf("matrix_init failed\n");
        return NULL;
    }

    unsigned int row, col;
    for (row = 0; row < rows; row++) {

        for (col = 0; col < cols; col++) {

            matrix->values[row][col] = randn();

        }

    }

    return matrix;
}


Matrix *matrix_init_identity(unsigned int rows, unsigned int cols) {

    if (rows != cols) {
        printf("Identity rows (%d) != cols (%d)\n", rows, cols);
        return NULL;
    }


    Matrix *matrix;
    matrix = matrix_init_empty(rows, cols);
    if (matrix == NULL) {
        printf("matrix_init failed\n");
        return NULL;
    }

    unsigned int row, col;
    for (row = 0; row < rows; row++) {

        for (col = 0; col < cols; col++) {

            if (row == col) {
                matrix->values[row][col] = 1.0;
            } else {
                matrix->values[row][col] = 0.0;
            }

        }

    }

    return matrix;

}


Matrix *matrix_init_empty(unsigned int rows, unsigned int cols) {
    Matrix *matrix;
    matrix = malloc(sizeof (Matrix));
    if (matrix == NULL) {
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->values = malloc(rows * sizeof(double *));
    if (matrix == NULL) {
        free(matrix);
        return NULL;
    }

    unsigned int row;
    unsigned int col;
    for (row = 0; row < rows; row++) {

        matrix->values[row] = malloc(cols * sizeof(double));
        if (matrix->values[row] == NULL) {
            for (int i = 0; i < row; i++) {
                free(matrix->values[i]);
                free(matrix);
            }
            return NULL;
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


void matrix_print(const Matrix *matrix) {
    unsigned int row;
    unsigned int col;
    char *padding = "";
    printf("Matrix([");

    const int MAX_PRINT_ROWS = 20;
    const int MAX_PRINT_COLS = 20;
    unsigned int rows;
    unsigned int cols;
    char *end_chars = "";

    if (matrix->rows > MAX_PRINT_ROWS) {
        rows = MAX_PRINT_ROWS;
        end_chars = " ......";
    }

    if (matrix->cols > MAX_PRINT_ROWS) {
        cols = MAX_PRINT_COLS;
    }

    for (row = 0; row < rows - 1; row++) {
        printf("%s[", padding);
        padding = "        ";
        for (col = 0; col < cols - 1; col++) {
            printf("%.2f ", matrix->values[row][col]);
        }
        printf("%.2f%s],\n", matrix->values[row][col], end_chars);
    }

    printf("%s[", padding);
    for (col = 0; col < cols - 1; col++) {
        printf("%.2f ", matrix->values[row][col]);
    }
    printf("%.2f%s]])\n", matrix->values[row][col], end_chars);
}


Vector *matrix_times_vector(const Matrix *matrix, const Vector *input, Vector *output) {
    if (matrix->cols != input->rows) {
        printf("Error: matrix cols (%d) != vector input rows (%d)\n", matrix->cols, input->rows);
        return NULL;
    }

    if (matrix->rows != output->rows) {
        printf("Error: matrix rows (%d) != vector output rows (%d)\n", matrix->rows, output->rows);
        return NULL;
    }

    double dot_product;
    double product;
    unsigned int mat_row;
    unsigned int mat_col;
    for (mat_row = 0; mat_row < matrix->rows; mat_row++) {

        dot_product = 0;
        for (mat_col = 0; mat_col < matrix->cols; mat_col++) {

            product = matrix->values[mat_row][mat_col] * input->values[mat_col];
            dot_product += product;

            if (product > 1000 | product < -1000) {

                printf("Very large product: %.5f times %.5f == %.5f\n", matrix->values[mat_row][mat_col], input->values[mat_col]);
            }

        }
        
        output->values[mat_row] = dot_product;

    }

    return output;
}
/* END MATRIX FUNCTIONS */