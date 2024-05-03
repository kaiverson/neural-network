#include "neural_network.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>


/* RANDOM */
double randn(double mean, double variance) {
    double u1, u2;
    do {
        u1 = rand() / (double)RAND_MAX;
        u2 = rand() / (double)RAND_MAX;
    } while (u1 == 0 | u2 == 0);

    double randn = sqrtf(-2 * logf(u1)) * cosf(2 * 3.14159 * u2);

    return randn * sqrt(variance) + mean;
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


Vector *vector_init_randn(unsigned int rows) {
    Vector *vector;
    vector = vector_init_empty(rows);
    if (vector == NULL) {
        printf("vector_init_randn error\n");
        return NULL;
    }

    vector = vector_set_randn(vector);

    return vector;
}


Vector *vector_init_zero(unsigned int rows) {
    Vector *vector;
    vector = vector_init_empty(rows);
    if (vector == NULL) {
        printf("vector_init_zero error\n");
        return NULL;
    }

    vector = vector_set_zero(vector);

    return vector;
}


Vector *vector_set_randn(Vector *vector) {
    if (vector == NULL) {
        printf("NULLPTR\n");
        return NULL;
    }

    unsigned int row;
    for (row = 0; row < vector->rows; row++) {

        vector->values[row] = randn(0, sqrt(2.0 / (double) vector->rows));

    }

    return vector;
}


Vector *vector_set_zero(Vector *vector) {
    if (vector == NULL) {
        printf("NULLPTR\n");
        return NULL;
    }

    unsigned int row;
    for (row = 0; row < vector->rows; row++) {

        vector->values[row] = 0;
    }

    return vector;
}


void vector_free(Vector *vector) {
    free(vector->values);
    free(vector);
}


void vector_print(const Vector *vector) {
    printf("Vector([");
    unsigned int i;
    for (i = 0; i < vector->rows - 1; i++) {
        printf("%.2f ", vector->values[i]);
    }
    printf("%.2f])\n", vector->values[vector->rows - 1]);
}


bool vector_equal(const Vector *a, const Vector *b) {
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


int vector_argmax(const Vector *vector) {
    double max_value = vector->values[0];
    int max_value_index = 0;

    unsigned int index;
    for (index = 0; index < vector->rows; index++) {

        if (vector->values[index] > max_value) {
            max_value = vector->values[index];
            max_value_index = index;
        }

    }

    return max_value_index;
}
/* END VECTOR FUNCTIONS */




/* MATRIX FUNCTIONS */
Matrix *matrix_init(unsigned int rows, unsigned int cols, double **values);


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


Matrix *matrix_init_randn(unsigned int rows, unsigned int cols) {
    Matrix *matrix;
    matrix = matrix_init_empty(rows, cols);
    if (matrix == NULL) {
        printf("matrix_init failed\n");
        return NULL;
    }

    matrix = matrix_set_randn(matrix);

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


Matrix *matrix_init_zero(unsigned int rows, unsigned int cols) {
    Matrix *matrix;
    matrix = matrix_init_empty(rows, cols);
    if (matrix == NULL) {
        printf("MATRIX COULD NOT BE CREATED\n");
        return NULL;
    }

    matrix = matrix_set_zero(matrix);

    return matrix;
}


Matrix *matrix_set_randn(Matrix *matrix) {
    if (matrix == NULL) {
        printf("MATRIX RANDN RECIEVED NULL\n");
        return NULL;
    }

    unsigned int row, col;
    for (row = 0; row < matrix->rows; row++) {

        for (col = 0; col < matrix->cols; col++) {

            matrix->values[row][col] = randn(0, sqrt(2.0 / (double) matrix->cols));

        }

    }

    return matrix;
}


Matrix *matrix_set_zero(Matrix *matrix) {
    if (matrix == NULL) {
        printf("MATRIX ZERO RECIEVED NULLPTR\n");
        return NULL;
    }

    unsigned int row, col;
    for (row = 0; row < matrix->rows; row++) {

        for (col = 0; col < matrix->cols; col++) {

            matrix->values[row][col] = 0;

        }

    }

    return matrix;
}


Vector *matrix_get_row(const Matrix *matrix, const unsigned int row_index, Vector *row) {
    if (matrix == NULL | row_index >= matrix->rows | row->rows != matrix->cols) {
        printf("MATRIX_GET_ROW ERROR\n");
        return NULL;
    }

    unsigned int col_index;
    for (col_index = 0; col_index < matrix->cols; col_index++) {

        row->values[col_index] = matrix->values[row_index][col_index];

    }

    return row;
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
    unsigned int rows = matrix->rows;
    unsigned int cols = matrix->cols;
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

        }
        
        output->values[mat_row] = dot_product;

    }

    return output;
}


Vector *matrix_times_vector_plus_vector(const Matrix *matrix, const Vector *input, Vector *output) {
    if (matrix->cols != input->rows + 1) {
        printf("Error: matrix cols (%d) != vector input rows (%d) + 1\n", matrix->cols, input->rows);
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

            if (mat_col == matrix->cols - 1) {
                product = matrix->values[mat_row][mat_col];
            } else {
                product = matrix->values[mat_row][mat_col] * input->values[mat_col];
            }

            dot_product += product;

        }
        
        output->values[mat_row] = dot_product;

    }

    return output;
}
/* END MATRIX FUNCTIONS */




/* DATA LOADING FUNCTIONS */
Vector *mnist_load_labels(Vector *labels, char *file_name) {
    clock_t t; 
    t = clock();

    printf("LOADING MNIST LABELS FROM %s .....\n", file_name);

    FILE *file_pointer;
    char label_buffer[3500];
    char *token;

    file_pointer = fopen(file_name, "r");  

    fgets(label_buffer, 3500, file_pointer);
    fgets(label_buffer, 3500, file_pointer);

    int image = 0;
    int pixel = 0;
    while (feof(file_pointer) != true & labels->rows > image) {
        fgets(label_buffer, 3500, file_pointer);

        token = strtok(label_buffer, ",");
        labels->values[image] = strtod(token, NULL);

        image++;
    }

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("%d MNIST LABELS LOADED IN %.3f SECONDS!\n\n", image, time_taken);

    return labels;
}


Matrix *mnist_load_images(Matrix *images, char *file_name) {
    clock_t t; 
    t = clock(); 

    printf("LOADING MNIST IMAGES FROM %s .....\n", file_name);

    FILE *file_pointer;
    char image_buffer[3500];
    char *token;

    file_pointer = fopen(file_name, "r");  

    fgets(image_buffer, 3500, file_pointer);
    fgets(image_buffer, 3500, file_pointer);

    int image = 0;
    int pixel = 0;
    while (feof(file_pointer) != true & images->rows > image) {
        fgets(image_buffer, 3500, file_pointer);

        token = strtok(image_buffer, ",");
        token = strtok(NULL, ",");

        for (pixel = 0; pixel < 784; pixel++) {

            images->values[image][pixel] = strtod(token, NULL) / 256.0;
            token = strtok(NULL, ",");

        }

        image++;
    }

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("%d MNIST IMAGES LOADED IN %.3f SECONDS!\n\n", image, time_taken);

    return images;
}


void   *mnist_image_print(const Matrix *images, const unsigned int image_row, Vector *image_buffer) {
    if (images == NULL | image_row > images->rows) {
        printf("COULD NOT PRINT NMIST IMAGE: INVALID IMAGE\n");
    }

    image_buffer = matrix_get_row(images, image_row, image_buffer);

    unsigned int index;

    printf("\n");
    for (index = 0; index < 30; index++) {
        printf("%%%%%%");
    }
    printf("\n");

    for (index = 0; index < 28*28; index++) {

        if (index % 28 == 0) {
            printf("%%%%%%");
        }

        if (image_buffer->values[index] == 0) {
            printf(" . ");
        } else if (image_buffer->values[index] < 50.0 / 256.0) {
            printf("---");
        } else if (image_buffer->values[index] < 150.0 / 256.0) {
            printf("+++");
        } else {
            printf("###");
        }

        if (index % 28 == 27) {
            printf("%%%%%%\n");
        }
    }


    for (index = 0; index < 30; index++) {
        printf("%%%%%%");
    }
    printf("\n\n");

}
/* END DATA LOADING FUNCTION */