#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <stdbool.h>

double randn();

typedef struct {
    unsigned int rows;
    double *values;
} Vector;

Vector *vector_init(unsigned int rows, double *values);
Vector *vector_init_randn(unsigned int rows);
Vector *vector_init_empty(unsigned int rows);
void    vector_free(Vector *vector);
void    vector_print(Vector *vector);
bool    vector_equal(Vector *a, Vector *b);


typedef struct {
    unsigned int rows;
    unsigned int cols;
    double **values;
} Matrix;

Matrix *matrix_init(unsigned int rows, unsigned int cols, double **values);    /* Creates a matrix and fills it with the provided values. */
Matrix *matrix_init_randn(unsigned int rows, unsigned int cols);    /* Creates a matrix and fills it with random values following the normal distribution. */
Matrix *matrix_init_identity(unsigned int rows, unsigned int cols);    /* Creates an identity matrix. */
Matrix *matrix_init_empty(unsigned int rows, unsigned int cols);    /* Creates a matrix with uninitialized values. */
void    matrix_free(Matrix *matrix);    /* Frees the memory of a matrix. */
void    matrix_print(const Matrix *matrix);    /* Prints the values of a matrix. */
Vector *matrix_times_vector(const Matrix *matrix, const Vector *input, Vector *output);    /* Multiplies a matrix with a vector. */


typedef struct {
    int    size_input;
    int    size_output;
    Matrix *parameters;
} LayerDense;

LayerDense *nn_layer_dense_init(int size_input, int size_output);
void        nn_layer_dense_free(LayerDense *layer);
void        nn_layer_dense_print(LayerDense *layer);
Vector     *nn_layer_dense_forward(LayerDense *layer, Vector *x_int, Vector *x_out);
Vector     *nn_layer_dense_backwards(LayerDense *layer, Vector *x);
LayerDense *nn_layer_dense_load(char *file);
void       *nn_layer_dense_save(LayerDense *layer, char *file);


Vector *nn_layer_relu(Vector *x);
Vector *nn_layer_leaky_relu(Vector *x);
Vector *nn_layer_softmax(Vector *x);


#endif /* __NEURAL_NETWORK_H__ */