#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <stdbool.h>

double randn();    /* Returns random numbers with the normal distribution. Name inspired by numpy! */

typedef struct {
    unsigned int rows;
    double *values;
} Vector;

Vector *vector_init(unsigned int rows, double *values);   /* Creates a vector and fills it with the provided values. */
Vector *vector_init_empty(unsigned int rows);    /* Creates a vector with uninitialized values. */
Vector *vector_init_randn(unsigned int rows);    /* Creates a vector and fills it with random values following the normal distribution. */
Vector *vector_init_zero(unsigned int rows);    /* Creates a vector and fills it with zeros. */
Vector *vector_set_randn(Vector *vector);
Vector *vector_set_zero(Vector *vector);
void    vector_free(Vector *vector);    /* Frees the memory of a vector. */
void    vector_print(const Vector *vector);    /* Prints the values of a vector. */
bool    vector_equal(const Vector *a, const Vector *b);    /* Determines the equality of two vectors. */


typedef struct {
    unsigned int rows;
    unsigned int cols;
    double **values;
} Matrix;

Matrix *matrix_init(unsigned int rows, unsigned int cols, double **values);    /* Creates a matrix and fills it with the provided values. */
Matrix *matrix_init_empty(unsigned int rows, unsigned int cols);    /* Creates a matrix with uninitialized values. */
Matrix *matrix_init_randn(unsigned int rows, unsigned int cols);    /* Creates a matrix and fills it with random values following the normal distribution. */
Matrix *matrix_init_identity(unsigned int rows, unsigned int cols);    /* Creates an identity matrix. */
Matrix *matrix_init_zero(unsigned int rows, unsigned int cols);    /* Creates a mtrix and fills it with zeros. */
Matrix *matrix_set_randn(Matrix *matrix);    /* Set all of the values of a matrix to randn values. */
Matrix *matrix_set_zero(Matrix *matrix);    /* Set all of the values of a matrix to zero. */
void    matrix_free(Matrix *matrix);    /* Frees the memory of a matrix. */
void    matrix_print(const Matrix *matrix);    /* Prints the values of a matrix. */
Vector *matrix_times_vector(const Matrix *matrix, const Vector *input, Vector *output);    /* Multiplies a matrix with a vector. Note that you provide the memory this uses for the output. */


typedef struct {
    int size_input;
    int size_output;
    Matrix *parameters;  
    Vector *x_input;     
    Matrix *dL_dW;       
    Vector *dL_db;
} LayerDense;

LayerDense *nn_layer_dense_init_randn(int size_input, int size_output);    /* Creates a dense nn layer and fills it with randn parameters. */
LayerDense *nn_layer_dense_init_empty(int size_input, int size_output);    /* Creates a dense nn layer with uninitialized parameters. */
void        nn_layer_dense_free(LayerDense *layer);    /* Frees the memory of a dense nn layer. */
void        nn_layer_dense_print(LayerDense *layer);    /* Prints the contents of a dense nn layer. */
Vector     *nn_layer_dense_forward(LayerDense *layer, Vector *x_int, Vector *x_out);    /* Computes the forward pass of a dense nn layer. Note that you provide the memory this uses for the output. */
Vector     *nn_layer_dense_backwards(LayerDense *layer, Vector *x);    /* Computes the backwards pass of a dense nn layer. */
void       *nn_layer_dense_zero_gradients(LayerDense *layer);    /* Sets the gradients to zero. */
LayerDense *nn_layer_dense_load(char *file);    /* Loads the nn parameters from a file and insertes them into the dense nn layer. */
void       *nn_layer_dense_save(LayerDense *layer, char *file);    /* Saves the dense nn layer parameters and insertes them into a file. */


Vector *nn_layer_relu(Vector *x);
Vector *nn_layer_leaky_relu(Vector *x);
Vector *nn_layer_sigmoid(Vector *x);
Vector *nn_layer_softmax(Vector *x);


#endif /* __NEURAL_NETWORK_H__ */