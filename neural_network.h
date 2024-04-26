#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

float randn();

typedef struct {
    unsigned int rows;
    float *values;
} Vector;

Vector *vector_init(unsigned int rows, float *values);
void vector_free(Vector *vector);
void vector_print(Vector *vector);
Vector *vector_append(Vector *vector, float value);


typedef struct {
    unsigned int rows;
    unsigned int cols;
    float **values;
} Matrix;

Matrix *matrix_init(unsigned int rows, unsigned int cols);
void matrix_free(Matrix *matrix);
void matrix_print(Matrix *matrix);
Vector *matrix_times_vector(Matrix *matrix, Vector *vector);


typedef struct {
    int    size_input;
    int    size_output;
    Matrix *parameters;
} LayerDense;

LayerDense *nn_layer_dense_init(int size_input, int size_output);
void        nn_layer_dense_free(LayerDense *layer);
void        nn_layer_dense_print(LayerDense *layer);
Vector     *nn_layer_dense_forward(LayerDense *layer, Vector *x);
Vector     *nn_layer_dense_backwards(LayerDense *layer, Vector *x);
LayerDense *nn_layer_dense_load(char *file);
void       *nn_layer_dense_save(LayerDense *layer, char *file);


Vector *nn_layer_relu(Vector *x);
Vector *nn_layer_leaky_relu(Vector *x);
Vector *nn_layer_softmax(Vector *x);


#endif /* __NEURAL_NETWORK_H__ */