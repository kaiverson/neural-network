#include "neural_network.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* NN_LAYER_DENSE FUNCTIONS */
LayerDense *nn_layer_dense_init_randn(int size_input, int size_output) {
    LayerDense *layer;
    layer = nn_layer_dense_init_empty(size_input, size_output);
    if (layer == NULL) {
        return NULL;
    }

    layer->parameters = matrix_set_randn(layer->parameters);

    return layer;
}


LayerDense *nn_layer_dense_init_empty(int size_input, int size_output) {
    LayerDense *layer;
    layer = malloc(sizeof(LayerDense));
    if (layer == NULL) {
        return NULL;
    }

    layer->size_input = size_input;
    layer->size_output = size_output;
    layer->parameters = matrix_init_empty(size_output, size_input);
    if (layer->parameters == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        free(layer);
        return NULL;
    }

    layer->dL_db = vector_init_zero(size_output);
    if (layer->dL_db == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        matrix_free(layer->parameters);
        free(layer);
        return NULL;
    }

    layer->dL_dW = matrix_init_zero(size_output, size_input);
    if (layer->dL_dW == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        vector_free(layer->dL_db);
        matrix_free(layer->parameters);
        free(layer);
        return NULL;
    }

    return layer;
}


void nn_layer_dense_free(LayerDense *layer) {
    matrix_free(layer->dL_dW);
    vector_free(layer->dL_db);
    matrix_free(layer->parameters);
    free(layer);
}


void nn_layer_dense_print(LayerDense *layer) {
    matrix_print(layer->parameters);
}


Vector *nn_layer_dense_forward(LayerDense *layer, Vector *x_in, Vector *x_out) {
    x_out = matrix_times_vector(layer->parameters, x_in, x_out);
    return x_out;
}


void *nn_layer_dense_zero_grad(LayerDense *layer);


LayerDense *nn_layer_dense_load(char *file);
void       *nn_layer_dense_save(LayerDense *layer, char *file);
/* END NN_LAYER_DENSE_FUNCTIONS */



/* ACTIVATION FUNCTIONS */
Vector *nn_layer_relu(Vector *x) {
    unsigned int i;
    for (i = 0; i < x->rows; i++) {
        if (x->values[i] < 0) {
            x->values[i] = 0;
        }
    }

    return x;
}


Vector *nn_layer_leaky_relu(Vector *x) {
    unsigned int i;
    for (i = 0; i < x->rows; i++) {
        if (x->values[i] < 0) {
            x->values[i] *= 0.1;
        }
    }

    return x;
}


Vector *nn_layer_sigmoid(Vector *x) {
    unsigned int row;
    double exponential = 0;

    for (row = 0; row < x->rows; row++) {
        x->values[row] = 1 / (1 + exp(-x->values[row]));
    }

    return x;
}


Vector *nn_layer_softmax(Vector *x) {
    unsigned int row;
    double exponential_sum = 0;
    double exponential;

    for (row = 0; row < x->rows; row++) {
        exponential = exp(x->values[row]);
        exponential_sum += exponential;
        x->values[row] = exponential;
    }

    for (row = 0; row < x->rows; row++) {
        x->values[row] /= exponential_sum;
    }

    return x;
}
/* END ACTIVATION FUNCTIONS */