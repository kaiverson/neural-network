#include "neural_network.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* NN_LAYER_DENSE FUNCTIONS */
LayerDense *nn_layer_dense_init(int size_input, int size_output) {
    LayerDense *layer;
    layer = malloc(sizeof(LayerDense));
    if (layer == NULL) {
        return NULL;
    }

    layer->size_input = size_input + 1;
    layer->size_output = size_output;
    layer->parameters = matrix_init(size_output, size_input + 1);
    if (layer->parameters == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        free(layer);
    }

    return layer;
}


void nn_layer_dense_free(LayerDense *layer) {
    matrix_free(layer->parameters);
    free(layer);
}


void nn_layer_dense_print(LayerDense *layer) {
    matrix_print(layer->parameters);
}


Vector *nn_layer_dense_forward(LayerDense *layer, Vector *x) {
    Vector *result = matrix_times_vector(layer->parameters, x);
    return result;
}


Vector *nn_layer_dense_backwards(LayerDense *layer, Vector *x);

LayerDense *nn_layer_dense_load(char *file);
void       *nn_layer_dense_save(LayerDense *layer, char *file);
/* END NN_LAYER_DENSE_FUNCTIONS */



/* ACTIVATION FUNCTIONS */
Vector *nn_layer_relu(Vector *x) {
    unsigned int i;
    for (i = 0; i < x->rows - 1; i++) {
        if (x->values[i] < 0) {
            x->values[i] = 0;
        }
    }

    return x;
}


Vector *nn_layer_leaky_relu(Vector *x) {
    unsigned int i;
    for (i = 0; i < x->rows - 1; i++) {
        if (x->values[i] < 0) {
            x->values[i] *= 0.1;
        }
    }

    return x;
}


Vector *nn_layer_softmax(Vector *x) {
    unsigned int row;
    double exponential_sum = 0;
    double exponential;

    for (row = 0; row < x->rows - 1; row++) {
        exponential = exp(x->values[row]);
        exponential_sum += exponential;
        x->values[row] = exponential;
    }

    for (row = 0; row < x->rows - 1; row++) {
        x->values[row] /= exponential_sum;
    }

    return x;
}
/* END ACTIVATION FUNCTIONS */