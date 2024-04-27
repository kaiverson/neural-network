#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>

/* BUILD UR NEURAL NETWORK HERE! */
typedef struct {
    LayerDense *dense1;
    Vector *buffer1;
    LayerDense *dense2;
    Vector *buffer2;
} NeuralNetwork;

NeuralNetwork *nn_init() {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    
    nn->dense1 = nn_layer_dense_init(3072, 512);  // Input size: 32x32x3 = 3072
    nn->buffer1 = vector_init_empty(512);
    nn->dense2 = nn_layer_dense_init(512, 10);     // Output size: 10 classes
    nn->buffer2 = vector_init_empty(10);
    
    return nn;
}

void nn_free(NeuralNetwork *nn) {
    nn_layer_dense_free(nn->dense1);
    vector_free(nn->buffer1);
    nn_layer_dense_free(nn->dense2);
    vector_free(nn->buffer2);
    free(nn);
}

Vector *nn_forward(NeuralNetwork *nn, Vector *x) {
    nn->buffer1 = nn_layer_dense_forward(nn->dense1, x, nn->buffer1);
    nn->buffer1 = nn_layer_relu(nn->buffer1);

    nn->buffer2 = nn_layer_dense_forward(nn->dense2, nn->buffer1, nn->buffer2);
    
    vector_print(nn->buffer2);
    
    nn->buffer2 = nn_layer_sigmoid(nn->buffer2);

    return nn->buffer2;
}




int main(int argc, char **argv) {

    Vector *x = vector_init_randn(3072);
    NeuralNetwork *nn = nn_init();

    Vector *y_pred = nn_forward(nn, x);
    vector_print(y_pred);

    return 0;
}