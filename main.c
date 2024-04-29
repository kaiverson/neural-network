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
    
    nn->dense1 = nn_dense_init_randn(3072, 512);  // Input size: 32x32x3 = 3072
    nn->buffer1 = vector_init_empty(512);
    nn->dense2 = nn_dense_init_randn(512, 10);     // Output size: 10 classes
    nn->buffer2 = vector_init_empty(10);
    
    return nn;
}

void nn_free(NeuralNetwork *nn) {
    nn_dense_free(nn->dense1);
    vector_free(nn->buffer1);
    nn_dense_free(nn->dense2);
    vector_free(nn->buffer2);
    free(nn);
}

Vector *nn_forward(NeuralNetwork *nn, Vector *x) {
    nn->buffer1 = nn_dense_forward(nn->dense1, x, nn->buffer1);
    nn->buffer1 = nn_relu(nn->buffer1);

    nn->buffer2 = nn_dense_forward(nn->dense2, nn->buffer1, nn->buffer2);
    nn->buffer2 = nn_softmax(nn->buffer2);

    return nn->buffer2;
}




int main(int argc, char **argv) {

    Vector *x = vector_init_randn(3072);
    NeuralNetwork *nn = nn_init();

    Vector *y_pred = nn_forward(nn, x);

    printf("y_pred: ");
    vector_print(y_pred);

    Vector *y = vector_init_zero(10);
    
    unsigned int index;
    for (index = 0; index < y->rows; index++) {

        y->values[index] = 1.0;
        
        printf("Loss when y->values[%d] == 1: %f\n", index, nn_cross_entropy_loss(y_pred, y));

        vector_set_zero(y);
    }



    free(nn);

    return 0;
}