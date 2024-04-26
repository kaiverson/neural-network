#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>

/* BUILD UR NEURAL NETWORK HERE! */
typedef struct {
    LayerDense *dense1;
    LayerDense *dense2;
} NeuralNetwork;

NeuralNetwork *nn_init() {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    
    // Initialize layers
    nn->dense1 = nn_layer_dense_init(3072, 512);  // Input size: 32x32x3 = 3072
    nn->dense2 = nn_layer_dense_init(512, 10);     // Output size: 10 classes
    
    return nn;
}

void nn_free(NeuralNetwork *nn) {
    nn_layer_dense_free(nn->dense1);
    nn_layer_dense_free(nn->dense2);
    free(nn);
}

Vector *nn_forward(NeuralNetwork *nn, Vector *x) {
    // Forward pass through the first dense layer
    nn_layer_dense_print(nn->dense1);
    Vector *dense1_output = nn_layer_dense_forward(nn->dense1, x);
    printf("\n\nDENSE1 OUTPUT\n");
    vector_print(dense1_output);
    
    // Apply ReLU activation function
    Vector *relu_output = nn_layer_relu(dense1_output);
    printf("\n\nRELU OUTPUT\n");
    vector_print(relu_output);
    
    // Forward pass through the second dense layer
    Vector *dense2_output = nn_layer_dense_forward(nn->dense2, relu_output);
    vector_print(dense2_output);
    
    // Apply softmax activation function
    Vector *softmax_output = nn_layer_softmax(dense2_output);
    vector_print(softmax_output);
    
    // Free intermediate outputs
    vector_free(dense1_output);
    vector_free(relu_output);
    vector_free(dense2_output);
    
    return softmax_output;
}




int main(int argc, char **argv) {
    float values[3072];
    for (int i = 0; i < 3072; i++) {
        values[i] = randn();
    }

    Vector *x = vector_init(3072, values);
    NeuralNetwork *nn = nn_init();


    Vector *y_pred = nn_forward(nn, x);
    vector_print(y_pred);



    return 0;
}