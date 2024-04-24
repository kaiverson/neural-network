#include "neural_network.h"

#include <stdio.h>

/* BUILD UR NEURAL NETWORK HERE! */
typedef struct {
    LayerDense *layer1;

    
} MyNeuralNetwork;




int main(int argc, char **argv) {
    float values[] = {randn(), randn(), randn(), randn(), randn()};
    Vector *x1 = vector_init(5, values);

    printf("softmax of: \n");
    vector_print(x1);
    printf("is: \n");
    
    LayerDense *layer1 = nn_layer_dense_init(5, 5);
    LayerDense *layer2 = nn_layer_dense_init(5, 3);


    nn_layer_dense_print(layer1);
    vector_print(x1);
    printf("\n\n");


    Vector *x2 = nn_layer_dense_forward(layer1, x1);
    x2 = nn_layer_leaky_relu(x2);


    nn_layer_dense_print(layer2);
    vector_print(x2);
    printf("\n\n");

    Vector *x3 = nn_layer_dense_forward(layer2, x2);
    x3 = nn_layer_softmax(x3);

    vector_print(x3);

    

    nn_layer_dense_free(layer1);
    nn_layer_dense_free(layer2);
    vector_free(x1);
    vector_free(x2);
    vector_free(x3);

    return 0;
}