#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>

/* BUILD UR NEURAL NETWORK HERE! */
typedef struct {
    LayerDense *dense1;
    LayerDense *dense2;
    LayerDense *dense3;
} NeuralNetwork;

NeuralNetwork *nn_init() {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    
    nn->dense1 = nn_dense_init_randn(784, 64);  // Input size: 28x28x1 = 784
    nn->dense2 = nn_dense_init_randn(64, 16);
    nn->dense3 = nn_dense_init_randn(16, 10);     // Output size: 10 classes
    
    return nn;
}

void nn_free(NeuralNetwork *nn) {
    nn_dense_free(nn->dense1);
    nn_dense_free(nn->dense2);
    nn_dense_free(nn->dense3);
    free(nn);
}

Vector *nn_forward(NeuralNetwork *nn, Vector *x) {
    nn->dense1->buffer = nn_dense_forward(nn->dense1, x);
    nn->dense1->buffer = nn_relu(nn->dense1->buffer);

    nn->dense2->buffer = nn_dense_forward(nn->dense2, nn->dense1->buffer);
    nn->dense2->buffer = nn_relu(nn->dense2->buffer);
    
    nn->dense3->buffer = nn_dense_forward(nn->dense3, nn->dense2->buffer);
    nn->dense3->buffer = nn_softmax(nn->dense3->buffer);

    return nn->dense3->buffer;
}




int main(int argc, char **argv) {
    #define TRAIN_SIZE 60000

    srand(6);

    Matrix *mnist_images = matrix_init_empty(TRAIN_SIZE, 784);
    mnist_images = mnist_load_images(mnist_images, "mnist_train.csv");

    Vector *image_buffer = vector_init_empty(28 * 28);
    Vector *mnist_labels = vector_init_empty(TRAIN_SIZE);
    mnist_labels = mnist_load_labels(mnist_labels, "mnist_train.csv");

    NeuralNetwork *nn = nn_init();

    int accuray = 0;
    for (int i = 0; i < TRAIN_SIZE; i++) {
        // printf("NN prediction: %d\n", vector_argmax(nn_forward(nn, matrix_get_row(mnist_images, i, image_buffer))));
        // printf("  Image label: %d", (int) mnist_labels->values[i]);
        // mnist_image_print(mnist_images, i, image_buffer);

        if (vector_argmax(nn_forward(nn, matrix_get_row(mnist_images, i, image_buffer))) == (int) mnist_labels->values[i]) {
            accuray++;
        }

    }
    printf("%d / %d correct predictions\n%.2lf%%\n\n", accuray, TRAIN_SIZE, 100 * (double) accuray / (double) 60000);

    int prediction;
    int label;
    int row = 0;
    for (int digit = 0; digit < 10; digit++) {
        do {
            row++;
            label = mnist_labels->values[row];
        } while (label != digit);

        prediction = vector_argmax(nn_forward(nn, matrix_get_row(mnist_images, row, image_buffer)));
        printf("NN prediction: %d\n", prediction);
        printf("  Image label: %d\n", (int) mnist_labels->values[row]);
        mnist_image_print(mnist_images, row, image_buffer);

    }

    Vector *a_buf = vector_init_randn(8);
    Vector *d_val = vector_init_randn(8);

    vector_print(d_val);
    d_val = nn_relu_backwards(a_buf, d_val);
    vector_print(d_val);


    free(mnist_images);
    free(nn);

    return 0;
}