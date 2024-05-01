#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>

/* BUILD UR NEURAL NETWORK HERE! */
typedef struct {
    LayerDense *dense1;
    Vector *buffer1;
    LayerDense *dense2;
    Vector *buffer2;
    LayerDense *dense3;
    Vector *buffer3;
} NeuralNetwork;

NeuralNetwork *nn_init() {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    
    nn->dense1 = nn_dense_init_randn(784, 64);  // Input size: 28x28x1 = 784
    nn->buffer1 = vector_init_empty(64);
    nn->dense2 = nn_dense_init_randn(64, 16);
    nn->buffer2 = vector_init_empty(16);
    nn->dense3 = nn_dense_init_randn(16, 10);     // Output size: 10 classes
    nn->buffer3 = vector_init_empty(10);
    
    return nn;
}

void nn_free(NeuralNetwork *nn) {
    nn_dense_free(nn->dense1);
    vector_free(nn->buffer1);
    nn_dense_free(nn->dense2);
    vector_free(nn->buffer2);
    nn_dense_free(nn->dense3);
    vector_free(nn->buffer3);
    free(nn);
}

Vector *nn_forward(NeuralNetwork *nn, Vector *x) {
    nn->buffer1 = nn_dense_forward(nn->dense1, x, nn->buffer1);
    nn->buffer1 = nn_relu(nn->buffer1);

    nn->buffer2 = nn_dense_forward(nn->dense2, nn->buffer1, nn->buffer2);
    nn->buffer2 = nn_relu(nn->buffer2);
    
    nn->buffer3 = nn_dense_forward(nn->dense3, nn->buffer2, nn->buffer3);
    nn->buffer3 = nn_softmax(nn->buffer3);

    return nn->buffer3;
}




int main(int argc, char **argv) {
    srand(6);


    Vector *x = vector_init_randn(784);
    NeuralNetwork *nn = nn_init();

    Vector *y_pred = nn_forward(nn, x);

    printf("y_pred: ");
    vector_print(y_pred);

    Vector *y = vector_init_zero(10);
    
    unsigned int index;
    for (index = 0; index < y->rows; index++) {

        y->values[index] = 1.0;
        
        printf("Loss when digit is %d: %f\n", index, nn_cross_entropy_loss(y_pred, y));

        vector_set_zero(y);
    }

    printf("NN predicted the digit: %d\n", vector_argmax(y_pred));


    Matrix *matrix = matrix_init_randn(3, 5);
    matrix_print(matrix);

    Vector *vector = vector_init_empty(5);
    vector_print(vector);

    vector = matrix_get_row(matrix, 2, vector);
    vector_print(vector);

    char *test = "123";
    printf("%f\n", strtof(test, NULL));

    Matrix *mnist_images = matrix_init_empty(60001, 784);
    mnist_images = mnist_load_images(mnist_images, "mnist_train.csv");

    Vector *image_buffer = vector_init_empty(28 * 28);
    Vector *mnist_labels = vector_init_empty(60001);
    mnist_labels = mnist_load_labels(mnist_labels, "mnist_train.csv");

    int accuray = 0;

    for (int i = 0; i < 60001; i++) {
        // printf("NN prediction: %d\n", vector_argmax(nn_forward(nn, matrix_get_row(mnist_images, i, image_buffer))));
        // printf("  Image label: %d", (int) mnist_labels->values[i]);
        // mnist_image_print(mnist_images, i, image_buffer);

        if (vector_argmax(nn_forward(nn, matrix_get_row(mnist_images, i, image_buffer))) == (int) mnist_labels->values[i]) {
            accuray++;
        }

    }

    printf("%d / 60001 correct predictions\n%lf%%", accuray, (double) accuray / (double) 60001);

    int prediction;
    int label;
    for (int i = 0; i < 100; i++) {
        label = mnist_labels->values[i];

        if (label != 4) {
            continue;
        }


        prediction = vector_argmax(nn_forward(nn, matrix_get_row(mnist_images, i, image_buffer)));
        printf("NN prediction: %d\n", prediction);
        printf("  Image label: %d", (int) mnist_labels->values[i]);
        mnist_image_print(mnist_images, i, image_buffer);

    }


    free(nn);

    return 0;
}