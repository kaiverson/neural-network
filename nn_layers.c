#include "neural_network.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* NN_DENSE FUNCTIONS */
LayerDense *nn_dense_init_randn(int size_input, int size_output) {
    LayerDense *layer;
    layer = nn_dense_init_empty(size_input, size_output);
    if (layer == NULL) {
        return NULL;
    }

    layer->parameters = matrix_set_randn(layer->parameters);

    return layer;
}


LayerDense *nn_dense_init_empty(int size_input, int size_output) {
    LayerDense *layer;
    layer = malloc(sizeof(LayerDense));
    if (layer == NULL) {
        return NULL;
    }

    layer->size_input = size_input;
    layer->size_output = size_output;
    layer->parameters = matrix_init_empty(size_output, size_input + 1);
    if (layer->parameters == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        free(layer);
        return NULL;
    }

    layer->buffer = vector_init_empty(size_output);
    if (layer->parameters == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        matrix_free(layer->parameters);
        free(layer);
    }

    /*
    layer->dL_db = vector_init_zero(size_output);
    if (layer->dL_db == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        matrix_free(layer->parameters);
        free(layer);
        return NULL;
    }
    */

    layer->dL_dW = matrix_init_zero(size_output, size_input + 1);
    if (layer->dL_dW == NULL) {
        printf("LAYER (DENSE) COULD NOT BE CREATED\n");
        // vector_free(layer->dL_db);
        matrix_free(layer->parameters);
        vector_free(layer->buffer);
        free(layer);
        return NULL;
    }

    return layer;
}


void nn_dense_free(LayerDense *layer) {
    matrix_free(layer->dL_dW);
    // vector_free(layer->dL_db);
    matrix_free(layer->parameters);
    vector_free(layer->buffer);
    free(layer);
}


void nn_dense_print(LayerDense *layer) {
    matrix_print(layer->parameters);
}


Vector *nn_dense_forward(LayerDense *layer, Vector *x_in) {
    layer->buffer = matrix_times_vector_plus_vector(layer->parameters, x_in, layer->buffer);
    return layer->buffer;
}


void *nn_dense_zero_grad(LayerDense *layer);


LayerDense *nn_dense_load(char *file);
void       *nn_dense_save(LayerDense *layer, char *file);
/* END NN_DENSE_FUNCTIONS */



/* ACTIVATION FUNCTIONS */
Vector *nn_relu(Vector *x) {
    unsigned int i;
    for (i = 0; i < x->rows; i++) {
        if (x->values[i] < 0) {
            x->values[i] = 0;
        }
    }

    return x;
}


Vector *nn_leaky_relu(Vector *x) {
    unsigned int i;
    for (i = 0; i < x->rows; i++) {
        if (x->values[i] < 0) {
            x->values[i] *= 0.1;
        }
    }

    return x;
}


Vector *nn_sigmoid(Vector *x) {
    unsigned int row;
    double exponential = 0;

    for (row = 0; row < x->rows; row++) {
        x->values[row] = 1 / (1 + exp(-x->values[row]));
    }

    return x;
}


Vector *nn_softmax(Vector *x) {
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


Vector *nn_relu_backwards(Vector *buffer, Vector *d_values) {
    if (buffer->rows != d_values->rows) {
        printf("VECTORS IN RELU BACKWARDS ARE A DIFFERENT SIZE buffer->rows != d_value->rows\n");
        return NULL;
    }

    unsigned int row;
    for (row = 0; row < buffer->rows; row++) {
        // Uses fact that true == 1 and false == 0.
        d_values->values[row] *= (buffer->values[row] > 0);

    }

    return d_values;
}


Vector *nn_softmax_backwards(Matrix *buffer, Matrix *d_values) {
    if (buffer->rows != d_values->rows || buffer->cols != d_values->cols) {
        printf("MATRICES IN SOFTMAX BACKWARDS ARE OF DIFFERENT SIZES.\n");
        return NULL;
    }

    double sum_exp;
    unsigned int row;
    unsigned int col;
    for (row = 0; row < buffer->rows; row++) {

        sum_exp = 0.0;
        for (col = 0; col < buffer->cols; col++) {

            sum_exp += exp(buffer->values[row][col]);

        }
    }



    for (unsigned int i = 0; i < result->rows; i++) {
        // Calculate the denominator for softmax
        double sum_exp = 0.0;
        for (unsigned int j = 0; j < buffer->cols; j++) {
            sum_exp += exp(buffer->values[i][j]);
        }

        // Calculate softmax and its derivative
        for (unsigned int j = 0; j < result->cols; j++) {
            double softmax = exp(buffer->values[i][j]) / sum_exp;
            double softmax_derivative = softmax * (1 - softmax); // Derivative of softmax
            result->values[i][j] = softmax_derivative * d_values->values[i][j];
        }
    }
}
/* END ACTIVATION FUNCTIONS */


/* LOSS FUNCTIONS */
double nn_mean_squared_error_loss(const Vector *y_hat, const Vector *y) {
    if (y_hat->rows != y->rows) {
        printf("LOSS FUNCTION INPUT AND TARGET ARE DIFFERENT SIZES (BAD)\n");
        return -100000;
    }

    double mean_squared_error = 0;

    unsigned int row;
    for (row = 0; row < y_hat->rows; row++) {

        mean_squared_error += (y_hat->values[row] - y->values[row]) * (y_hat->values[row] - y->values[row]);

    }

    mean_squared_error /= y_hat->rows;
    return mean_squared_error;
}


double nn_cross_entropy_loss(const Vector *y_hat, const Vector *y) {
    if (y_hat->rows != y->rows) {
        printf("LOSS FUNCTION Y_HAT AND Y ARE DIFFERENT SIZES (BAD)\n");
        return -100000;
    }

    double cross_entropy_loss = 0;

    unsigned int row;
    for (row = 0; row < y_hat->rows; row++) {

        cross_entropy_loss -= (y->values[row] * log(y_hat->values[row]));

    }

    return cross_entropy_loss;
}
/* END LOSS FUNCTIONS */