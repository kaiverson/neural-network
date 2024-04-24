#include "neural_network.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

Vector *vector_init(unsigned int rows, float *values) {
    Vector *vector;
    vector = malloc((sizeof (Vector)));
    if (vector == NULL) {
        return NULL;
    }

    vector->rows = rows;
    vector->values = values;

    return vector;
}


void vector_free(Vector *vector) {
    free(vector->values);
    free(vector);
}


void vector_print(Vector *vector) {
    printf("[");
    unsigned int i;
    for (i = 0; i < vector->rows - 1; i++) {
        printf("%.2f ", vector->values[i]);
    }
    printf("%.2ff]\n", vector->values[vector->rows - 1]);
}