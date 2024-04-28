#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>


int main() {
    printf("Starting test...\n");

    Matrix *identity = matrix_init_identity(50, 50);
    Vector *vector_in = vector_init_randn(50);
    Vector *vector_out = vector_init_empty(50);

    vector_out = matrix_times_vector(identity, vector_in, vector_out);

    int passes = vector_equal(vector_in, vector_out);
    

    if (passes) {
        printf("PASSES ");
    } else {
        printf("FAILS ");
    }
    printf("MATRIX TIMES VECTOR TEST\n");

    matrix_free(identity);
    vector_free(vector_in);
    vector_free(vector_out);

    return 0;
}