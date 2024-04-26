#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>


int main() {
    printf("Starting test...\n");
    srand(3);

    Matrix *matrix = matrix_init_randn(400, 500);

    Vector *input = vector_init_randn(500);
    vector_print(input);
    
    Vector *output = vector_init_empty(400);

    output = matrix_times_vector(matrix, input, output);

    bool passes = vector_equal(input, output);

    if (passes) {
        printf("PASSES\n");
    } else {
        printf("FAILS\n");
    }

    vector_print(output);

    matrix_free(matrix);
    vector_free(input);
    vector_free(output);

    return 0;
}