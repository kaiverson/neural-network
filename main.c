#include "neural_network.h"

#include <stdio.h>

int main(int argc, char **argv) {
    float values[] = {-5.0, 3.9, 6.7, 7.2, 8.5};
    Vector *vector = vector_init(5, values);

    vector_print(vector);

    vector_free(vector);

    return 0;
}