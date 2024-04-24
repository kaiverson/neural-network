#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

typedef struct {
    unsigned int rows;
    float *values;
} Vector;

Vector *vector_init(unsigned int rows, float *values);
void vector_free(Vector *vector);
void vector_print(Vector *vector);


typedef struct {
    unsigned int rows;
    unsigned int cols;
    float **values;
} Matrix;

Matrix *matrix_init(unsigned int rows, unsigned int cols);
void matrix_free(Matrix *matrix);


#endif /* __NEURAL_NETWORK_H__ */