all: main test

main: main.c nn_layers.c linear_algebra.c
	gcc main.c nn_layers.c linear_algebra.c -o main

test: test.c nn_layers.c linear_algebra.c
	gcc test.c nn_layers.c linear_algebra.c -o test

clean:
	rm -f main test
