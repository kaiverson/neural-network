CC=gcc
CFLAGS=-Wall -Wextra -std=c11

SRCS=main.c linear_algebra.c nn_layers.c
OBJS=$(SRCS:.c=.o)
EXEC=main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(EXEC) $(OBJS)
