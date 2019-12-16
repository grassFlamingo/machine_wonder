
.PHONY: all

default: all

CC=gcc
CFLAGS = -Wall -O3
CFLAGS += -I "$(PWD)"
CFLAGS_END = -lGL -lGLU -lglut
CCCOMP = $(CC) $(CFLAGS) -c

all: main
	@echo "Builded"
	
main: main.c
	$(CC) $(CFLAGS) -o main.o main.c $(CFLAGS_END)

%.o: %.cpp %.h
	$(CCCOMP) $< -o $@

.PHONY: clean

clean:
	@-rm -f *.o