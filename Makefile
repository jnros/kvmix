all: quant

quant: quant.c
	gcc -Wall -Wextra -g quant.c -o quant -lm

clean:
	rm quant

.PHONY: all clean
