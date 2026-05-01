#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <math.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h>

#define TOKENS 4096
#define HEADS 2
#define DIM 64

/* C-order [4096,2,64] */
typedef struct {
	_Float16 values[DIM];
} hvector_t;

typedef struct {
	hvector_t head[HEADS];
} tensor_t;

static int strat = 0;
static int csv_mode = 0;
static char *in_filename = NULL;

static void print_report(int head, float max, float total_mse)
{
	char *fn = strrchr(in_filename, '/');
	int layer,rc;
	char korv;

	if (fn)
		fn = fn + 1;
	else
		fn = in_filename;

	rc = sscanf(fn, "layer_%d.%c.bin", &layer, &korv);
	if (rc != 2) {
		fprintf(stderr, "error: unexpected filename\n");
		return;
	}

	if (csv_mode)
		printf("%s,%d,%c,%d,%d,%d,%d,%.6f,%.6f,%0.6f\n", fn, layer, korv, TOKENS, head, DIM, strat, max, total_mse, total_mse/(TOKENS*DIM));
	else
		printf("%s strat %d head %d max %0.6f MSE %0.6f\n", fn, strat, head, max, total_mse/(TOKENS*DIM));
}

static double quant_head_per_tensor(tensor_t *tensor, int head, float *maxout)
{
	float max = 0.0, scale, orig, deq, v;
	double mse = 0.0;
	int8_t q;
	int i, j;

	for (i = 0; i < TOKENS; i++)
		for (j = 0; j < DIM; j++) {
			v = fabsf((float) tensor[i].head[head].values[j]);
			if (v > max)
				max = v;
		}

	scale = 127 / max;
	for (i = 0; i < TOKENS; i++)
		for (j = 0; j < DIM; j++) {
			orig = (float) tensor[i].head[head].values[j];
			q = (int8_t) nearbyintf(scale * orig);
			deq = (float) q / scale;
			mse = mse + ((deq - orig)*(deq - orig));
		}

	*maxout = max;
	return mse;
}

static double quant_head_per_token(tensor_t *tensor, int head, float *maxout)
{
	float max = 0.0, tok_max, scale, orig, deq, v;
	double mse = 0.0;
	int8_t q;
	int i, j;

	for (i = 0; i < TOKENS; i++) {
		tok_max = 0;
		for (j = 0; j < DIM; j++) {
			v = fabsf((float) tensor[i].head[head].values[j]);
			if (v > tok_max)
				tok_max = v;
		}
		scale = 127 / tok_max;
		for (j = 0; j < DIM; j++) {
			orig = (float) tensor[i].head[head].values[j];
			q = (int8_t) nearbyintf(scale * orig);
			deq = (float) q / scale;
			mse = mse + ((deq - orig)*(deq - orig));
		}
		if (tok_max > max)
			max = tok_max;
	}

	*maxout = max;
	return mse;
}

static int strat1(tensor_t *tensor)
{
	float max0, max1;
	double mse0, mse1;

	mse0 = quant_head_per_tensor(tensor, 0, &max0);
	mse1 = quant_head_per_tensor(tensor, 1, &max1);
	print_report(0, max0, mse0);
	print_report(1, max1, mse1);
	return 0;
}

static int strat2(tensor_t *tensor)
{
	float max0, max1;
	double mse0, mse1;

	mse0 = quant_head_per_token(tensor, 0, &max0);
	mse1 = quant_head_per_token(tensor, 1, &max1);
	print_report(0, max0, mse0);
	print_report(1, max1, mse1);
	return 0;
}

static int k_outlier(int layer, int head)
{
	static const int ol[][2] = {
		{0,0},{0,1},{1,1},{2,0},{2,1},{8,0}
	};
	int i;
	for (i = 0; i < 6; i++)
		if (ol[i][0] == layer && ol[i][1] == head)
			return 1;
	return 0;
}

static int strat3(tensor_t *tensor)
{
	char *fn = strrchr(in_filename, '/');
	int layer, rc, h;
	char korv;
	float max;
	double mse;

	if (fn)
		fn = fn + 1;
	else
		fn = in_filename;

	rc = sscanf(fn, "layer_%d.%c.bin", &layer, &korv);
	if (rc != 2) {
		fprintf(stderr, "error: unexpected filename\n");
		return 1;
	}

	for (h = 0; h < HEADS; h++) {
		if (korv == 'k' && k_outlier(layer, h)) {
			print_report(h, 0.0, 0.0);
		} else if (korv == 'v') {
			mse = quant_head_per_token(tensor, h, &max);
			print_report(h, max, mse);
		} else {
			mse = quant_head_per_tensor(tensor, h, &max);
			print_report(h, max, mse);
		}
	}
	return 0;
}

int mainloop()
{
	tensor_t *tensor;
	struct stat st;
	int rc;
	int fd;

	fd = open(in_filename, O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "Error: file open\n");
		return 1;
	}

	fstat(fd, &st);
	tensor = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (tensor == NULL) {
		fprintf(stderr, "Error: mmap\n");
		return 1;
	}

	if (st.st_size != (TOKENS * sizeof(tensor_t))) {
		fprintf(stderr, "Error: unexpected file format\n");	
		return 1;
	}

	if (strat == 1)
		strat1(tensor);
	else if (strat == 2)
		strat2(tensor);
	else if (strat == 3)
		strat3(tensor);

	rc = munmap(tensor, st.st_size);
	if (rc == -1) {
		fprintf(stderr, "Error: munmap\n");
		return 1;
	}
	close(fd);

	return 0;
}

int main(int argc, char **argv)
{
	int opt;
	struct option long_options[] = {
		{"strategy", required_argument, 0, 's'},
		{"input", required_argument, 0, 'i'},
		{"csv", no_argument, 0, 'c'},
		{0, 0, 0, 0}
	};

	while((opt = getopt_long(argc, argv, "s:i:c", long_options, NULL)) != -1) {
		switch(opt) {
			case 's': 
				strat = atoi(optarg); 
				break;
			case 'i': 
				in_filename = optarg; 
				break;
			case 'c': 
				csv_mode = 1;
				break;
			default: 
				break;
		}
	}

	if (in_filename == NULL || strat == 0) {
		fprintf(stderr, "Usage: %s --strategy <1|2|3> --input <file> [--csv]\n", argv[0]);
		return 1;
	}

	return mainloop();
}

