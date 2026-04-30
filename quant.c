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

static int strat1(tensor_t *tensor)
{
	float scale_h0;
	float scale_h1;
	float v0;
	float v1;
	float tok_max_h0 = 0.0;
	float tok_max_h1 = 0.0;
	double total_mse0 = 0.0;
	double total_mse1 = 0.0;
	float orig_h0, orig_h1;
	int8_t q_h0, q_h1;
	float deq_h0, deq_h1;
	int i, j;

	for (i = 0; i < TOKENS; i++) {
		for (j = 0; j < DIM; j++) {
			v0 = fabsf((float) tensor[i].head[0].values[j]);
			v1 = fabsf((float) tensor[i].head[1].values[j]);		
			if (v0 > tok_max_h0) 
				tok_max_h0 = v0;
			if (v1 > tok_max_h1) 
				tok_max_h1 = v1;
		}
	}

	scale_h0 = 127 / tok_max_h0;
	scale_h1 = 127 / tok_max_h1;

	for (i = 0; i < TOKENS; i++) {
		for (j = 0; j < DIM; j++) {
			orig_h0 = (float) tensor[i].head[0].values[j];
			orig_h1 = (float) tensor[i].head[1].values[j];

			q_h0 = (int8_t) nearbyintf(scale_h0 * orig_h0);
			q_h1 = (int8_t) nearbyintf(scale_h1 * orig_h1);

			deq_h0 = (float) q_h0 / scale_h0;
			deq_h1 = (float) q_h1 / scale_h1;

			total_mse0 = total_mse0 + ((deq_h0 - orig_h0)*(deq_h0 - orig_h0));
			total_mse1 = total_mse1 + ((deq_h1 - orig_h1)*(deq_h1 - orig_h1));
		}
	}

	print_report(0, tok_max_h0, total_mse0);
	print_report(1, tok_max_h1, total_mse1);

	return 0;
}

static int strat2(tensor_t *tensor)
{
	float scale_h0;
	float scale_h1;
	float v0;
	float v1;
	float tok_max_h0 = 0.0;
	float tok_max_h1 = 0.0;
	float max_h0 = 0.0;
	float max_h1 = 0.0;
	double total_mse0 = 0.0;
	double total_mse1 = 0.0;
	float orig_h0, orig_h1;
	int8_t q_h0, q_h1;
	float deq_h0, deq_h1;
	int i, j;

	for (i = 0; i < TOKENS; i++) {
		tok_max_h0 = 0;
		tok_max_h1 = 0;
		for (j = 0; j < DIM; j++) {
			v0 = fabsf((float) tensor[i].head[0].values[j]);
			v1 = fabsf((float) tensor[i].head[1].values[j]);		
			if (v0 > tok_max_h0)
				tok_max_h0 = v0;
			if (v1 > tok_max_h1)
				tok_max_h1 = v1;
		}

		scale_h0 = 127 / tok_max_h0;
		scale_h1 = 127 / tok_max_h1;

		for (j = 0; j < DIM; j++) {
			orig_h0 = (float) tensor[i].head[0].values[j];
			orig_h1 = (float) tensor[i].head[1].values[j];

			q_h0 = (int8_t) nearbyintf(scale_h0 * orig_h0);
			q_h1 = (int8_t) nearbyintf(scale_h1 * orig_h1);

			deq_h0 = (float) q_h0 / scale_h0;
			deq_h1 = (float) q_h1 / scale_h1;
			
			total_mse0 = total_mse0 + ((deq_h0 - orig_h0)*(deq_h0 - orig_h0));
			total_mse1 = total_mse1 + ((deq_h1 - orig_h1)*(deq_h1 - orig_h1));
		}
		if (tok_max_h0 > max_h0)
			max_h0 = tok_max_h0;
		
		if (tok_max_h1 > max_h1)
			max_h1 = tok_max_h1;
	}

	print_report(0, max_h0, total_mse0);
	print_report(1, max_h1, total_mse1);

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

