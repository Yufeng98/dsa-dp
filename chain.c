#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <string.h>

typedef int64_t anchor_idx_t;
typedef uint32_t tag_t;
typedef int32_t loc_t;
typedef int32_t loc_dist_t;
typedef int32_t score_t;
typedef int32_t parent_t;
typedef int32_t target_t;
typedef int32_t peak_score_t;

#define ANCHOR_NULL (anchor_idx_t)(-1)

#ifndef N
#define N 14888
#endif

typedef struct anchor_t {
    uint64_t x;
    uint64_t y;
} anchor_t;

typedef struct call_t {
    anchor_idx_t n;
    float avg_qspan;
    int max_dist_x, max_dist_y, bw, n_segs;
    anchor_t* anchors;
} call_t;

typedef struct return_t {
    anchor_idx_t n;
    score_t* scores;
    parent_t* parents;
} return_t;

struct Arguments {
    anchor_idx_t n;
    float avg_qspan;
    int max_dist_x, max_dist_y, bw, n_segs;
    anchor_t anchors[N];
    score_t scores[N];
    parent_t parents[N];
    int32_t max_index[N];
    int32_t score[N];
    int32_t q_span_array[N];
} args_;


void skip_to_EOR(FILE *fp) {
    const char *loc = "EOR";
    while (*loc != '\0') {
        if (fgetc(fp) == *loc) {
            loc++;
        }
    }
}

call_t read_call(FILE *fp) {
    call_t call;

    long long n;
    float avg_qspan;
    int max_dist_x, max_dist_y, bw, n_segs;

    int t = fscanf(fp, "%lld%f%d%d%d%d",
            &n, &avg_qspan, &max_dist_x, &max_dist_y, &bw, &n_segs);
    // fprintf(stderr, "read %d arguments\n", t);
    if (t != 6) {
        call.n = ANCHOR_NULL;
        call.avg_qspan = .0;
        return call;
    }

    call.n = n;
    call.avg_qspan = avg_qspan;
    call.max_dist_x = max_dist_x;
    call.max_dist_y = max_dist_y;
    call.bw = bw;
    call.n_segs = n_segs;
    // fprintf(stderr, "%lld\t%f\t%d\t%d\t%d\t%d\n", n, avg_qspan, max_dist_x, max_dist_y, bw, n_segs);

    call.anchors = malloc(sizeof(anchor_t) * call.n);

    for (anchor_idx_t i = 0; i < call.n; i++) {
        uint64_t x, y;
        fscanf(fp, "%lu%lu", &x, &y);

        anchor_t t;
        t.x = x; t.y = y;

        call.anchors[i] = t;
    }

    skip_to_EOR(fp);
    return call;
}

void print_return(FILE *fp, return_t *data)
{
    for (anchor_idx_t i = 0; i < data->n; i++) {
        fprintf(fp, "%d\n", (int)data->scores[i]);
    }
}


static const char LogTable256[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
	-1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
	LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};

static inline int ilog2_32(uint32_t v)
{
	uint32_t t, tt;
	if ((tt = v>>16)) return (t = tt>>8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
	return (t = v>>8) ? 8 + LogTable256[t] : LogTable256[v];
}

int32_t compute_sc(int32_t dr, int32_t dq, int32_t dd, float avg_qspan, int32_t q_span){
	int32_t min_d = dq < dr? dq : dr;	
	int32_t alpha = min_d > q_span? q_span : min_d;	// alpha
	int32_t log_dd = dd? ilog2_32(dd) : 0;
	int32_t gap_cost = (int)(dd * .01 * avg_qspan) + (log_dd>>1);
	return alpha - gap_cost;
}

void run_accelerator(struct Arguments *args)
{

	// TODO: make sure this works when n has more than 32 bits
	int64_t i, j;

	int64_t ri, rj;
	int32_t qi, qj, dr, dq, l, end, flag;
	for (j = 0; j < args->n; ++j) {
		rj = args->anchors[j].x;
		qj = (int32_t)args->anchors[j].y;
		end = (j + 64) >= args->n ? args->n-1 : j + 64;
		for (i = j + 1; i <= end; i++) {
			ri = args->anchors[i].x;
			qi = (int32_t)args->anchors[i].y;
			
			dr = (int32_t)ri - (int32_t)rj;
			dq = qi - qj;
			l = dr - dq > dq - dr ? dr - dq : dq - dr;

			flag = dr == 0 ? 1 : 0;
			flag = dq <= 0 ? 1 : flag;
			flag = dq > args->max_dist_y ? 1 : flag;
			flag = l > args->bw ? 1 : flag;

			int32_t sc;
			if (flag) sc = INT32_MIN;
			else sc = compute_sc(dr, dq, l, args->avg_qspan, args->q_span_array[i]);

			sc += args->score[j];
			if (sc >= args->score[i]) {
				args->score[i] = sc, args->max_index[i] = j;
			}
		}
	}

	for (j = 0; j < args->n; j++) {
		args->scores[j] = args->score[j], args->parents[j] = args->max_index[j];
	}
}

struct Arguments *init_data() {
    FILE *in;

    in = fopen("in-3.txt", "r");
    call_t *calls;
    calls = malloc(sizeof(call_t) * 3);

    int i;
    int index = 0;
    for (call_t call = read_call(in);
            call.n != ANCHOR_NULL;
            call = read_call(in)) {
        calls[index++] = call;
    }
    call_t* arg;
    arg = &calls[0];
    // struct Arguments *args;
    // args = malloc(sizeof(struct Arguments));
    args_.n = arg->n;
    args_.avg_qspan = arg->avg_qspan;
    args_.max_dist_x = arg->max_dist_x;
    args_.max_dist_y = arg->max_dist_y;
    args_.bw = arg->bw;
    args_.n_segs = arg->n_segs;
    for (i=0; i<N; i++) {
        args_.anchors[i] = arg->anchors[i];
        args_.max_index[i] = -1;
        args_.q_span_array[i] = arg->anchors[i].y>>32 & 0xff;
        args_.score[i] = args_.q_span_array[i];
    }
    fclose(in);
    for (i=0; i<index; i++){
        free(calls[i].anchors);
    }
    free(calls);

    return &args_;
}


int main() {
    
    return_t *rets;
    rets = malloc(sizeof(return_t) * 3);

    struct timeval start_time, end_time;
    double runtime = 0;
    

    gettimeofday(&start_time, NULL);

    return_t* ret;
    ret = &rets[0];
    run_accelerator(init_data());
    ret->scores = args_.scores;
    ret->parents = args_.parents;

    gettimeofday(&end_time, NULL);

    runtime += (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_usec - start_time.tv_usec);
    

    fprintf(stderr, "Time in kernel: %.2f sec\n", runtime * 1e-6);

    free(rets);

    return 0;
}
