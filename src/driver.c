#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#include <pthread.h>

#include "asm/atanhf-decls.h"


static pthread_barrier_t bench_barrier;
#include "svml-bench-gen.h"

make_ALU(__m128);
make_ALU(__m256);
make_ALU(__m512);


make_P5(__m128);
make_P5(__m256);
make_P5(__m512);

make_TRANS(__m128);
make_TRANS(__m256);
make_TRANS(__m512);

make_FLOP(__m128);
make_FLOP(__m256);
make_FLOP(__m512);


make_bench(atanhf16_glibc, __m512);
make_bench(atanhf8_glibc, __m256);
make_bench(atanhf4_glibc, __m128);

make_bench(atanhf16_dev, __m512);
make_bench(atanhf8_dev, __m256);
make_bench(atanhf4_dev, __m128);

make_bench(ALU__m512, __m512);
make_bench(ALU__m256, __m256);
make_bench(ALU__m128, __m128);

make_bench(P5__m512, __m512);
make_bench(P5__m256, __m256);
make_bench(P5__m128, __m128);

make_bench(TRANS__m512, __m512);
make_bench(TRANS__m256, __m256);
make_bench(TRANS__m128, __m128);

make_bench(FLOP__m512, __m512);
make_bench(FLOP__m256, __m256);
make_bench(FLOP__m128, __m128);


#define _V_TO_STR(x) #x
#define V_TO_STR(x)  _V_TO_STR(x)

typedef struct bench_res {
    char const * name0;
    char const * name1;
    char const * confs;
    uint32_t     nthreads;
    uint32_t     ntrials;
    uint64_t     cycles[8];
} bench_res_t;


double
get_accum(bench_res_t const * bench_res) {
    double accum = 0.0;
    for (uint32_t i = 0; i < bench_res->nthreads; ++i) {
        if (i % 2 == 0 || !strcmp(bench_res->name0, bench_res->name1)) {
            accum += bench_res->cycles[i];
        }
    }
    uint32_t denum = bench_res->nthreads;
    if (denum > 1 && strcmp(bench_res->name0, bench_res->name1)) {
        denum /= 2;
    }
    accum /= denum;
    accum /= (bench_res->ntrials);
    return accum;
}

static void
print_bench_res(bench_res_t const * bench_res) {

    fprintf(stderr, "%-16s,%-16s,%-6s,%-3u,%-10u,%.3lf\n", bench_res->name0,
            bench_res->name1, bench_res->confs, bench_res->nthreads,
            bench_res->ntrials,
            get_accum(bench_res + 1) / get_accum(bench_res));
}


#define run_bench_kernel(results, n, func0, func1, conf)                       \
    ({                                                                         \
        for (uint32_t _i = 0; _i < (n); ++_i) {                                \
            if (_i % 2 == 0) {                                                 \
                assert(pthread_create(tids + _i, attrs + _i,                   \
                                      CAT3(bench_, func0, conf),               \
                                      (void *)CAT(conf, _TRIALS)) == 0);       \
            }                                                                  \
            else {                                                             \
                assert(pthread_create(tids + _i, attrs + _i,                   \
                                      CAT3(bench_, func1, conf),               \
                                      (void *)CAT(conf, _TRIALS)) == 0);       \
            }                                                                  \
        }                                                                      \
                                                                               \
        for (uint32_t _i = 0; _i < (n); ++_i) {                                \
            pthread_join(tids[_i], (void **)((results)->cycles + _i));         \
        }                                                                      \
                                                                               \
        (results)->name0    = V_TO_STR(func0);                                 \
        (results)->name1    = V_TO_STR(func1);                                 \
        (results)->confs    = V_TO_STR(conf);                                  \
        (results)->nthreads = n;                                               \
        (results)->ntrials  = CAT(conf, _TRIALS);                              \
        ++(results);                                                           \
    })

enum { _tput_TRIALS = (100 * 1000 * 1000), _lat_TRIALS = (10 * 1000 * 1000) };

bench_res_t *
run_benchmarks_kernel(bench_res_t * results, uint32_t n) {
    pthread_t      tids[8];
    pthread_attr_t attrs[8];
    cpu_set_t      csets[8];
    assert((n) <= 8);
    pthread_barrier_init(&bench_barrier, NULL, n);
    for (uint32_t _i = 0; _i < (n); ++_i) {
        assert(pthread_attr_init(attrs + _i) == 0);
        assert(pthread_attr_setstacksize(attrs + _i, 32768) == 0);

        CPU_ZERO(csets + _i);
        CPU_SET((_i / 2) + ((_i % 2) ? 4 : 0), csets + _i);

        assert(pthread_attr_setaffinity_np(attrs + _i, sizeof(cpu_set_t),
                                           csets + _i) == 0);
    }

    run_bench_kernel(results, n, atanhf16_glibc, atanhf16_glibc, _tput);
    run_bench_kernel(results, n, atanhf16_dev, atanhf16_dev, _tput);
    run_bench_kernel(results, n, atanhf8_glibc, atanhf8_glibc, _tput);
    run_bench_kernel(results, n, atanhf8_dev, atanhf8_dev, _tput);
    run_bench_kernel(results, n, atanhf4_glibc, atanhf4_glibc, _tput);
    run_bench_kernel(results, n, atanhf4_dev, atanhf4_dev, _tput);

    run_bench_kernel(results, n, atanhf16_glibc, ALU__m512, _tput);
    run_bench_kernel(results, n, atanhf16_dev, ALU__m512, _tput);
    run_bench_kernel(results, n, atanhf8_glibc, ALU__m256, _tput);
    run_bench_kernel(results, n, atanhf8_dev, ALU__m256, _tput);
    run_bench_kernel(results, n, atanhf4_glibc, ALU__m128, _tput);
    run_bench_kernel(results, n, atanhf4_dev, ALU__m128, _tput);

    run_bench_kernel(results, n, atanhf16_glibc, P5__m512, _tput);
    run_bench_kernel(results, n, atanhf16_dev, P5__m512, _tput);
    run_bench_kernel(results, n, atanhf8_glibc, P5__m256, _tput);
    run_bench_kernel(results, n, atanhf8_dev, P5__m256, _tput);
    run_bench_kernel(results, n, atanhf4_glibc, P5__m128, _tput);
    run_bench_kernel(results, n, atanhf4_dev, P5__m128, _tput);

    run_bench_kernel(results, n, atanhf16_glibc, FLOP__m512, _tput);
    run_bench_kernel(results, n, atanhf16_dev, FLOP__m512, _tput);
    run_bench_kernel(results, n, atanhf8_glibc, FLOP__m256, _tput);
    run_bench_kernel(results, n, atanhf8_dev, FLOP__m256, _tput);
    run_bench_kernel(results, n, atanhf4_glibc, FLOP__m128, _tput);
    run_bench_kernel(results, n, atanhf4_dev, FLOP__m128, _tput);

    run_bench_kernel(results, n, atanhf16_glibc, TRANS__m512, _tput);
    run_bench_kernel(results, n, atanhf16_dev, TRANS__m512, _tput);
    run_bench_kernel(results, n, atanhf8_glibc, TRANS__m256, _tput);
    run_bench_kernel(results, n, atanhf8_dev, TRANS__m256, _tput);
    run_bench_kernel(results, n, atanhf4_glibc, TRANS__m128, _tput);
    run_bench_kernel(results, n, atanhf4_dev, TRANS__m128, _tput);

    run_bench_kernel(results, n, atanhf16_glibc, atanhf16_glibc, _lat);
    run_bench_kernel(results, n, atanhf16_dev, atanhf16_dev, _lat);
    run_bench_kernel(results, n, atanhf8_glibc, atanhf8_glibc, _lat);
    run_bench_kernel(results, n, atanhf8_dev, atanhf8_dev, _lat);
    run_bench_kernel(results, n, atanhf4_glibc, atanhf4_glibc, _lat);
    run_bench_kernel(results, n, atanhf4_dev, atanhf4_dev, _lat);

    run_bench_kernel(results, n, atanhf16_glibc, ALU__m512, _lat);
    run_bench_kernel(results, n, atanhf16_dev, ALU__m512, _lat);
    run_bench_kernel(results, n, atanhf8_glibc, ALU__m256, _lat);
    run_bench_kernel(results, n, atanhf8_dev, ALU__m256, _lat);
    run_bench_kernel(results, n, atanhf4_glibc, ALU__m128, _lat);
    run_bench_kernel(results, n, atanhf4_dev, ALU__m128, _lat);

    run_bench_kernel(results, n, atanhf16_glibc, P5__m512, _lat);
    run_bench_kernel(results, n, atanhf16_dev, P5__m512, _lat);
    run_bench_kernel(results, n, atanhf8_glibc, P5__m256, _lat);
    run_bench_kernel(results, n, atanhf8_dev, P5__m256, _lat);
    run_bench_kernel(results, n, atanhf4_glibc, P5__m128, _lat);
    run_bench_kernel(results, n, atanhf4_dev, P5__m128, _lat);

    run_bench_kernel(results, n, atanhf16_glibc, FLOP__m512, _lat);
    run_bench_kernel(results, n, atanhf16_dev, FLOP__m512, _lat);
    run_bench_kernel(results, n, atanhf8_glibc, FLOP__m256, _lat);
    run_bench_kernel(results, n, atanhf8_dev, FLOP__m256, _lat);
    run_bench_kernel(results, n, atanhf4_glibc, FLOP__m128, _lat);
    run_bench_kernel(results, n, atanhf4_dev, FLOP__m128, _lat);

    run_bench_kernel(results, n, atanhf16_glibc, TRANS__m512, _lat);
    run_bench_kernel(results, n, atanhf16_dev, TRANS__m512, _lat);
    run_bench_kernel(results, n, atanhf8_glibc, TRANS__m256, _lat);
    run_bench_kernel(results, n, atanhf8_dev, TRANS__m256, _lat);
    run_bench_kernel(results, n, atanhf4_glibc, TRANS__m128, _lat);
    run_bench_kernel(results, n, atanhf4_dev, TRANS__m128, _lat);


    for (uint32_t _i = 0; _i < (n); ++_i) {
        pthread_attr_destroy(attrs + _i);
    }
    pthread_barrier_destroy(&bench_barrier);

    return results;
}

int
main() {
    bench_res_t   results[1024];
    bench_res_t * res_begin = results;
    bench_res_t * res_end   = results;

    for (uint32_t i = 1; i <= 2; i += i) {
        res_end = run_benchmarks_kernel(res_end, i);
    }

    for (; res_begin != res_end; res_begin += 2) {
        print_bench_res(res_begin);
    }
}
