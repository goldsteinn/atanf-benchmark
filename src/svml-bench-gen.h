#ifndef _SRC__SVML_BENCH_GEN_H_
#define _SRC__SVML_BENCH_GEN_H_

#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define make_TRANS(type)                                                       \
    __m256i BENCH_FUNC CAT(transpose_kernel, type)(                            \
        __m256i r0, __m256i r1, __m256i r2, __m256i r3, __m256i r4,            \
        __m256i r5, __m256i r6, __m256i r7) {                                  \
        __m256i p01_lo = _mm256_unpacklo_epi32(r0, r1);                        \
        __m256i p01_hi = _mm256_unpackhi_epi32(r0, r1);                        \
                                                                               \
        __m256i p23_lo = _mm256_unpacklo_epi32(r2, r3);                        \
        __m256i p23_hi = _mm256_unpackhi_epi32(r2, r3);                        \
                                                                               \
        __m256i p45_lo = _mm256_unpacklo_epi32(r4, r5);                        \
        __m256i p45_hi = _mm256_unpackhi_epi32(r4, r5);                        \
                                                                               \
        __m256i p67_lo = _mm256_unpacklo_epi32(r6, r7);                        \
        __m256i p67_hi = _mm256_unpackhi_epi32(r6, r7);                        \
                                                                               \
        __m256i p0123_lo_lo = _mm256_unpacklo_epi64(p01_lo, p23_lo);           \
        __m256i p4567_lo_lo = _mm256_unpacklo_epi64(p45_lo, p67_lo);           \
                                                                               \
        __m256i p0123_lo_hi = _mm256_unpackhi_epi64(p01_lo, p23_lo);           \
        __m256i p4567_lo_hi = _mm256_unpackhi_epi64(p45_lo, p67_lo);           \
                                                                               \
        __m256i p0123_hi_lo = _mm256_unpacklo_epi64(p01_hi, p23_hi);           \
        __m256i p4567_hi_lo = _mm256_unpacklo_epi64(p45_hi, p67_hi);           \
                                                                               \
        __m256i p0123_hi_hi = _mm256_unpackhi_epi64(p01_hi, p23_hi);           \
        __m256i p4567_hi_hi = _mm256_unpackhi_epi64(p45_hi, p67_hi);           \
                                                                               \
        __m256i p01234567_lo_lo_lo =                                           \
            _mm256_permute2x128_si256(p0123_lo_lo, p4567_lo_lo, 0x20);         \
        __m256i p01234567_lo_lo_hi =                                           \
            _mm256_permute2x128_si256(p0123_lo_lo, p4567_lo_lo, 0x31);         \
                                                                               \
        __m256i p01234567_lo_hi_lo =                                           \
            _mm256_permute2x128_si256(p0123_lo_hi, p4567_lo_hi, 0x20);         \
        __m256i p01234567_lo_hi_hi =                                           \
            _mm256_permute2x128_si256(p0123_lo_hi, p4567_lo_hi, 0x31);         \
                                                                               \
        __m256i p01234567_hi_lo_lo =                                           \
            _mm256_permute2x128_si256(p0123_hi_lo, p4567_hi_lo, 0x20);         \
        __m256i p01234567_hi_lo_hi =                                           \
            _mm256_permute2x128_si256(p0123_hi_lo, p4567_hi_lo, 0x31);         \
                                                                               \
        __m256i p01234567_hi_hi_lo =                                           \
            _mm256_permute2x128_si256(p0123_hi_hi, p4567_hi_hi, 0x20);         \
        __m256i p01234567_hi_hi_hi =                                           \
            _mm256_permute2x128_si256(p0123_hi_hi, p4567_hi_hi, 0x31);         \
                                                                               \
        return _mm256_castps_si256(_mm256_add_ps(                              \
            _mm256_add_ps(                                                     \
                _mm256_add_ps(_mm256_castsi256_ps(p01234567_lo_lo_lo),         \
                              _mm256_castsi256_ps(p01234567_lo_lo_hi)),        \
                _mm256_add_ps(_mm256_castsi256_ps(p01234567_lo_hi_lo),         \
                              _mm256_castsi256_ps(p01234567_lo_hi_hi))),       \
            _mm256_add_ps(                                                     \
                _mm256_add_ps(_mm256_castsi256_ps(p01234567_hi_lo_lo),         \
                              _mm256_castsi256_ps(p01234567_hi_lo_hi)),        \
                _mm256_add_ps(_mm256_castsi256_ps(p01234567_hi_hi_lo),         \
                              _mm256_castsi256_ps(p01234567_hi_hi_hi)))));     \
    }                                                                          \
                                                                               \
    type CAT(TRANS, type)(type v) {                                            \
        __m256i _v;                                                            \
        __builtin_memcpy(&_v, &v, MIN(sizeof(_v), sizeof(v)));                 \
        _v = CAT(transpose_kernel, type)(_v, _v, _v, _v, _v, _v, _v, _v);      \
        _v = CAT(transpose_kernel, type)(_v, _v, _v, _v, _v, _v, _v, _v);      \
        _v = CAT(transpose_kernel, type)(_v, _v, _v, _v, _v, _v, _v, _v);      \
        _v = CAT(transpose_kernel, type)(_v, _v, _v, _v, _v, _v, _v, _v);      \
        _v = CAT(transpose_kernel, type)(_v, _v, _v, _v, _v, _v, _v, _v);      \
        _v = CAT(transpose_kernel, type)(_v, _v, _v, _v, _v, _v, _v, _v);      \
        __builtin_memcpy(&v, &_v, MIN(sizeof(_v), sizeof(v)));                 \
        return v;                                                              \
    }


#define make_P5(type)                                                          \
    static type BENCH_FUNC CAT(P5, type)(type v) {                             \
        type sink;                                                             \
        __asm__ volatile(                                                      \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            "vpalignr $0, %[v], %[v], %[sink]\n"                               \
            : [sink] "=&v"(sink)                                               \
            : [v] "v"(v)                                                       \
            : "eax", "ecx");                                                   \
        return sink;                                                           \
    }


#define make_ALU(type)                                                         \
    static type BENCH_FUNC CAT(ALU, type)(type v) {                            \
        type sink;                                                             \
        __asm__ volatile(                                                      \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            "vpaddd %[v], %[v], %[sink]\n"                                     \
            : [sink] "=&v"(sink)                                               \
            : [v] "v"(v)                                                       \
            :);                                                                \
        return sink;                                                           \
    }


#define make_FLOP(type)                                                        \
    static type BENCH_FUNC CAT(FLOP, type)(type v) {                           \
        type sink;                                                             \
        __asm__ volatile(                                                      \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            "vaddps %[v], %[v], %[sink]\n"                                     \
            : [sink] "=&v"(sink)                                               \
            : [v] "v"(v)                                                       \
            :);                                                                \
        return sink;                                                           \
    }


#define _CAT(x, y) x##y
#define CAT(x, y)  _CAT(x, y)

#define CAT3(x, y, z) CAT(CAT(x, y), z)

#define zero___m128 _mm_set1_ps(0)
#define zero___m256 _mm256_set1_ps(0)
#define zero___m512 _mm512_set1_ps(0)

#define BENCH_FUNC __attribute__((noinline, noclone, aligned(4096)))
#define IMPOSSIBLE(x)                                                          \
    if (x) {                                                                   \
        __builtin_unreachable();                                               \
    }

#define ll_time_t         uint64_t
#define ll_time_dif(e, s) (e) - (s)
#define get_ll_time()     _rdtsc()

#define serialize_ooe() __asm__ volatile("lfence" : : :)
#define tput_do_not_optimize_out(x)                                            \
    __asm__ volatile("" : : "r,m,v"(x) : "memory")
#define lat_do_not_optimize_out(x)                                             \
    __asm__ volatile("lfence" : : "r,m,v"(x) : "memory")

#define _bench_name(func_name, bench_conf)                                     \
    CAT(CAT(run_bench_, func_name), bench_conf)


#define make_bench_kernel(func_name, type, run_bench_iter, bench_conf)         \
    static BENCH_FUNC uint64_t _bench_name(func_name,                          \
                                           bench_conf)(uint32_t trials) {      \
        ll_time_t start, end;                                                  \
        start  = get_ll_time();                                                \
        type v = CAT(zero_, type);                                             \
        serialize_ooe();                                                       \
        IMPOSSIBLE(!trials);                                                   \
        for (; trials; --trials) {                                             \
            run_bench_iter(func_name, v);                                      \
        }                                                                      \
        tput_do_not_optimize_out(v);                                           \
        end = get_ll_time();                                                   \
        return ll_time_dif(end, start);                                        \
    }

#define bench_tput(func, arg) tput_do_not_optimize_out(func(arg))
//#define bench_lat(func, arg) lat_do_not_optimize_out(func(arg))
#define bench_lat(func, arg) (arg = func(arg))


#define make_bench_driver(func_name, type, bench_conf)                         \
    make_bench_kernel(func_name, type, CAT(bench, bench_conf),                 \
                      bench_conf) static void *                                \
    CAT3(bench_, func_name, bench_conf)(void * trials) {                       \
        uint32_t _trials = (uint32_t)(uint64_t)trials;                         \
        pthread_barrier_wait(&bench_barrier);                                  \
        tput_do_not_optimize_out(                                              \
            CAT3(run_bench_, func_name, bench_conf)(_trials / 4));             \
        return (void *)CAT3(run_bench_, func_name, bench_conf)(_trials);       \
    }


#define make_bench(func_name, type)                                            \
    make_bench_driver(func_name, type, _tput);                                 \
    make_bench_driver(func_name, type, _lat);

#endif
