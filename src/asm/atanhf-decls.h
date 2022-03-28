#ifndef _SRC__ASM__ATANHF_DECLS_H_
#define _SRC__ASM__ATANHF_DECLS_H_

extern __m512 atanhf16_glibc(__m512);
extern __m256 atanhf8_glibc(__m256);
extern __m128 atanhf4_glibc(__m128);

extern __m512 atanhf16_dev(__m512);
extern __m256 atanhf8_dev(__m256);
extern __m128 atanhf4_dev(__m128);

#endif
