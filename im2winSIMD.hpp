#ifndef _IM2WINSIMD_
#define _IM2WINSIMD_

#include <omp.h>
#include <immintrin.h>  // AVX2


template<typename dataType>
union v_t{
  __m256d vec;
  dataType value[4];
};

template<typename dataType>
union v_t_s{
  __m256 vec;
  dataType value[8];
};

template<typename dataType>
void ElementMul1x1(dataType *a, dataType *b, dataType &c);

template<typename dataType>
void ElementMul1x8_fma(dataType *a, dataType *b, v_t_s<dataType> &c);

template<typename dataType>
void ElementMul1x4_fma(dataType *a, dataType *b, v_t<dataType> &c);

template<typename dataType>
void ElementMul1x16_fma(dataType *a, dataType *b, v_t_s<dataType> &c);

template<typename dataType>
void ElementMul1x8_fma(dataType *a, dataType *b, v_t<dataType> &c);

template<typename dataType>
void ElementMul1x32_fma(dataType *a, dataType *b, v_t_s<dataType> &c);

template<typename dataType>
void ElementMul1x16_fma(dataType *a, dataType *b, v_t<dataType> &c);

template<typename dataType>
void ElementMul1x64_fma(dataType *a, dataType *b, v_t_s<dataType> &c);

template<typename dataType>
void ElementMul1x32_fma(dataType *a, dataType *b, v_t<dataType> &c);

template<typename dataType>
void ElementMul5x21_fma(dataType *a, dataType *b, v_t<dataType> &c_0, v_t<dataType> &c_1, v_t<dataType> &c_2, v_t<dataType> &c_3, v_t<dataType> &c_4);

template<typename dataType>
void ElementMulWin_S(size_t k, dataType *a, dataType *b, dataType *c);

template<typename dataType>
void ElementMulWin_M(size_t k, size_t n, dataType *a, dataType *b, dataType *c);

template<typename dataType>
void ElementMulWin_col(dataType *a, dataType *b, size_t *dims_b, dataType *c, size_t *dims_c, size_t stride);

template<typename dataType>
void ElementMulWin_col_row(dataType *a, size_t *dims_a, dataType *b, size_t *dims_b, dataType *c, size_t *dims_c, size_t stride);

template<typename dataType>
void SW_CONV(dataType *const a, dataType *const b, dataType *const c, size_t *const dims_a, size_t *const dims_b, size_t *const dims_c);

template<typename dataType>
void Init_C(dataType *const c, size_t *const dims_c);

template<typename dataType>
inline void ElementMul1x1(dataType *a, dataType *b, dataType &c)
{
    c += (*a) * (*b);
    return;
  }

template<>
inline void ElementMul1x8_fma(float *a, float *b, v_t_s<float> &c){
  v_t_s<float>
    a_0p_a_7p_vreg,
    b_p0_b_p7_vreg;

    a_0p_a_7p_vreg.vec = _mm256_loadu_ps((float *) a); 
    b_p0_b_p7_vreg.vec = _mm256_loadu_ps((float *) b);

    // c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec;
    c.vec = _mm256_fmadd_ps(a_0p_a_7p_vreg.vec, b_p0_b_p7_vreg.vec, c.vec);
  return;
}

template<>
inline void ElementMul1x4_fma(double *a, double *b, v_t<double> &c){
  v_t<double>
    a_0p_a_1p_a_2p_a_3p_vreg,   b_p0_b_p1_b_p2_b_p3_vreg;

    a_0p_a_1p_a_2p_a_3p_vreg.vec = _mm256_loadu_pd((double *) a); 
    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm256_loadu_pd((double *) b);

    // c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec;
    c.vec = _mm256_fmadd_pd(a_0p_a_1p_a_2p_a_3p_vreg.vec, b_p0_b_p1_b_p2_b_p3_vreg.vec, c.vec);
  return;
}

template<>
inline void ElementMul1x16_fma(float *a, float *b, v_t_s<float> &c){
  v_t_s<float>
    a_0p_a_7p_vreg,   a_8p_a_15p_vreg,
    b_p0_b_p7_vreg,   b_p8_b_p15_vreg;

    a_0p_a_7p_vreg.vec = _mm256_loadu_ps((float *) a); 
    a_8p_a_15p_vreg.vec = _mm256_loadu_ps((float *) a + 8); 
    b_p0_b_p7_vreg.vec = _mm256_loadu_ps((float *) b);
    b_p8_b_p15_vreg.vec = _mm256_loadu_ps((float *) b + 8);

    // c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec
    //         +a_4p_a_5p_a_6p_a_7p_vreg.vec * b_p4_b_p5_b_p6_b_p7_vreg.vec;
    c.vec = _mm256_fmadd_ps(a_8p_a_15p_vreg.vec, b_p8_b_p15_vreg.vec, 
            _mm256_fmadd_ps(a_0p_a_7p_vreg.vec, b_p0_b_p7_vreg.vec, c.vec));
  return;
}

template<>
inline void ElementMul1x8_fma(double *a, double *b, v_t<double> &c){
  v_t<double>
    a_0p_a_1p_a_2p_a_3p_vreg,   a_4p_a_5p_a_6p_a_7p_vreg,
    b_p0_b_p1_b_p2_b_p3_vreg,   b_p4_b_p5_b_p6_b_p7_vreg;

    a_0p_a_1p_a_2p_a_3p_vreg.vec = _mm256_loadu_pd((double *) a); 
    a_4p_a_5p_a_6p_a_7p_vreg.vec = _mm256_loadu_pd((double *) a + 4); 
    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm256_loadu_pd((double *) b);
    b_p4_b_p5_b_p6_b_p7_vreg.vec = _mm256_loadu_pd((double *) b + 4);

    // c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec
    //         +a_4p_a_5p_a_6p_a_7p_vreg.vec * b_p4_b_p5_b_p6_b_p7_vreg.vec;
    c.vec = _mm256_fmadd_pd(a_4p_a_5p_a_6p_a_7p_vreg.vec, b_p4_b_p5_b_p6_b_p7_vreg.vec, 
            _mm256_fmadd_pd(a_0p_a_1p_a_2p_a_3p_vreg.vec, b_p0_b_p1_b_p2_b_p3_vreg.vec, c.vec));
  return;
}

template<>
inline void ElementMul1x32_fma(float *a, float *b, v_t_s<float> &c){
  v_t_s<float>
    a_0p_a_7p_vreg,   a_8p_a_15p_vreg,   a_16p_a_23p_vreg,   a_24p_a_31p_vreg,
    b_p0_b_p7_vreg,   b_p8_b_p15_vreg,   b_p16_b_p23_vreg,   b_p24_b_p31_vreg;

    a_0p_a_7p_vreg.vec = _mm256_loadu_ps((float *) a);
    a_8p_a_15p_vreg.vec = _mm256_loadu_ps((float *) a + 8);
    a_16p_a_23p_vreg.vec = _mm256_loadu_ps((float *) a + 16);
    a_24p_a_31p_vreg.vec = _mm256_loadu_ps((float *) a + 24); 
    b_p0_b_p7_vreg.vec = _mm256_loadu_ps((float *) b);
    b_p8_b_p15_vreg.vec = _mm256_loadu_ps((float *) b + 8);
    b_p16_b_p23_vreg.vec = _mm256_loadu_ps((float *) b + 16);
    b_p24_b_p31_vreg.vec = _mm256_loadu_ps((float *) b + 24);

    c.vec = _mm256_fmadd_ps(a_24p_a_31p_vreg.vec, b_p24_b_p31_vreg.vec,
            _mm256_fmadd_ps(a_16p_a_23p_vreg.vec, b_p16_b_p23_vreg.vec,
            _mm256_fmadd_ps(a_8p_a_15p_vreg.vec, b_p8_b_p15_vreg.vec,
            _mm256_fmadd_ps(a_0p_a_7p_vreg.vec, b_p0_b_p7_vreg.vec, c.vec))));
  return;
}

template<>
inline void ElementMul1x16_fma(double *a, double *b, v_t<double> &c){
  v_t<double>
    a_0p_a_1p_a_2P_a_3p_vreg,   a_4p_a_5p_a_6P_a_7p_vreg,   a_8p_a_9p_a_10P_a_11p_vreg,   a_12p_a_13p_a_14P_a_15p_vreg,
    b_p0_b_p1_b_p2_b_p3_vreg,   b_p4_b_p5_b_p6_b_p7_vreg,   b_p8_b_p9_b_p10_b_p11_vreg,   b_p12_b_p13_b_p14_b_p15_vreg;

    a_0p_a_1p_a_2P_a_3p_vreg.vec = _mm256_loadu_pd((double *) a);
    a_4p_a_5p_a_6P_a_7p_vreg.vec = _mm256_loadu_pd((double *) a + 4);
    a_8p_a_9p_a_10P_a_11p_vreg.vec = _mm256_loadu_pd((double *) a + 8);
    a_12p_a_13p_a_14P_a_15p_vreg.vec = _mm256_loadu_pd((double *) a + 12); 
    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm256_loadu_pd((double *) b);
    b_p4_b_p5_b_p6_b_p7_vreg.vec = _mm256_loadu_pd((double *) b + 4);
    b_p8_b_p9_b_p10_b_p11_vreg.vec = _mm256_loadu_pd((double *) b + 8);
    b_p12_b_p13_b_p14_b_p15_vreg.vec = _mm256_loadu_pd((double *) b + 12);

    // c.vec += a_0p_a_1p_a_2P_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec
    //     +  a_4p_a_5p_a_6P_a_7p_vreg.vec * b_p4_b_p5_b_p6_b_p7_vreg.vec
    //     +  a_8p_a_9p_a_10P_a_11p_vreg.vec * b_p8_b_p9_b_p10_b_p11_vreg.vec
    //     +  a_12p_a_13p_a_14P_a_15p_vreg.vec * b_p12_b_p13_b_p14_b_p15_vreg.vec;
    c.vec = _mm256_fmadd_pd(a_12p_a_13p_a_14P_a_15p_vreg.vec, b_p12_b_p13_b_p14_b_p15_vreg.vec,
            _mm256_fmadd_pd(a_8p_a_9p_a_10P_a_11p_vreg.vec, b_p8_b_p9_b_p10_b_p11_vreg.vec,
            _mm256_fmadd_pd(a_4p_a_5p_a_6P_a_7p_vreg.vec, b_p4_b_p5_b_p6_b_p7_vreg.vec,
            _mm256_fmadd_pd(a_0p_a_1p_a_2P_a_3p_vreg.vec, b_p0_b_p1_b_p2_b_p3_vreg.vec, c.vec))));
  return;
}

template<>
inline void ElementMul1x64_fma(float *a, float *b, v_t_s<float> &c){
  v_t_s<float>
    a_0p_a_7p_vreg,   a_8p_a_15p_vreg,   a_16p_a_23p_vreg,   a_24p_a_31p_vreg,
    a_32p_a_39p_vreg,   a_40p_a_47p_vreg,   a_48p_a_55p_vreg,   a_56p_a_63p_vreg,
    b_p0_b_p7_vreg,   b_p8_b_p15_vreg,   b_p16_b_p23_vreg,   b_p24_b_p31_vreg,
    b_p32_b_p39_vreg,   b_p40_b_p47_vreg,   b_p48_b_p55_vreg,   b_p56_b_p63_vreg;

    a_0p_a_7p_vreg.vec = _mm256_loadu_ps((float *) a);
    a_8p_a_15p_vreg.vec = _mm256_loadu_ps((float *) a + 8);
    a_16p_a_23p_vreg.vec = _mm256_loadu_ps((float *) a + 16);
    a_24p_a_31p_vreg.vec = _mm256_loadu_ps((float *) a + 24);
    a_32p_a_39p_vreg.vec = _mm256_loadu_ps((float *) a + 32);
    a_40p_a_47p_vreg.vec = _mm256_loadu_ps((float *) a + 40);
    a_48p_a_55p_vreg.vec = _mm256_loadu_ps((float *) a + 48);
    a_56p_a_63p_vreg.vec = _mm256_loadu_ps((float *) a + 56); 

    b_p0_b_p7_vreg.vec = _mm256_loadu_ps((float *) b);
    b_p8_b_p15_vreg.vec = _mm256_loadu_ps((float *) b + 8);
    b_p16_b_p23_vreg.vec = _mm256_loadu_ps((float *) b + 16);
    b_p24_b_p31_vreg.vec = _mm256_loadu_ps((float *) b + 24);
    b_p32_b_p39_vreg.vec = _mm256_loadu_ps((float *) b + 32);
    b_p40_b_p47_vreg.vec = _mm256_loadu_ps((float *) b + 40);
    b_p48_b_p55_vreg.vec = _mm256_loadu_ps((float *) b + 48);
    b_p56_b_p63_vreg.vec = _mm256_loadu_ps((float *) b + 56);

    c.vec = _mm256_fmadd_ps(a_56p_a_63p_vreg.vec, b_p56_b_p63_vreg.vec,
            _mm256_fmadd_ps(a_48p_a_55p_vreg.vec, b_p48_b_p55_vreg.vec,
            _mm256_fmadd_ps(a_40p_a_47p_vreg.vec, b_p40_b_p47_vreg.vec,
            _mm256_fmadd_ps(a_32p_a_39p_vreg.vec, b_p32_b_p39_vreg.vec,
            _mm256_fmadd_ps(a_24p_a_31p_vreg.vec, b_p24_b_p31_vreg.vec,
            _mm256_fmadd_ps(a_16p_a_23p_vreg.vec, b_p16_b_p23_vreg.vec,
            _mm256_fmadd_ps(a_8p_a_15p_vreg.vec, b_p8_b_p15_vreg.vec,
            _mm256_fmadd_ps(a_0p_a_7p_vreg.vec, b_p0_b_p7_vreg.vec, c.vec))))))));
  return;
}

template<>
inline void ElementMul1x32_fma(double *a, double *b, v_t<double> &c){
  v_t<double>
    a_0p_a_1p_a_2p_a_3p_vreg,   a_4p_a_5p_a_6p_a_7p_vreg,   a_8p_a_9p_a_10p_a_11p_vreg,   a_12p_a_13p_a_14p_a_15p_vreg,
    a_16p_a_17p_a_18p_a_19p_vreg,   a_20p_a_21p_a_22p_a_23p_vreg,   a_24p_a_25p_a_26p_a_27p_vreg,   a_28p_a_29p_a_30p_a_31p_vreg,
    b_p0_b_p1_b_p2_b_p3_vreg,   b_p4_b_p5_b_p6_b_p7_vreg,   b_p8_b_p9_b_p10_b_p11_vreg,   b_p12_b_p13_b_p14_b_p15_vreg,
    b_p16_b_p17_b_p18_b_p19_vreg,   b_p20_b_p21_b_p22_b_p23_vreg,   b_p24_b_p25_b_p26_b_p27_vreg,   b_p28_b_p29_b_p30_b_p31_vreg;

    a_0p_a_1p_a_2p_a_3p_vreg.vec = _mm256_loadu_pd((double *) a);
    a_4p_a_5p_a_6p_a_7p_vreg.vec = _mm256_loadu_pd((double *) a + 4);
    a_8p_a_9p_a_10p_a_11p_vreg.vec = _mm256_loadu_pd((double *) a + 8);
    a_12p_a_13p_a_14p_a_15p_vreg.vec = _mm256_loadu_pd((double *) a + 12);
    a_16p_a_17p_a_18p_a_19p_vreg.vec = _mm256_loadu_pd((double *) a + 16);
    a_20p_a_21p_a_22p_a_23p_vreg.vec = _mm256_loadu_pd((double *) a + 20);
    a_24p_a_25p_a_26p_a_27p_vreg.vec = _mm256_loadu_pd((double *) a + 24);
    a_28p_a_29p_a_30p_a_31p_vreg.vec = _mm256_loadu_pd((double *) a + 28); 

    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm256_loadu_pd((double *) b);
    b_p4_b_p5_b_p6_b_p7_vreg.vec = _mm256_loadu_pd((double *) b + 4);
    b_p8_b_p9_b_p10_b_p11_vreg.vec = _mm256_loadu_pd((double *) b + 8);
    b_p12_b_p13_b_p14_b_p15_vreg.vec = _mm256_loadu_pd((double *) b + 12);
    b_p16_b_p17_b_p18_b_p19_vreg.vec = _mm256_loadu_pd((double *) b + 16);
    b_p20_b_p21_b_p22_b_p23_vreg.vec = _mm256_loadu_pd((double *) b + 20);
    b_p24_b_p25_b_p26_b_p27_vreg.vec = _mm256_loadu_pd((double *) b + 24);
    b_p28_b_p29_b_p30_b_p31_vreg.vec = _mm256_loadu_pd((double *) b + 28);

    // c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec
    //     +  a_4p_a_5p_a_6p_a_7p_vreg.vec * b_p4_b_p5_b_p6_b_p7_vreg.vec
    //     +  a_8p_a_9p_a_10p_a_11p_vreg.vec * b_p8_b_p9_b_p10_b_p11_vreg.vec
    //     +  a_12p_a_13p_a_14p_a_15p_vreg.vec * b_p12_b_p13_b_p14_b_p15_vreg.vec
    //     +  a_16p_a_17p_a_18p_a_19p_vreg.vec * b_p16_b_p17_b_p18_b_p19_vreg.vec
    //     +  a_20p_a_21p_a_22p_a_23p_vreg.vec * b_p20_b_p21_b_p22_b_p23_vreg.vec
    //     +  a_24p_a_25p_a_26p_a_27p_vreg.vec * b_p24_b_p25_b_p26_b_p27_vreg.vec
    //     +  a_28p_a_29p_a_30p_a_31p_vreg.vec * b_p28_b_p29_b_p30_b_p31_vreg.vec;
    c.vec = _mm256_fmadd_pd(a_28p_a_29p_a_30p_a_31p_vreg.vec, b_p28_b_p29_b_p30_b_p31_vreg.vec,
            _mm256_fmadd_pd(a_24p_a_25p_a_26p_a_27p_vreg.vec, b_p24_b_p25_b_p26_b_p27_vreg.vec,
            _mm256_fmadd_pd(a_20p_a_21p_a_22p_a_23p_vreg.vec, b_p20_b_p21_b_p22_b_p23_vreg.vec,
            _mm256_fmadd_pd(a_16p_a_17p_a_18p_a_19p_vreg.vec, b_p16_b_p17_b_p18_b_p19_vreg.vec,
            _mm256_fmadd_pd(a_12p_a_13p_a_14p_a_15p_vreg.vec, b_p12_b_p13_b_p14_b_p15_vreg.vec,
            _mm256_fmadd_pd(a_8p_a_9p_a_10p_a_11p_vreg.vec, b_p8_b_p9_b_p10_b_p11_vreg.vec,
            _mm256_fmadd_pd(a_4p_a_5p_a_6p_a_7p_vreg.vec, b_p4_b_p5_b_p6_b_p7_vreg.vec,
            _mm256_fmadd_pd(a_0p_a_1p_a_2p_a_3p_vreg.vec, b_p0_b_p1_b_p2_b_p3_vreg.vec, c.vec))))))));
  return;
}

template<>
inline void ElementMul5x21_fma(double *a, double *b, v_t<double> &c_0, v_t<double> &c_1, v_t<double> &c_2, v_t<double> &c_3, v_t<double> &c_4){
  v_t<double>
    a_0p_a_1p_a_2p_vreg,   a_3p_a_4p_a_5p_vreg,   a_6p_a_7p_a_8p_vreg,   a_9p_a_10p_a_11p_vreg,
    a_12p_a_13p_a_14p_vreg,   a_15p_a_16p_a_17p_vreg,   a_18p_a_19p_a_20p_vreg,
    b_p0_b_p1_b_p2_vreg,   b_p3_b_p4_b_p5_vreg,   b_p6_b_p7_b_p8_vreg;

    a_0p_a_1p_a_2p_vreg.vec = _mm256_loadu_pd((double *) a);
    a_3p_a_4p_a_5p_vreg.vec = _mm256_loadu_pd((double *) a + 3);
    a_6p_a_7p_a_8p_vreg.vec = _mm256_loadu_pd((double *) a + 6);
    a_9p_a_10p_a_11p_vreg.vec = _mm256_loadu_pd((double *) a + 9);
    a_12p_a_13p_a_14p_vreg.vec = _mm256_loadu_pd((double *) a + 12);
    a_15p_a_16p_a_17p_vreg.vec = _mm256_loadu_pd((double *) a + 15);
    a_18p_a_19p_a_20p_vreg.vec = _mm256_loadu_pd((double *) a + 18);

    b_p0_b_p1_b_p2_vreg.vec = _mm256_loadu_pd((double *) b);
    b_p3_b_p4_b_p5_vreg.vec = _mm256_loadu_pd((double *) b + 3);
    b_p6_b_p7_b_p8_vreg.vec = _mm256_loadu_pd((double *) b + 6);

    // c_0.vec += a_0p_a_1p_a_2p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
    //     +  a_3p_a_4p_a_5p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
    //     +  a_6p_a_7p_a_8p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;
    c_0.vec = _mm256_fmadd_pd(a_6p_a_7p_a_8p_vreg.vec, b_p6_b_p7_b_p8_vreg.vec,
              _mm256_fmadd_pd(a_3p_a_4p_a_5p_vreg.vec, b_p3_b_p4_b_p5_vreg.vec,
              _mm256_fmadd_pd(a_0p_a_1p_a_2p_vreg.vec, b_p0_b_p1_b_p2_vreg.vec, c_0.vec)));

    // c_1.vec += a_3p_a_4p_a_5p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
    //     +  a_6p_a_7p_a_8p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
    //     +  a_9p_a_10p_a_11p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;
    c_1.vec = _mm256_fmadd_pd(a_9p_a_10p_a_11p_vreg.vec, b_p6_b_p7_b_p8_vreg.vec,
              _mm256_fmadd_pd(a_6p_a_7p_a_8p_vreg.vec, b_p3_b_p4_b_p5_vreg.vec,
              _mm256_fmadd_pd(a_3p_a_4p_a_5p_vreg.vec, b_p0_b_p1_b_p2_vreg.vec, c_1.vec)));

    // c_2.vec += a_6p_a_7p_a_8p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
    //     +  a_9p_a_10p_a_11p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
    //     +  a_12p_a_13p_a_14p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;
    c_2.vec = _mm256_fmadd_pd(a_12p_a_13p_a_14p_vreg.vec, b_p6_b_p7_b_p8_vreg.vec,
              _mm256_fmadd_pd(a_9p_a_10p_a_11p_vreg.vec, b_p3_b_p4_b_p5_vreg.vec,
              _mm256_fmadd_pd(a_6p_a_7p_a_8p_vreg.vec, b_p0_b_p1_b_p2_vreg.vec, c_2.vec)));

    // c_3.vec += a_9p_a_10p_a_11p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
    //     +  a_12p_a_13p_a_14p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
    //     +  a_15p_a_16p_a_17p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;
    c_3.vec = _mm256_fmadd_pd(a_15p_a_16p_a_17p_vreg.vec, b_p6_b_p7_b_p8_vreg.vec,
              _mm256_fmadd_pd(a_12p_a_13p_a_14p_vreg.vec, b_p3_b_p4_b_p5_vreg.vec,
              _mm256_fmadd_pd(a_9p_a_10p_a_11p_vreg.vec, b_p0_b_p1_b_p2_vreg.vec, c_3.vec)));

    // c_4.vec += a_12p_a_13p_a_14p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
    //     +  a_15p_a_16p_a_17p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
    //     +  a_18p_a_19p_a_20p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;
    c_4.vec = _mm256_fmadd_pd(a_18p_a_19p_a_20p_vreg.vec, b_p6_b_p7_b_p8_vreg.vec,
              _mm256_fmadd_pd(a_15p_a_16p_a_17p_vreg.vec, b_p3_b_p4_b_p5_vreg.vec,
              _mm256_fmadd_pd(a_12p_a_13p_a_14p_vreg.vec, b_p0_b_p1_b_p2_vreg.vec, c_4.vec)));
    return;
}

template<>
inline void ElementMulWin_S(size_t k, float *a, float *b, float *c){
  size_t p, q, t, e, w, r, u, s, i, j, m, n, v;
  p = q  = t = r = s = 0;
  p = k / 64;
  r = k % 64;
  q = r / 32;
  w = r % 32;
  e = w / 16;
  u = w % 16;
  t = u / 8;
  s = u % 8;

  v_t_s<float>
    c_0_c_0_c_0_c_0_vreg;
    c_0_c_0_c_0_c_0_vreg.vec = _mm256_setzero_ps(); 
    float *a_to = a;
    float *b_to = b;

  for(i = 0; i < p; ++i){
    ElementMul1x64_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 64;
    b_to += 64;
  }

  for(j = 0; j < q; ++j){
    ElementMul1x32_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 32;
    b_to += 32;
  }
  
  for(m = 0; m < e; ++m){
    ElementMul1x16_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 16;
    b_to += 16;
  }

  for(n = 0; n < t; ++n){
    ElementMul1x8_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 8;
    b_to += 8;
  }
  
  float remained = 0.0;
  for(v = 0; v < s; ++v){
    ElementMul1x1(a_to, b_to, remained);
    a_to++;
    b_to++;
  }
    *c += c_0_c_0_c_0_c_0_vreg.value[0] + c_0_c_0_c_0_c_0_vreg.value[1] + c_0_c_0_c_0_c_0_vreg.value[2] + c_0_c_0_c_0_c_0_vreg.value[3]
        + c_0_c_0_c_0_c_0_vreg.value[4] + c_0_c_0_c_0_c_0_vreg.value[5] + c_0_c_0_c_0_c_0_vreg.value[6] + c_0_c_0_c_0_c_0_vreg.value[7] + remained;
  return;
}

template<>
inline void ElementMulWin_S(size_t k, double *a, double *b, double *c){
  size_t p, q, t, e, w, r, u, s, i, j, m, n, v;
  p = q  = t = r = s = 0;
  p = k / 32;
  r = k % 32;
  q = r / 16;
  w = r % 16;
  e = w / 8;
  u = w % 8;
  t = u / 4;
  s = u % 4;

  v_t<double>
    c_0_c_0_c_0_c_0_vreg;
    c_0_c_0_c_0_c_0_vreg.vec = _mm256_setzero_pd(); 
    double *a_to = a;
    double *b_to = b;

  for(i = 0; i < p; ++i){
    ElementMul1x32_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 32;
    b_to += 32;
  }

  for(j = 0; j < q; ++j){
    ElementMul1x16_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 16;
    b_to += 16;
  }
  
  for(m = 0; m < e; ++m){
    ElementMul1x8_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 8;
    b_to += 8;
  }

  for(n = 0; n < t; ++n){
    ElementMul1x4_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to += 4;
    b_to += 4;
  }
  
  double remained = 0.0;
  for(v = 0; v < s; ++v){
    ElementMul1x1(a_to, b_to, remained);
    a_to++;
    b_to++;
  }

  
    *c += c_0_c_0_c_0_c_0_vreg.value[0] + c_0_c_0_c_0_c_0_vreg.value[1] + c_0_c_0_c_0_c_0_vreg.value[2] + c_0_c_0_c_0_c_0_vreg.value[3] + remained;
  return;
}

template<>
inline void ElementMulWin_M(size_t k, size_t n, double *a, double *b, double *c){
  if(n == 5){
  v_t<double>
    c_0_c_0_c_0_c_0_vreg, c_1_c_1_c_1_c_1_vreg, c_2_c_2_c_2_c_2_vreg,
    c_3_c_3_c_3_c_3_vreg, c_4_c_4_c_4_c_4_vreg;
    c_0_c_0_c_0_c_0_vreg.vec = _mm256_setzero_pd(); 
    c_1_c_1_c_1_c_1_vreg.vec = _mm256_setzero_pd();
    c_2_c_2_c_2_c_2_vreg.vec = _mm256_setzero_pd(); 
    c_3_c_3_c_3_c_3_vreg.vec = _mm256_setzero_pd();
    c_4_c_4_c_4_c_4_vreg.vec = _mm256_setzero_pd(); 
 
    double *a_to = a;
    double *b_to = b;
    ElementMul5x21_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg, c_1_c_1_c_1_c_1_vreg,
                   c_2_c_2_c_2_c_2_vreg, c_3_c_3_c_3_c_3_vreg, c_4_c_4_c_4_c_4_vreg);
    *c += c_0_c_0_c_0_c_0_vreg.value[0] + c_0_c_0_c_0_c_0_vreg.value[1] + c_0_c_0_c_0_c_0_vreg.value[2];
    *(c + 1) += c_1_c_1_c_1_c_1_vreg.value[0] + c_1_c_1_c_1_c_1_vreg.value[1] + c_1_c_1_c_1_c_1_vreg.value[2];
    *(c + 2) += c_2_c_2_c_2_c_2_vreg.value[0] + c_2_c_2_c_2_c_2_vreg.value[1] + c_2_c_2_c_2_c_2_vreg.value[2];
    *(c + 3) += c_3_c_3_c_3_c_3_vreg.value[0] + c_3_c_3_c_3_c_3_vreg.value[1] + c_3_c_3_c_3_c_3_vreg.value[2];
    *(c + 4) += c_4_c_4_c_4_c_4_vreg.value[0] + c_4_c_4_c_4_c_4_vreg.value[1] + c_4_c_4_c_4_c_4_vreg.value[2];
  }
  return;
}

template<>
inline void ElementMulWin_col(float *a, float *b, size_t *dims_b, float *c, size_t *dims_c, size_t stride){
  size_t i; 
  size_t w = dims_c[3];
  size_t w_w = dims_b[3];
  size_t h_w = dims_b[2];
  size_t k = w_w * h_w; 
  for(i = 0; i < w; ++i){
    float *a_to = a + i * h_w * stride;
    float *c_to = c + i;
    ElementMulWin_S(k, a_to, b, c_to);
  }
  return;
}

template<>
inline void ElementMulWin_col(double *a, double *b, size_t *dims_b, double *c, size_t *dims_c, size_t stride){
  size_t i, j;
  size_t w = dims_c[3];
  size_t w_w = dims_b[3];
  size_t h_w = dims_b[2];
  size_t k = w_w * h_w;
  size_t win_to_vec = k / 16;

  if(win_to_vec >=1||stride >1)
  for(i = 0; i < w; ++i){
    double *a_to = a + i * h_w * stride;
    double *c_to = c + i;
    ElementMulWin_S(k, a_to, b, c_to);
  }
  else{
    size_t num_vec = h_w % 4 == 0 ? h_w/4 : h_w/4+1;
    size_t s = (16 - num_vec * w_w * 2 - 1)/(num_vec + 1) + 1;
    size_t q = w / s;
    size_t dataType = w % s;
    
    for(i = 0; i < q; ++i){
      double *a_to = a + i * s * h_w;
      double *c_to = c + i * s;
      ElementMulWin_M(k, s, a_to, b, c_to);
    }
      double *a_to_ = a + q * s * h_w;
      double *c_to_ = c + q * s;
    for(j = 0; j < dataType; ++j){
      double *a_to_S = a_to_ + j * h_w;
      double *c_to_S = c_to_ + j;
      ElementMulWin_S(k, a_to_S, b, c_to_S);
    }
  }
  return;
}

template<typename dataType>
void ElementMulWin_col_row(dataType *a, size_t *dims_a,  dataType *b, size_t *dims_b, dataType *c, size_t *dims_c, size_t stride){
  size_t i;
  size_t h = dims_c[2];
  size_t w_o = dims_c[3];
  size_t h_w = dims_b[2]; 
  size_t w_i = dims_a[3];
  size_t a_i = h_w * w_i;

  for(i = 0; i < h; ++i){
    dataType *a_to = a + i * a_i;
    dataType *c_to = c + i * w_o;
    ElementMulWin_col(a_to, b, dims_b, c_to, dims_c, stride);
  }
  return;
}

template<typename dataType>
void IM2WIN_CONV_SIMD(dataType *const a, dataType *const b, dataType *const c, size_t *const dims_a, size_t *const dims_b, size_t *const dims_c){
  size_t b_o = dims_c[0];
  size_t c_o = dims_c[1];
  size_t h_o = dims_c[2];

  size_t w_o = dims_c[3];
  size_t c_i = dims_b[1];
  size_t h_w = dims_b[2];
  size_t w_w = dims_b[3];
  size_t w_i = dims_a[3];
  size_t stride = (w_i - w_w) / (w_o - 1);

  size_t a_l = h_w * w_i * h_o;
  size_t a_i = a_l * c_i;
  size_t b_l = h_w * w_w;
  size_t b_j = b_l * c_i;
  size_t c_j = h_o * w_o;
  size_t c_i_ = c_j * c_o;

#pragma omp parallel for schedule(dynamic)
  for(size_t i = 0; i < b_o; ++i){
    size_t ia = i * a_i;
    size_t ic = i * c_i_;
    for(size_t j = 0; j < c_o; ++j){
      size_t jb = j * b_j;
      dataType *c_to = c + j * c_j + ic;
      for(size_t l = 0; l < c_i; ++l){
        dataType *a_to = a + l * a_l + ia;
        dataType *b_to = b + l * b_l + jb;
        ElementMulWin_col_row(a_to, dims_a, b_to, dims_b, c_to, dims_c, stride);
      }
    }
  }
  return;
}

#endif