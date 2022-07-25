#ifndef _IM2WINSIMD_
#define _IM2WINSIMD_

#include <omp.h>
#include <immintrin.h>  // AVX2

template<typename T>
union v_t_fma{
  __m128d vec;
  T value[128 / sizeof(T)];
};

template<typename T>
union v_t_s_fma{
  __m128 vec;
  T value[128 / sizeof(T)];
};

template<typename T>
void ElementMul1x1_fma(T *a, T *b, v_t_fma<T> &c);

template<typename T>
void ElementMul1x1_fma(T *a, T *b, v_t_s_fma<T> &c);

template<typename T>
void ElementMul1x2_fma(T *a, T *b, v_t_fma<T> &c);

template<typename T>
void ElementMul1x4_fma(T *a, T *b, v_t_s_fma<T> &c);

template<typename T>
void ElementMul1x8_fma(T *a, T *b, v_t_s_fma<T> &c);

template<typename T>
void ElementMul1x8_fma(T *a, T *b, v_t_fma<T> &c);

template<typename T>
void ElementMul1x16_fma(T *a, T *b, v_t_s_fma<T> &c);

template<typename T>
void ElementMul1x32_fma(T *a, T *b, v_t_s_fma<T> &c);

template<typename T>
void ElementMul1x16_fma(T *a, T *b, v_t_fma<T> &c);

template<typename T>
void ElementMul5x21_fma(T *a, T *b, v_t_s_fma<T> &c_0, v_t_s_fma<T> &c_1, v_t_s_fma<T> &c_2, v_t_s_fma<T> &c_3, v_t_s_fma<T> &c_4);

template<typename T>
void ElementMul2x12_fma(T *a, T *b, v_t_fma<T> &c_0, v_t_fma<T> &c_1);

template<typename T>
void ElementMul6x14_fma(T *a, T *b, v_t_fma<T> &c_0, v_t_fma<T> &c_1, v_t_fma<T> &c_2,
                            v_t_fma<T> &c_3, v_t_fma<T> &c_4, v_t_fma<T> &c_5);

template<typename T>
void ElementMulWin_S_fma(size_t k, T *a, T *b, T *c);

template<typename T>
void ElementMulWin_M_fma(size_t k, size_t n, T *a, T *b, T *c);

template<typename T>
void ElElementMulWin_col_fma(T *a, T *b, size_t *dims_b, T *c, size_t *dims_c, size_t stride);

template<typename T>
void ElementMulWin_col_row_fma(T *a, size_t *dims_a, T *b, size_t *dims_b, T *c, size_t *dims_c, size_t stride);

template<typename T>
void IM2WIN_CONV_SIMD(T *const a, T *const b, T *const c, size_t *const dims_a, size_t *const dims_b, size_t *const dims_c);

template<typename T>
void Init_C_fma(T *const c, size_t *const dims_c);

template<>
inline void ElementMul1x1_fma(double *a, double *b, v_t_fma<double> &c)
{
  v_t_fma<double>
    a_0p_vreg,   b_p0_vreg;

    a_0p_vreg.vec = _mm_load_sd((double *) a); 
    b_p0_vreg.vec = _mm_load_sd((double *) b);
    //c.vec += a_0p_vreg.vec * b_p0_vreg.vec;
    c.vec = _mm_fmadd_pd(a_0p_vreg.vec, b_p0_vreg.vec, c.vec);
  }

template<>
inline void ElementMul1x1_fma(float *a, float *b, v_t_s_fma<float> &c)
{
  v_t_s_fma<float>
    a_0p_vreg,   b_p0_vreg;

    a_0p_vreg.vec = _mm_load_ss((float *) a); 
    b_p0_vreg.vec = _mm_load_ss((float *) b);
 
    c.vec += a_0p_vreg.vec * b_p0_vreg.vec;
  }

template<>
inline void ElementMul1x4_fma(float *a, float *b, v_t_s_fma<float> &c){
  v_t_s_fma<float>
    a_0p_a_1p_a_2p_a_3p_vreg,   b_p0_b_p1_b_p2_b_p3_vreg;

    a_0p_a_1p_a_2p_a_3p_vreg.vec = _mm_loadu_ps((float *) a); 
    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm_loadu_ps((float *) b);

    c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec;
}

template<>
inline void ElementMul1x2_fma(double *a, double *b, v_t_fma<double> &c){
  v_t_fma<double>
    a_0p_a_1p_vreg,   b_p0_b_p1_vreg;

    a_0p_a_1p_vreg.vec = _mm_loadu_pd((double *) a); 
    b_p0_b_p1_vreg.vec = _mm_loadu_pd((double *) b);

    //c.vec += a_0p_a_1p_vreg.vec * b_p0_b_p1_vreg.vec;
    c.vec = _mm_fmadd_pd(a_0p_a_1p_vreg.vec, b_p0_b_p1_vreg.vec, c.vec);
}

template<>
inline void ElementMul1x8_fma(float *a, float *b, v_t_s_fma<float> &c){
  v_t_s_fma<float>
    a_0p_a_1p_a_2p_a_3p_vreg,   a_4p_a_5p_a_6p_a_7p_vreg,
    b_p0_b_p1_b_p2_b_p3_vreg,   b_p4_b_p5_b_p6_b_p7_vreg;

    a_0p_a_1p_a_2p_a_3p_vreg.vec = _mm_loadu_ps((float *) a); 
    a_4p_a_5p_a_6p_a_7p_vreg.vec = _mm_loadu_ps((float *) a + 4); 
    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm_loadu_ps((float *) b);
    b_p4_b_p5_b_p6_b_p7_vreg.vec = _mm_loadu_ps((float *) b + 4);

    c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec
            +a_4p_a_5p_a_6p_a_7p_vreg.vec * b_p4_b_p5_b_p6_b_p7_vreg.vec;
}

template<>
inline void ElementMul1x16_fma(float *a, float *b, v_t_s_fma<float> &c){
  v_t_s_fma<float>
    a_0p_a_1p_a_2P_a_3p_vreg,   a_4p_a_5p_a_6P_a_7p_vreg,   a_8p_a_9p_a_10P_a_11p_vreg,   a_12p_a_13p_a_14P_a_15p_vreg,
    b_p0_b_p1_b_p2_b_p3_vreg,   b_p4_b_p5_b_p6_b_p7_vreg,   b_p8_b_p9_b_p10_b_p11_vreg,   b_p12_b_p13_b_p14_b_p15_vreg;

    a_0p_a_1p_a_2P_a_3p_vreg.vec = _mm_loadu_ps((float *) a);
    a_4p_a_5p_a_6P_a_7p_vreg.vec = _mm_loadu_ps((float *) a + 4);
    a_8p_a_9p_a_10P_a_11p_vreg.vec = _mm_loadu_ps((float *) a + 8);
    a_12p_a_13p_a_14P_a_15p_vreg.vec = _mm_loadu_ps((float *) a + 12); 
    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm_loadu_ps((float *) b);
    b_p4_b_p5_b_p6_b_p7_vreg.vec = _mm_loadu_ps((float *) b + 4);
    b_p8_b_p9_b_p10_b_p11_vreg.vec = _mm_loadu_ps((float *) b + 8);
    b_p12_b_p13_b_p14_b_p15_vreg.vec = _mm_loadu_ps((float *) b + 12);

    c.vec += a_0p_a_1p_a_2P_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec
        +  a_4p_a_5p_a_6P_a_7p_vreg.vec * b_p4_b_p5_b_p6_b_p7_vreg.vec
        +  a_8p_a_9p_a_10P_a_11p_vreg.vec * b_p8_b_p9_b_p10_b_p11_vreg.vec
        +  a_12p_a_13p_a_14P_a_15p_vreg.vec * b_p12_b_p13_b_p14_b_p15_vreg.vec;
}

template<>
inline void ElementMul1x8_fma(double *a, double *b, v_t_fma<double> &c){
  v_t_fma<double>
    a_0p_a_1p_vreg,   a_2p_a_3p_vreg,   a_4p_a_5p_vreg,   a_6p_a_7p_vreg,
    b_p0_b_p1_vreg,   b_p2_b_p3_vreg,   b_p4_b_p5_vreg,   b_p6_b_p7_vreg;

    a_0p_a_1p_vreg.vec = _mm_loadu_pd((double *) a);
    a_2p_a_3p_vreg.vec = _mm_loadu_pd((double *) a + 2);
    a_4p_a_5p_vreg.vec = _mm_loadu_pd((double *) a + 4);
    a_6p_a_7p_vreg.vec = _mm_loadu_pd((double *) a + 6); 
    b_p0_b_p1_vreg.vec = _mm_loadu_pd((double *) b);
    b_p2_b_p3_vreg.vec = _mm_loadu_pd((double *) b + 2);
    b_p4_b_p5_vreg.vec = _mm_loadu_pd((double *) b + 4);
    b_p6_b_p7_vreg.vec = _mm_loadu_pd((double *) b + 6);
/*
    c.vec += a_0p_a_1p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_2p_a_3p_vreg.vec * b_p2_b_p3_vreg.vec
        +  a_4p_a_5p_vreg.vec * b_p4_b_p5_vreg.vec
        +  a_6p_a_7p_vreg.vec * b_p6_b_p7_vreg.vec;
*/
    c.vec = _mm_fmadd_pd(a_0p_a_1p_vreg.vec, b_p0_b_p1_vreg.vec,
            _mm_fmadd_pd(a_2p_a_3p_vreg.vec, b_p2_b_p3_vreg.vec,
            _mm_fmadd_pd(a_4p_a_5p_vreg.vec, b_p4_b_p5_vreg.vec, 
            _mm_fmadd_pd(a_6p_a_7p_vreg.vec, b_p6_b_p7_vreg.vec, c.vec))));
}

template<>
inline void ElementMul1x32_fma(float *a, float *b, v_t_s_fma<float> &c){
  v_t_s_fma<float>
    a_0p_a_1p_a_2p_a_3p_vreg,   a_4p_a_5p_a_6p_a_7p_vreg,   a_8p_a_9p_a_10p_a_11p_vreg,   a_12p_a_13p_a_14p_a_15p_vreg,
    a_16p_a_17p_a_18p_a_19p_vreg,   a_20p_a_21p_a_22p_a_23p_vreg,   a_24p_a_25p_a_26p_a_27p_vreg,   a_28p_a_29p_a_30p_a_31p_vreg,
    b_p0_b_p1_b_p2_b_p3_vreg,   b_p4_b_p5_b_p6_b_p7_vreg,   b_p8_b_p9_b_p10_b_p11_vreg,   b_p12_b_p13_b_p14_b_p15_vreg,
    b_p16_b_p17_b_p18_b_p19_vreg,   b_p20_b_p21_b_p22_b_p23_vreg,   b_p24_b_p25_b_p26_b_p27_vreg,   b_p28_b_p29_b_p30_b_p31_vreg;

    a_0p_a_1p_a_2p_a_3p_vreg.vec = _mm_loadu_ps((float *) a);
    a_4p_a_5p_a_6p_a_7p_vreg.vec = _mm_loadu_ps((float *) a + 4);
    a_8p_a_9p_a_10p_a_11p_vreg.vec = _mm_loadu_ps((float *) a + 8);
    a_12p_a_13p_a_14p_a_15p_vreg.vec = _mm_loadu_ps((float *) a + 12);
    a_16p_a_17p_a_18p_a_19p_vreg.vec = _mm_loadu_ps((float *) a + 16);
    a_20p_a_21p_a_22p_a_23p_vreg.vec = _mm_loadu_ps((float *) a + 20);
    a_24p_a_25p_a_26p_a_27p_vreg.vec = _mm_loadu_ps((float *) a + 24);
    a_28p_a_29p_a_30p_a_31p_vreg.vec = _mm_loadu_ps((float *) a + 28); 

    b_p0_b_p1_b_p2_b_p3_vreg.vec = _mm_loadu_ps((float *) b);
    b_p4_b_p5_b_p6_b_p7_vreg.vec = _mm_loadu_ps((float *) b + 4);
    b_p8_b_p9_b_p10_b_p11_vreg.vec = _mm_loadu_ps((float *) b + 8);
    b_p12_b_p13_b_p14_b_p15_vreg.vec = _mm_loadu_ps((float *) b + 12);
    b_p16_b_p17_b_p18_b_p19_vreg.vec = _mm_loadu_ps((float *) b + 16);
    b_p20_b_p21_b_p22_b_p23_vreg.vec = _mm_loadu_ps((float *) b + 20);
    b_p24_b_p25_b_p26_b_p27_vreg.vec = _mm_loadu_ps((float *) b + 24);
    b_p28_b_p29_b_p30_b_p31_vreg.vec = _mm_loadu_ps((float *) b + 28);

    c.vec += a_0p_a_1p_a_2p_a_3p_vreg.vec * b_p0_b_p1_b_p2_b_p3_vreg.vec
        +  a_4p_a_5p_a_6p_a_7p_vreg.vec * b_p4_b_p5_b_p6_b_p7_vreg.vec
        +  a_8p_a_9p_a_10p_a_11p_vreg.vec * b_p8_b_p9_b_p10_b_p11_vreg.vec
        +  a_12p_a_13p_a_14p_a_15p_vreg.vec * b_p12_b_p13_b_p14_b_p15_vreg.vec
        +  a_16p_a_17p_a_18p_a_19p_vreg.vec * b_p16_b_p17_b_p18_b_p19_vreg.vec
        +  a_20p_a_21p_a_22p_a_23p_vreg.vec * b_p20_b_p21_b_p22_b_p23_vreg.vec
        +  a_24p_a_25p_a_26p_a_27p_vreg.vec * b_p24_b_p25_b_p26_b_p27_vreg.vec
        +  a_28p_a_29p_a_30p_a_31p_vreg.vec * b_p28_b_p29_b_p30_b_p31_vreg.vec;
}

template<>
inline void ElementMul1x16_fma(double *a, double *b, v_t_fma<double> &c){
  v_t_fma<double>
    a_0p_a_1p_vreg,   a_2p_a_3p_vreg,   a_4p_a_5p_vreg,   a_6p_a_7p_vreg,
    a_8p_a_9p_vreg,   a_10p_a_11p_vreg,   a_12p_a_13p_vreg,   a_14p_a_15p_vreg,
    b_p0_b_p1_vreg,   b_p2_b_p3_vreg,   b_p4_b_p5_vreg,   b_p6_b_p7_vreg,
    b_p8_b_p9_vreg,   b_p10_b_p11_vreg,   b_p12_b_p13_vreg,   b_p14_b_p15_vreg;

    a_0p_a_1p_vreg.vec = _mm_loadu_pd((double *) a);
    a_2p_a_3p_vreg.vec = _mm_loadu_pd((double *) a + 2);
    a_4p_a_5p_vreg.vec = _mm_loadu_pd((double *) a + 4);
    a_6p_a_7p_vreg.vec = _mm_loadu_pd((double *) a + 6);
    a_8p_a_9p_vreg.vec = _mm_loadu_pd((double *) a + 8);
    a_10p_a_11p_vreg.vec = _mm_loadu_pd((double *) a + 10);
    a_12p_a_13p_vreg.vec = _mm_loadu_pd((double *) a + 12);
    a_14p_a_15p_vreg.vec = _mm_loadu_pd((double *) a + 14); 

    b_p0_b_p1_vreg.vec = _mm_loadu_pd((double *) b);
    b_p2_b_p3_vreg.vec = _mm_loadu_pd((double *) b + 2);
    b_p4_b_p5_vreg.vec = _mm_loadu_pd((double *) b + 4);
    b_p6_b_p7_vreg.vec = _mm_loadu_pd((double *) b + 6);
    b_p8_b_p9_vreg.vec = _mm_loadu_pd((double *) b + 8);
    b_p10_b_p11_vreg.vec = _mm_loadu_pd((double *) b + 10);
    b_p12_b_p13_vreg.vec = _mm_loadu_pd((double *) b + 12);
    b_p14_b_p15_vreg.vec = _mm_loadu_pd((double *) b + 14);
/*
    c.vec += a_0p_a_1p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_2p_a_3p_vreg.vec * b_p2_b_p3_vreg.vec
        +  a_4p_a_5p_vreg.vec * b_p4_b_p5_vreg.vec
        +  a_6p_a_7p_vreg.vec * b_p6_b_p7_vreg.vec
        +  a_8p_a_9p_vreg.vec * b_p8_b_p9_vreg.vec
        +  a_10p_a_11p_vreg.vec * b_p10_b_p11_vreg.vec
        +  a_12p_a_13p_vreg.vec * b_p12_b_p13_vreg.vec
        +  a_14p_a_15p_vreg.vec * b_p14_b_p15_vreg.vec;
*/
    c.vec = _mm_fmadd_pd(a_0p_a_1p_vreg.vec, b_p0_b_p1_vreg.vec, 
            _mm_fmadd_pd(a_2p_a_3p_vreg.vec, b_p2_b_p3_vreg.vec, 
            _mm_fmadd_pd(a_4p_a_5p_vreg.vec, b_p4_b_p5_vreg.vec, 
            _mm_fmadd_pd(a_6p_a_7p_vreg.vec, b_p6_b_p7_vreg.vec, 
            _mm_fmadd_pd(a_8p_a_9p_vreg.vec, b_p8_b_p9_vreg.vec, 
            _mm_fmadd_pd(a_10p_a_11p_vreg.vec, b_p10_b_p11_vreg.vec, 
            _mm_fmadd_pd(a_12p_a_13p_vreg.vec, b_p12_b_p13_vreg.vec, 
            _mm_fmadd_pd(a_14p_a_15p_vreg.vec, b_p14_b_p15_vreg.vec, c.vec))))))));
}

template<>
inline void ElementMul5x21_fma(float *a, float *b, v_t_s_fma<float> &c_0, v_t_s_fma<float> &c_1, v_t_s_fma<float> &c_2, v_t_s_fma<float> &c_3, v_t_s_fma<float> &c_4){
  v_t_s_fma<float>
    a_0p_a_1p_a_2p_vreg,   a_3p_a_4p_a_5p_vreg,   a_6p_a_7p_a_8p_vreg,   a_9p_a_10p_a_11p_vreg,
    a_12p_a_13p_a_14p_vreg,   a_15p_a_16p_a_17p_vreg,   a_18p_a_19p_a_20p_vreg,
    b_p0_b_p1_b_p2_vreg,   b_p3_b_p4_b_p5_vreg,   b_p6_b_p7_b_p8_vreg;

    a_0p_a_1p_a_2p_vreg.vec = _mm_loadu_ps((float *) a);
    a_3p_a_4p_a_5p_vreg.vec = _mm_loadu_ps((float *) a + 3);
    a_6p_a_7p_a_8p_vreg.vec = _mm_loadu_ps((float *) a + 6);
    a_9p_a_10p_a_11p_vreg.vec = _mm_loadu_ps((float *) a + 9);
    a_12p_a_13p_a_14p_vreg.vec = _mm_loadu_ps((float *) a + 12);
    a_15p_a_16p_a_17p_vreg.vec = _mm_loadu_ps((float *) a + 15);
    a_18p_a_19p_a_20p_vreg.vec = _mm_loadu_ps((float *) a + 18);

    b_p0_b_p1_b_p2_vreg.vec = _mm_loadu_ps((float *) b);
    b_p3_b_p4_b_p5_vreg.vec = _mm_loadu_ps((float *) b + 3);
    b_p6_b_p7_b_p8_vreg.vec = _mm_loadu_ps((float *) b + 6);

    c_0.vec += a_0p_a_1p_a_2p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
        +  a_3p_a_4p_a_5p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
        +  a_6p_a_7p_a_8p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;
    
    c_1.vec += a_3p_a_4p_a_5p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
        +  a_6p_a_7p_a_8p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
        +  a_9p_a_10p_a_11p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;

    c_2.vec += a_6p_a_7p_a_8p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
        +  a_9p_a_10p_a_11p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
        +  a_12p_a_13p_a_14p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;

    c_3.vec += a_9p_a_10p_a_11p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
        +  a_12p_a_13p_a_14p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
        +  a_15p_a_16p_a_17p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;

    c_4.vec += a_12p_a_13p_a_14p_vreg.vec * b_p0_b_p1_b_p2_vreg.vec
        +  a_15p_a_16p_a_17p_vreg.vec * b_p3_b_p4_b_p5_vreg.vec
        +  a_18p_a_19p_a_20p_vreg.vec * b_p6_b_p7_b_p8_vreg.vec;
}

template<>
inline void ElementMul2x12_fma(double *a, double *b, v_t_fma<double> &c_0, v_t_fma<double> &c_1){
  v_t_fma<double>
    a_0p_a_1p_vreg,   a_2p_a_0_vreg,   a_3p_a_4p_vreg,   a_5p_a_0_vreg,
    a_6p_a_7p_vreg,   a_8p_a_0_vreg,   a_9p_a_10p_vreg,   a_11p_a_0_vreg,
    b_p0_b_p1_vreg,   b_p2_b_0_vreg,   b_p3_b_p4_vreg,   b_p5_b_0_vreg,
    b_p6_b_p7_vreg,   b_p8_b_0_vreg;

    a_0p_a_1p_vreg.vec = _mm_loadu_pd((double *) a);
    a_2p_a_0_vreg.vec = _mm_load_sd((double *) a + 2);
    a_3p_a_4p_vreg.vec = _mm_loadu_pd((double *) a + 3);
    a_5p_a_0_vreg.vec = _mm_load_sd((double *) a + 5);
    a_6p_a_7p_vreg.vec = _mm_loadu_pd((double *) a + 6);
    a_8p_a_0_vreg.vec = _mm_load_sd((double *) a + 8);
    a_9p_a_10p_vreg.vec = _mm_loadu_pd((double *) a + 9);
    a_11p_a_0_vreg.vec = _mm_load_sd((double *) a + 11); 

    b_p0_b_p1_vreg.vec = _mm_loadu_pd((double *) b);
    b_p2_b_0_vreg.vec = _mm_load_sd((double *) b + 2);
    b_p3_b_p4_vreg.vec = _mm_loadu_pd((double *) b + 3);
    b_p5_b_0_vreg.vec = _mm_load_sd((double *) b + 5);
    b_p6_b_p7_vreg.vec = _mm_loadu_pd((double *) b + 6);
    b_p8_b_0_vreg.vec = _mm_load_sd((double *) b + 8);
/*
    c_0.vec += a_0p_a_1p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_2p_a_0_vreg.vec * b_p2_b_0_vreg.vec
        +  a_3p_a_4p_vreg.vec * b_p3_b_p4_vreg.vec
        +  a_5p_a_0_vreg.vec * b_p5_b_0_vreg.vec
        +  a_6p_a_7p_vreg.vec * b_p6_b_p7_vreg.vec
        +  a_8p_a_0_vreg.vec * b_p8_b_0_vreg.vec;
*/
    c_0.vec = _mm_fmadd_pd(a_0p_a_1p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_2p_a_0_vreg.vec, b_p2_b_0_vreg.vec, 
              _mm_fmadd_pd(a_3p_a_4p_vreg.vec, b_p3_b_p4_vreg.vec, 
              _mm_fmadd_pd(a_5p_a_0_vreg.vec, b_p5_b_0_vreg.vec,  
              _mm_fmadd_pd(a_6p_a_7p_vreg.vec, b_p6_b_p7_vreg.vec, 
              _mm_fmadd_pd(a_8p_a_0_vreg.vec, b_p8_b_0_vreg.vec, c_0.vec))))));
/*
    c_1.vec += a_3p_a_4p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_5p_a_0_vreg.vec * b_p2_b_0_vreg.vec
        +  a_6p_a_7p_vreg.vec * b_p3_b_p4_vreg.vec
        +  a_8p_a_0_vreg.vec * b_p5_b_0_vreg.vec
        +  a_9p_a_10p_vreg.vec * b_p6_b_p7_vreg.vec
        +  a_11p_a_0_vreg.vec * b_p8_b_0_vreg.vec;
*/
    c_1.vec = _mm_fmadd_pd(a_3p_a_4p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_5p_a_0_vreg.vec, b_p2_b_0_vreg.vec, 
              _mm_fmadd_pd(a_6p_a_7p_vreg.vec, b_p3_b_p4_vreg.vec, 
              _mm_fmadd_pd(a_8p_a_0_vreg.vec, b_p5_b_0_vreg.vec,  
              _mm_fmadd_pd(a_9p_a_10p_vreg.vec, b_p6_b_p7_vreg.vec, 
              _mm_fmadd_pd(a_11p_a_0_vreg.vec, b_p8_b_0_vreg.vec, c_1.vec))))));
}

template<>
inline void ElementMul6x14_fma(double *a, double *b, v_t_fma<double> &c_0, v_t_fma<double> &c_1, v_t_fma<double> &c_2,
                            v_t_fma<double> &c_3, v_t_fma<double> &c_4, v_t_fma<double> &c_5){
  v_t_fma<double>
    a_0p_a_1p_vreg,   a_2p_a_3p_vreg,   a_4p_a_5p_vreg,   a_6p_a_7p_vreg,
    a_8p_a_9p_vreg,   a_10p_a_11p_vreg,   a_12p_a_13p_vreg,
    b_p0_b_p1_vreg,   b_p2_b_p3_vreg;

    a_0p_a_1p_vreg.vec = _mm_loadu_pd((double *) a);
    a_2p_a_3p_vreg.vec = _mm_loadu_pd((double *) a + 2);
    a_4p_a_5p_vreg.vec = _mm_loadu_pd((double *) a + 4);
    a_6p_a_7p_vreg.vec = _mm_loadu_pd((double *) a + 6);
    a_8p_a_9p_vreg.vec = _mm_loadu_pd((double *) a + 8);
    a_10p_a_11p_vreg.vec = _mm_loadu_pd((double *) a + 10);
    a_12p_a_13p_vreg.vec = _mm_loadu_pd((double *) a + 12);

    b_p0_b_p1_vreg.vec = _mm_loadu_pd((double *) b);
    b_p2_b_p3_vreg.vec = _mm_loadu_pd((double *) b + 2);

/*
    c_0.vec += a_0p_a_1p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_2p_a_3p_vreg.vec * b_p2_b_p3_vreg.vec;
    c_1.vec += a_2p_a_3p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_4p_a_5p_vreg.vec * b_p2_b_p3_vreg.vec;
    c_2.vec += a_4p_a_5p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_6p_a_7p_vreg.vec * b_p2_b_p3_vreg.vec;
    c_3.vec += a_6p_a_7p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_8p_a_9p_vreg.vec * b_p2_b_p3_vreg.vec;
    c_4.vec += a_8p_a_9p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_10p_a_11p_vreg.vec * b_p2_b_p3_vreg.vec;
    c_5.vec += a_10p_a_11p_vreg.vec * b_p0_b_p1_vreg.vec
        +  a_12p_a_13p_vreg.vec * b_p2_b_p3_vreg.vec;
*/
    c_0.vec = _mm_fmadd_pd(a_0p_a_1p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_2p_a_3p_vreg.vec, b_p2_b_p3_vreg.vec, c_0.vec));
    c_1.vec = _mm_fmadd_pd(a_2p_a_3p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_4p_a_5p_vreg.vec, b_p2_b_p3_vreg.vec, c_1.vec));
    c_2.vec = _mm_fmadd_pd(a_4p_a_5p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_6p_a_7p_vreg.vec, b_p2_b_p3_vreg.vec, c_2.vec));
    c_3.vec = _mm_fmadd_pd(a_6p_a_7p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_8p_a_9p_vreg.vec, b_p2_b_p3_vreg.vec, c_3.vec));
    c_4.vec = _mm_fmadd_pd(a_8p_a_9p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_10p_a_11p_vreg.vec, b_p2_b_p3_vreg.vec, c_4.vec));
    c_5.vec = _mm_fmadd_pd(a_10p_a_11p_vreg.vec, b_p0_b_p1_vreg.vec, 
              _mm_fmadd_pd(a_12p_a_13p_vreg.vec, b_p2_b_p3_vreg.vec, c_5.vec));                             
}

template<>
inline void ElementMulWin_S_fma(size_t k, float *a, float *b, float *c){
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

  v_t_s_fma<float>
    c_0_c_0_c_0_c_0_vreg;
    c_0_c_0_c_0_c_0_vreg.vec = _mm_setzero_ps(); 
    float *a_to = a;
    float *b_to = b;

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
  
  for(v = 0; v < s; ++v){
    ElementMul1x1_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg);
    a_to++;
    b_to++;
  }

  
    *c += c_0_c_0_c_0_c_0_vreg.value[0] + c_0_c_0_c_0_c_0_vreg.value[1] + c_0_c_0_c_0_c_0_vreg.value[2] + c_0_c_0_c_0_c_0_vreg.value[3];
}

template<>
inline void ElementMulWin_S_fma(size_t k, double *a, double *b, double *c){
  size_t p, q, t, w, r, s, i, j, m;
  p = q  = t = r = s = 0;
  p = k / 16;
  r = k % 16;
  q = r / 8;
  w = r % 8;
  t = w / 2;
  s = w % 2;

  v_t_fma<double>
    c_l_c_r_vreg;
    c_l_c_r_vreg.vec = _mm_setzero_pd(); 
    double *a_to = a;
    double *b_to = b;

  for(i = 0; i < p; ++i){
    ElementMul1x16_fma(a_to, b_to, c_l_c_r_vreg);
    a_to += 16;
    b_to += 16;
  }

  for(j = 0; j < q; ++j){
    ElementMul1x8_fma(a_to, b_to, c_l_c_r_vreg);
    a_to += 8;
    b_to += 8;
  }
  
  for(m = 0; m < t; ++m){
    ElementMul1x2_fma(a_to, b_to, c_l_c_r_vreg);
    a_to += 2;
    b_to += 2;
  }
  
  if(s == 1){
    ElementMul1x1_fma(a_to, b_to, c_l_c_r_vreg);
  }
  
    *c += c_l_c_r_vreg.value[0] + c_l_c_r_vreg.value[1];
}


template<>
inline void ElementMulWin_M_fma(size_t k, size_t n, float *a, float *b, float *c){
  if(n == 5){
  v_t_s_fma<float>
    c_0_c_0_c_0_c_0_vreg, c_1_c_1_c_1_c_1_vreg, c_2_c_2_c_2_c_2_vreg,
    c_3_c_3_c_3_c_3_vreg, c_4_c_4_c_4_c_4_vreg;
    c_0_c_0_c_0_c_0_vreg.vec = _mm_setzero_ps(); 
    c_1_c_1_c_1_c_1_vreg.vec = _mm_setzero_ps();
    c_2_c_2_c_2_c_2_vreg.vec = _mm_setzero_ps(); 
    c_3_c_3_c_3_c_3_vreg.vec = _mm_setzero_ps();
    c_4_c_4_c_4_c_4_vreg.vec = _mm_setzero_ps(); 
 
    float *a_to = a;
    float *b_to = b;
    ElementMul5x21_fma(a_to, b_to, c_0_c_0_c_0_c_0_vreg, c_1_c_1_c_1_c_1_vreg,
                   c_2_c_2_c_2_c_2_vreg, c_3_c_3_c_3_c_3_vreg, c_4_c_4_c_4_c_4_vreg);
    *c += c_0_c_0_c_0_c_0_vreg.value[0] + c_0_c_0_c_0_c_0_vreg.value[1] + c_0_c_0_c_0_c_0_vreg.value[2];
    *(c + 1) += c_1_c_1_c_1_c_1_vreg.value[0] + c_1_c_1_c_1_c_1_vreg.value[1] + c_1_c_1_c_1_c_1_vreg.value[2];
    *(c + 2) += c_2_c_2_c_2_c_2_vreg.value[0] + c_2_c_2_c_2_c_2_vreg.value[1] + c_2_c_2_c_2_c_2_vreg.value[2];
    *(c + 3) += c_3_c_3_c_3_c_3_vreg.value[0] + c_3_c_3_c_3_c_3_vreg.value[1] + c_3_c_3_c_3_c_3_vreg.value[2];
    *(c + 4) += c_4_c_4_c_4_c_4_vreg.value[0] + c_4_c_4_c_4_c_4_vreg.value[1] + c_4_c_4_c_4_c_4_vreg.value[2];
  }
}

template<>
inline void ElementMulWin_M_fma(size_t k, size_t n, double *a, double *b, double *c){
  if(n == 2){
  v_t_fma<double>
    c_0_c_0_vreg, c_1_c_1_vreg;
    c_0_c_0_vreg.vec = _mm_setzero_pd(); 
    c_1_c_1_vreg.vec = _mm_setzero_pd(); 
    double *a_to = a;
    double *b_to = b;
    ElementMul2x12_fma(a_to, b_to, c_0_c_0_vreg, c_1_c_1_vreg);
    *c += c_0_c_0_vreg.value[0] + c_0_c_0_vreg.value[1];
    *(c + 1) += c_1_c_1_vreg.value[0] + c_1_c_1_vreg.value[1];
  }
}

template<>
inline void ElElementMulWin_col_fma(float *a, float *b, size_t *dims_b, float *c, size_t *dims_c, size_t stride){
  size_t i, j;
  size_t w = dims_c[3];
  size_t w_w = dims_b[3];
  size_t h_w = dims_b[2];
  size_t k = w_w * h_w;
  size_t win_to_vec = k / 16;

  if(win_to_vec >=1||stride >1)
  for(i = 0; i < w; ++i){
    float *a_to = a + i * h_w * stride;
    float *c_to = c + i;
    ElementMulWin_S_fma(k, a_to, b, c_to);
  }
  else{
    size_t num_vec = h_w % 4 == 0 ? h_w/4 : h_w/4+1;
    size_t s = (16 - num_vec * w_w * 2 - 1)/(num_vec + 1) + 1;
    size_t q = w / s;
    size_t t = w % s;
    
    for(i = 0; i < q; ++i){
      float *a_to = a + i * s * h_w;
      float *c_to = c + i * s;
      ElementMulWin_M_fma(k, s, a_to, b, c_to);
    }
      float *a_to_ = a + q * s * h_w;
      float *c_to_ = c + q * s;
    for(j = 0; j < t; ++j){
      float *a_to_S = a_to_ + j * h_w;
      float *c_to_S = c_to_ + j;
      ElementMulWin_S_fma(k, a_to_S, b, c_to_S);
    }
  }
}

template<>
inline void ElElementMulWin_col_fma(double *a, double *b, size_t *dims_b, double *c, size_t *dims_c, size_t stride){
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
    ElementMulWin_S_fma(k, a_to, b, c_to);
  }
  else{
    size_t num_vec = h_w % 2 == 0 ? h_w/2 : h_w/2+1;
    size_t s = (16 - num_vec * w_w * 2 - 1)/(num_vec + 1) + 1;
    size_t q = w / s;
    size_t t = w % s;
    
    for(i = 0; i < q; ++i){
      double *a_to = a + i * s * h_w;
      double *c_to = c + i * s;
      ElementMulWin_M_fma(k, s, a_to, b, c_to);
    }
      double *a_to_ = a + q * s * h_w;
      double *c_to_ = c + q * s;
    for(j = 0; j < t; ++j){
      double *a_to_S = a_to_ + j * h_w;
      double *c_to_S = c_to_ + j;
      ElementMulWin_S_fma(k, a_to_S, b, c_to_S);
    }
  }
}

template<typename T>
void ElementMulWin_col_row_fma(T *a, size_t *dims_a,  T *b, size_t *dims_b, T *c, size_t *dims_c, size_t stride){
  size_t i;
  size_t h = dims_c[2];
  size_t w_o = dims_c[3];
  size_t h_w = dims_b[2]; 
  size_t w_i = dims_a[3];
  size_t a_i = h_w * w_i;

  for(i = 0; i < h; ++i){
    T *a_to = a + i * a_i;
    T *c_to = c + i * w_o;
    ElElementMulWin_col_fma(a_to, b, dims_b, c_to, dims_c, stride);
  }
}

template<typename T>
void Init_C_fma(T *const c, size_t *const dims_c){
  size_t b_o = dims_c[0];
  size_t c_o = dims_c[1];
  size_t h_o = dims_c[2];
  size_t w_o = dims_c[3];

#pragma omp parallel for schedule(dynamic)
  for(size_t i = 0; i < b_o; ++i){
    size_t ic = i * c_o;
    for(size_t j = 0; j < c_o; ++j){
     size_t jic = (ic + j) * h_o;
      for(size_t k = 0; k < h_o; ++k){
        size_t kjic = (jic + k) * w_o;
        for(size_t l = 0; l < w_o; ++l){
          *(c + kjic + l) = 0;
        }
      }
    }
  }  
}

template<typename T>
void IM2WIN_CONV_SIMD(T *const a, T *const b, T *const c, size_t *const dims_a, size_t *const dims_b, size_t *const dims_c){
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
  Init_C_fma(c, dims_c);
#pragma omp parallel for schedule(dynamic)
  for(size_t i = 0; i < b_o; ++i){
    size_t ia = i * a_i;
    size_t ic = i * c_i_;
    for(size_t j = 0; j < c_o; ++j){
      size_t jb = j * b_j;
      T *c_to = c + j * c_j + ic;
      for(size_t l = 0; l < c_i; ++l){
        T *a_to = a + l * a_l + ia;
        T *b_to = b + l * b_l + jb;
        ElementMulWin_col_row_fma(a_to, dims_a, b_to, dims_b, c_to, dims_c, stride);
      }
    }
  }
}

#endif