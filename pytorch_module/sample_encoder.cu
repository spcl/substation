#include "blocks.cuh"
#include "encoder.cuh"

/* #define B 4
#define S 128
#define H 8
#define P 32
#define N (H * P)
#define U (4 * N) */

struct dimB { enum { value = sizeB }; };
struct dimK { enum { value = sizeS }; };
struct dimJ { enum { value = sizeS }; };
struct dimH { enum { value = sizeH }; };
struct dimP { enum { value = sizeP }; };
struct dimN { enum { value = sizeI }; };
struct dimU { enum { value = sizeU }; };
struct dimQKV { enum { value = 3 }; };

using lX = metal::list<dimB, dimJ, dimN>;
using lK = metal::list<dimB, dimK, dimN>;
using lQ = metal::list<dimB, dimJ, dimN>;
using lV = metal::list<dimB, dimK, dimN>;
using lWKQV = metal::list<dimQKV, dimP, dimH, dimN>;
using lWK = metal::list<dimP, dimH, dimN>;
using lWQ = metal::list<dimP, dimH, dimN>;
using lWV = metal::list<dimP, dimH, dimN>;
using lKKQQVV = metal::list<dimQKV, dimP, dimH, dimB, dimJ>;
using lKK = metal::list<dimP, dimH, dimB, dimK>;
using lQQ = metal::list<dimP, dimH, dimB, dimJ>;
using lVV = metal::list<dimP, dimH, dimB, dimK>;
using lBETA = metal::list<dimH, dimB, dimJ, dimK>;
using lALPHA = metal::list<dimH, dimB, dimJ, dimK>;
using lGAMMA = metal::list<dimP, dimH, dimB, dimJ>;
using lWO = metal::list<dimP, dimH, dimN>;
using lATT = metal::list<dimB, dimJ, dimN>;
using lDROP1MASK = metal::list<dimB, dimJ, dimN>;
using lSB1 = metal::list<dimB, dimJ, dimN>;
using lS1 = metal::list<dimN>;
using lB1 = metal::list<dimN>;
using lLINB1 = metal::list<dimU>;
using lLINW1 = metal::list<dimU, dimN>;
using lSB1_LINW1 = metal::list<dimB, dimJ, dimU>;
using lDROP2 = metal::list<dimB, dimJ, dimU>;
using lLIN1 = metal::list<dimB, dimJ, dimU>;
using lLINB2 = metal::list<dimN>;
using lLINW2 = metal::list<dimN, dimU>;
using lDROP2_LINW2 = metal::list<dimB, dimJ, dimN>;
using lLIN2 = metal::list<dimB, dimJ, dimN>;
using lS2 = metal::list<dimN>;
using lDS2 = metal::list<dimN>;
using lB2 = metal::list<dimN>;
using lDB2 = metal::list<dimN>;
using lSB2 = metal::list<dimB, dimJ, dimN>;
using lDSB2 = metal::list<dimB, dimJ, dimN>;
using lLN2 = metal::list<dimB, dimJ, dimN>;
using lDLN2 = metal::list<dimB, dimJ, dimN>;
using lLN2STD = metal::list<dimB, dimJ>;
using lLN2DIFF = metal::list<dimB, dimJ, dimN>;
using lDROP3MASK = metal::list<dimB, dimJ, dimN>;
using lDRESID2 = metal::list<dimB, dimJ, dimN>;
using lDLIN2 = metal::list<dimB, dimJ, dimN>;
using lDDROP2 = metal::list<dimB, dimJ, dimU>;
using lDLINW2 = metal::list<dimN, dimU>;
using lDROP2MASK = metal::list<dimB, dimJ, dimU>;
using lACT = metal::list<dimB, dimJ, dimU>;
using lDLIN1 = metal::list<dimB, dimJ, dimU>;
using lDLIN1_LINW1 = metal::list<dimB, dimJ, dimN>;
using lDLINW1 = metal::list<dimU, dimN>;
using lLN1 = metal::list<dimB, dimJ, dimN>;
using lDLN1 = metal::list<dimB, dimJ, dimN>;
using lDSB1 = metal::list<dimB, dimJ, dimN>;
using lDS1 = metal::list<dimN>;
using lDB1 = metal::list<dimN>;
using lDLINB2 = metal::list<dimN>;
using lDLINB1 = metal::list<dimU>;
using lLN1STD = metal::list<dimB, dimJ>;
using lLN1DIFF = metal::list<dimB, dimJ, dimN>;
using lDRESID1 = metal::list<dimB, dimJ, dimN>;
using lDATT = metal::list<dimB, dimJ, dimN>;
using lDXATT = metal::list<dimB, dimJ, dimN>;
using lDX = metal::list<dimB, dimJ, dimN>;
using lDGAMMA = metal::list<dimP, dimH, dimB, dimJ>;
using lDVV = metal::list<dimP, dimH, dimB, dimK>;
using lDWV = metal::list<dimP, dimH, dimN>;
using lDALPHA = metal::list<dimH, dimB, dimJ, dimK>;
using lDBETA = metal::list<dimH, dimB, dimJ, dimK>;
using lDQQ = metal::list<dimP, dimH, dimB, dimJ>;
using lDWQ = metal::list<dimP, dimH, dimN>;
using lDKK = metal::list<dimP, dimH, dimB, dimK>;
using lDWK = metal::list<dimP, dimH, dimN>;
using lDKKQQVV = metal::list<dimQKV, dimP, dimH, dimB, dimJ>;
using lDWO = metal::list<dimP, dimH, dimN>;
using lBKQV = metal::list<dimQKV, dimH, dimP>;
using lWKK = metal::list<dimP, dimH, dimB, dimK>;
using lWQQ = metal::list<dimP, dimH, dimB, dimJ>;
using lWVV = metal::list<dimP, dimH, dimB, dimK>;
using lWKKself = metal::list<dimP, dimH, dimB, dimJ>;
using lWQQself = metal::list<dimP, dimH, dimB, dimJ>;
using lWVVself = metal::list<dimP, dimH, dimB, dimJ>;
using lBK = metal::list<dimH, dimP>;
using lBQ = metal::list<dimH, dimP>;
using lBV = metal::list<dimH, dimP>;
using lDBK = metal::list<dimH, dimP>;
using lDBQ = metal::list<dimH, dimP>;
using lDBV = metal::list<dimH, dimP>;
using lDKKself = metal::list<dimP, dimH, dimB, dimJ>;
using lDQQself = metal::list<dimP, dimH, dimB, dimJ>;
using lDVVself = metal::list<dimP, dimH, dimB, dimJ>;
using lKKself = metal::list<dimP, dimH, dimB, dimJ>;
using lQQself = metal::list<dimP, dimH, dimB, dimJ>;
using lVVself = metal::list<dimP, dimH, dimB, dimJ>;
using lBO = metal::list<dimN>;
using lDBO = metal::list<dimN>;
using lATTN_DROP = metal::list<dimH, dimB, dimJ, dimK>;
using lATTN_DROP_MASK = metal::list<dimH, dimB, dimJ, dimK>;
using lDATTN_DROP = metal::list<dimH, dimB, dimJ, dimK>;


using layoutsList = metal::list<ENCODER_LAYOUTS_LIST>;

using Real = half;

using Enc = Encoder<Real, dimB, dimK, dimJ, dimH, dimP, dimN, dimU, dimQKV, layoutsList>;


extern "C" Enc* init() {
    return new Enc();
}

extern "C" void encoder_forward(Enc* encoder, Real* X
    // weights
    , Real* WKQV
    , Real* BKQV
    , Real* WO
    , Real* BO
    , Real* S1
    , Real* B1
    , Real* LINB1
    , Real* LINW1
    , Real* S2
    , Real* B2
    , Real* LINB2
    , Real* LINW2
    // interm
    , Real* KKQQVV
    , Real* BETA
    , Real* ALPHA
    , Real* ATTN_DROP_MASK
    , Real* ATTN_DROP
    , Real* GAMMA
    , Real* ATT
    , Real* DROP1MASK
    , Real* SB1
    , Real* SB1_LINW1
    , Real* DROP2
    , Real* LIN1
    , Real* DROP2_LINW2
    , Real* LIN2
    , Real* LN2
    , Real* LN2STD
    , Real* LN2DIFF
    , Real* DROP2MASK
    , Real* DROP3MASK
    , Real* LN1
    , Real* ACT
    , Real* LN1STD
    , Real* LN1DIFF
    // out
    , Real* Y) 
{
    encoder->encoder_forward(X
        , WKQV
        , BKQV
        , WO
        , BO
        , S1
        , B1
        , LINB1
        , LINW1
        , S2
        , B2
        , LINB2
        , LINW2
        , KKQQVV
        , BETA
        , ALPHA
        , ATTN_DROP_MASK
        , ATTN_DROP
        , GAMMA
        , ATT
        , DROP1MASK
        , SB1
        , SB1_LINW1
        , DROP2
        , LIN1
        , DROP2_LINW2
        , LIN2
        , LN2
        , LN2STD
        , LN2DIFF
        , DROP2MASK
        , DROP3MASK
        , LN1
        , ACT
        , LN1STD
        , LN1DIFF
        , Y);
}

extern "C" void encoder_backward(Enc* encoder
    , Real* DY
    // param_gradients
    , Real* DWKQV
    , Real* DBKQV
    , Real* DWO
    , Real* DBO
    , Real* DS1
    , Real* DB1
    , Real* DLINB1
    , Real* DLINW1
    , Real* DS2
    , Real* DB2
    , Real* DLINB2
    , Real* DLINW2
    // backward intermediate
    , Real* DLN2
    , Real* DRESID2
    , Real* DLIN2
    , Real* DDROP2
    , Real* DLIN1
    , Real* DLIN1_LINW1
    , Real* DLN1
    , Real* DSB1
    , Real* DRESID1
    , Real* DATT
    , Real* DXATT
    , Real* DGAMMA
    , Real* DATTN_DROP
    , Real* DALPHA
    , Real* DBETA
    , Real* DKKQQVV
    // weights
    , Real* WKQV
    , Real* BKQV
    , Real* WO
    , Real* BO
    , Real* S1
    , Real* B1
    , Real* LINB1
    , Real* LINW1
    , Real* S2
    , Real* B2
    , Real* LINB2
    , Real* LINW2
    // forward intermediate
    , Real* KKQQVV
    , Real* BETA
    , Real* ALPHA
    , Real* ATTN_DROP_MASK
    , Real* ATTN_DROP
    , Real* GAMMA
    , Real* ATT
    , Real* DROP1MASK
    , Real* SB1
    , Real* SB1_LINW1
    , Real* DROP2
    , Real* LIN1
    , Real* DROP2_LINW2
    , Real* LIN2
    , Real* LN2
    , Real* LN2STD
    , Real* LN2DIFF
    , Real* DROP2MASK
    , Real* DROP3MASK
    , Real* LN1
    , Real* ACT
    , Real* LN1STD
    , Real* LN1DIFF
    //
    , Real* X
    //
    , Real* DX
) {
    encoder->encoder_backward(
        DY
        // param_gradients
        , DWKQV
        , DBKQV
        , DWO
        , DBO
        , DS1
        , DB1
        , DLINB1
        , DLINW1
        , DS2
        , DB2
        , DLINB2
        , DLINW2
        // backward intermediate
        , DLN2
        , DRESID2
        , DLIN2
        , DDROP2
        , DLIN1
        , DLIN1_LINW1
        , DLN1
        , DSB1
        , DRESID1
        , DATT
        , DXATT
        , DGAMMA
        , DATTN_DROP
        , DALPHA
        , DBETA
        , DKKQQVV
        // weights
        , WKQV
        , BKQV
        , WO
        , BO
        , S1
        , B1
        , LINB1
        , LINW1
        , S2
        , B2
        , LINB2
        , LINW2
        // forward intermediate
        , KKQQVV
        , BETA
        , ALPHA
        , ATTN_DROP_MASK
        , ATTN_DROP
        , GAMMA
        , ATT
        , DROP1MASK
        , SB1
        , SB1_LINW1
        , DROP2
        , LIN1
        , DROP2_LINW2
        , LIN2
        , LN2
        , LN2STD
        , LN2DIFF
        , DROP2MASK
        , DROP3MASK
        , LN1
        , ACT
        , LN1STD
        , LN1DIFF
        // 
        , X
        //
        , DX
    );
}

extern "C" void destroy(Enc* encoder) {
    delete encoder;
}

// extern "C" void softmax_test(double* in, double* out) {
//     struct d1 { enum { value = 1 }; };
//     struct d2 { enum { value = 1 }; };
//     struct dv { enum { value = 4 }; };
//     struct dr { enum { value = 64 }; };
//     using lIN = metal::list<d1, d2, dr, dv>;
//     using lOUT = metal::list<d1, d2, dr, dv>;

//     Softmax<double, d1, d2, dv, dr, lIN, lOUT>::run(in, out, 0);
//     cudaStreamSynchronize(0);
// }