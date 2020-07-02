#include <iostream>
#include <array>

#include <curand_kernel.h>

#include "all.cuh"
#include "blocks.cuh"
#include "half8.cuh"
#include "warpreduce.cuh"
#include "stridemapper.cuh"
#include "gemm.cuh"

#include "metal.hpp"

#include "block_aib.cuh"
#include "block_bad.cuh"
#include "block_baib.cuh"
#include "block_baob.cuh"
#include "block_bdrlb.cuh"
#include "block_bdrln.cuh"
#include "block_bei.cuh"
#include "block_blnrd.cuh"
#include "block_bs.cuh"
#include "block_bsb_ebsb.cuh"
#include "block_softmax.cuh"

#include "encoder_parameters.cuh"
    
struct Encoder {
    
    cublasHandle_t handle;

    cudaStream_t stream;

    cudaEvent_t event1;


    GlobalRandomState grs;

    float dropoutProbability = ENCODER_DROPOUT_PROBABILITY;
    
    Encoder()
    {        
        // cublas init
        CHECK(cublasCreate(&handle));
        CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

        // streams
        CHECK(cudaStreamCreate(&stream));

    }
    
    ~Encoder() {
    
        CHECK(cublasDestroy(handle));
    }

    void encoder_forward(ENCODER_FORWARD_DEF) 
    {
        Einsum<Real, lX, lWKQV, lWKKWQQWVV>::run(gX, gWKQV, gWKKWQQWVV, handle, stream, 1, CUBLAS_GEMM_ALGO3_TENSOR_OP);

        Real* gKK = gKKQQVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gQQ = gKKQQVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gVV = gKKQQVV + 2 * sizeP * sizeH * sizeB * sizeS;

        Real* gBK = gBKQV + 0 * sizeP * sizeH;
        Real* gBQ = gBKQV + 1 * sizeP * sizeH;
        Real* gBV = gBKQV + 2 * sizeP * sizeH;

        Real* gWKK = gWKKWQQWVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gWQQ = gWKKWQQWVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gWVV = gWKKWQQWVV + 2 * sizeP * sizeH * sizeB * sizeS;

        AttentionInputBiases<Real,
            dimH, dimP, dimB, dimJ,
            dimJ, dimJ,
            lWKKself, lWVVself, lWQQself,
            lBK, lBV, lBQ,
            lKKself, lVVself, lQQself
        >::run(gWKK, gWVV, gWQQ, gBK, gBV, gBQ, gKK, gVV, gQQ, stream);

        Einsum<Real, lKK, lQQ, lBETA>::run(gKK, gQQ, gBETA, handle, stream, 1./sqrtf(dimP::value), CUBLAS_GEMM_ALGO3_TENSOR_OP);
        
        Softmax<Real, dimK, dimK, lBETA, lALPHA, lATTN_DROP_MASK, lATTN_DROP>::run(gBETA, gALPHA, gATTN_DROP_MASK, gATTN_DROP,dropoutProbability, grs, stream);
        
        Einsum<Real, lVV, lATTN_DROP, lGAMMA>::run(gVV, gATTN_DROP, gGAMMA, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        Einsum<Real, lWO, lGAMMA, lATT>::run(gWO, gGAMMA, gATT, handle, stream, 1, CUBLAS_GEMM_ALGO6_TENSOR_OP);

        // end of attention
            
        BiasDropoutResidualLinearNorm<Real, true, dimI, dimI, lSB1, lATT, lX, lLN1, lLN1DIFF, lLN1STD, lATT, lDROP1MASK>::run(
            gSB1, gATT, gX, gLN1, gS1, gB1, gBO, gATT, gLN1DIFF, gLN1STD, gDROP1MASK, dropoutProbability, grs, stream);
        
        Einsum<Real, lSB1, lLINW1, lSB1_LINW1>::run(gSB1, gLINW1, gSB1_LINW1, handle, stream, 1, CUBLAS_GEMM_ALGO3_TENSOR_OP);
        
        BiasActivationDropout<Real, dimB, dimJ, dimU, dimU, dimJ, lSB1_LINW1, lDROP2, lLIN1, lDROP2MASK>::run(
            gSB1_LINW1, gDROP2, gLINB1, gLIN1, gDROP2MASK, dropoutProbability, grs, stream);
        
        Einsum<Real, lDROP2, lLINW2, lDROP2_LINW2>::run(gDROP2, gLINW2, gDROP2_LINW2, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        BiasDropoutResidualLinearNorm<Real, true, dimI, dimI, lSB2, lDROP2_LINW2, lSB1, lLN2, lLN2DIFF, lLN2STD, lLIN2, lDROP3MASK>::run(
            gSB2, gDROP2_LINW2, gSB1, gLN2, gS2, gB2, gLINB2, gLIN2, gLN2DIFF, gLN2STD, gDROP3MASK, dropoutProbability, grs, stream);

        //CHECK(cudaStreamSynchronize(stream));
    }
    void encoder_backward(ENCODER_BACKWARD_DEF) {

        BackwardScaleBias<Real, dimB, dimJ, dimI, dimI, dimJ, lLN2, lDLN2, lDSB2>::run(
            gLN2, gS2, gDSB2,
            gDLN2, gDS2, gDB2,
            stream);
            
        BackwardLayerNormResidualDropout<Real, dimI, dimI, 
            lDLN2, lLN2STD, lLN2DIFF, lDROP3MASK,
            lDRESID2, lDLIN2
        >::run(
            gDLN2, gLN2STD, gLN2DIFF, gDROP3MASK,
            gDRESID2, gDLIN2, stream
        );

        Einsum<Real, lDLIN2, lLINW2, lDDROP2>::run(gDLIN2, gLINW2, gDDROP2, handle, stream, 1, CUBLAS_GEMM_ALGO3_TENSOR_OP);
        
        Einsum<Real, lDLIN2, lDROP2, lDLINW2>::run(gDLIN2, gDROP2, gDLINW2, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            
        BackwardDropoutReluLinearBias<Real,
            dimB, dimJ, dimU,
            dimU, dimJ,
            lDDROP2, lDROP2MASK, lLIN1, lDLIN1
        >::run(gDDROP2, gDROP2MASK, gLIN1,
            gDLIN1, gDLINB1, stream);
        
        Einsum<Real, lDLIN1, lLINW1, lDLIN1_LINW1>::run(gDLIN1, gLINW1, gDLIN1_LINW1, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        Einsum<Real, lDLIN1, lSB1, lDLINW1>::run(gDLIN1, gSB1, gDLINW1, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        ExtendedBackwardScaleBias<Real,
            dimB, dimJ, dimI,
            dimI, dimJ,
            lLN1, lDLN1, lDRESID2, lDLIN1_LINW1, lDLIN2
        >::run(gLN1, gS1, gDRESID2, gDLIN1_LINW1, gDLIN2,
            gDLN1, gDS1, gDB1, gDLINB2, stream);
        
        BackwardLayerNormResidualDropout<Real,
            dimI, dimI, 
            lDLN1, lLN1STD, lLN1DIFF, lDROP1MASK,
            lDRESID1, lDATT
        >::run(
            gDLN1, gLN1STD, gLN1DIFF, gDROP1MASK,
            gDRESID1, gDATT, stream
        );

        // printf("gDRESID1: %p\n", gDRESID1);
        
        // attention backward start
        // attention_backward();

        BackwardAttnOutBias<Real,
            dimI, dimB, dimJ,
            dimI, dimJ,
            lDATT>::run(gDATT, gDBO, stream);

        Einsum<Real, lGAMMA, lDATT, lDWO>::run(gGAMMA, gDATT, gDWO, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        Einsum<Real, lWO, lDATT, lDGAMMA>::run(gWO, gDATT, gDGAMMA, handle, stream, 1, CUBLAS_GEMM_ALGO3_TENSOR_OP);

        Real* gDKK = gDKKQQVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gDQQ = gDKKQQVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gDVV = gDKKQQVV + 2 * sizeP * sizeH * sizeB * sizeS;

        Real* gKK = gKKQQVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gQQ = gKKQQVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gVV = gKKQQVV + 2 * sizeP * sizeH * sizeB * sizeS;

        Einsum<Real, lVV, lDGAMMA, lDATTN_DROP>::run(gVV, gDGAMMA, gDATTN_DROP, handle, stream, 1, CUBLAS_GEMM_ALGO3_TENSOR_OP);

        Einsum<Real, lDGAMMA, lATTN_DROP, lDVV>::run(gDGAMMA, gATTN_DROP, gDVV, handle, stream, 1, CUBLAS_GEMM_ALGO13_TENSOR_OP);

        BackwardSoftmax<Real, dimH, dimB, dimJ, dimK, dimK,
            lALPHA, lDBETA, lATTN_DROP_MASK, lDATTN_DROP>::run(gALPHA, gDBETA, gATTN_DROP_MASK, gDATTN_DROP, stream);

        Einsum<Real, lKK, lDBETA, lDQQ>::run(gKK, gDBETA, gDQQ, handle, stream, 1./sqrtf(dimP::value), CUBLAS_GEMM_ALGO0_TENSOR_OP);

        Einsum<Real, lQQ, lDBETA, lDKK>::run(gQQ, gDBETA, gDKK, handle, stream, 1./sqrtf(dimP::value), CUBLAS_GEMM_ALGO5_TENSOR_OP);

        Einsum<Real, lWKQV, lDKKQQVV, lDXATT>::run(gWKQV, gDKKQQVV, gDXATT, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        Einsum<Real, lX, lDKKQQVV, lDWKQV>::run(gX, gDKKQQVV, gDWKQV, handle, stream, 1, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        Real* gDBK = gDBKQV + 0 * sizeP * sizeH;
        Real* gDBQ = gDBKQV + 1 * sizeP * sizeH;
        Real* gDBV = gDBKQV + 2 * sizeP * sizeH;

        BackwardAttentionInputBiases<Real, 
            dimP, dimH, dimB, dimJ,
            dimB, dimJ,
            lDKKself, lDVVself, lDQQself, 
            lDBK, lDBV, lDBQ>::run(
                gDKK, gDVV, gDQQ, gDBK, gDBV, gDBQ, stream);

        // attention backward end
        
        BackwardEncoderInput<Real, dimB, dimJ, dimI, dimJ, dimI, lDXATT, lDRESID1, lDX>::run(gDXATT, gDRESID1, gDX, stream);

        //CHECK(cudaStreamSynchronize(stream));
    }

};


extern "C" Encoder* init() {
    return new Encoder();
}

extern "C" void encoder_forward(Encoder* encoder, ENCODER_FORWARD_DEF) {
    encoder->encoder_forward(ENCODER_FORWARD);
}

extern "C" void encoder_backward(Encoder* encoder, ENCODER_BACKWARD_DEF) {
    encoder->encoder_backward(ENCODER_BACKWARD);
}

extern "C" void destroy(Encoder* encoder) {
    delete encoder;
}



