#include <iostream>
#include <array>

#include <curand_kernel.h>
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"

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

void nvtx_region_begin(const char *s, int c = 0) {
#ifdef SUBSTATION_PROFILING
    nvtxEventAttributes_t ev;
    memset(&ev, 0, sizeof(nvtxEventAttributes_t));
    ev.version = NVTX_VERSION;
    ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    ev.colorType = NVTX_COLOR_ARGB;
    ev.color = c;
    ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
    ev.message.ascii = s;
    nvtxRangePushEx(&ev);
#else
    (void) s;
    (void) c;
#endif
}

void nvtx_region_end() {
#ifdef SUBSTATION_PROFILING
    nvtxRangePop();
#endif
}

struct Encoder {
    
    cublasHandle_t handle;

    cudaStream_t stream;

    cudaEvent_t event1;


    GlobalRandomState grs;

    float softmaxDropoutProbability = ENCODER_SOFTMAX_DROPOUT_PROBABILITY;
    float residual1DropoutProbability = ENCODER_RESIDUAL1_DROPOUT_PROBABILITY;
    float activationDropoutProbability = ENCODER_ACTIVATION_DROPOUT_PROBABILITY;
    float residual2DropoutProbability = ENCODER_RESIDUAL2_DROPOUT_PROBABILITY;
    
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
        nvtx_region_begin("QKV-fused");
        Einsum<Real, lX, lWKQV, lWKKWQQWVV>::run(gX, gWKQV, gWKKWQQWVV, handle, stream, 1, algoWKKWQQWVV);
        nvtx_region_end();

        Real* gKK = gKKQQVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gQQ = gKKQQVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gVV = gKKQQVV + 2 * sizeP * sizeH * sizeB * sizeS;

        Real* gBK = gBKQV + 0 * sizeP * sizeH;
        Real* gBQ = gBKQV + 1 * sizeP * sizeH;
        Real* gBV = gBKQV + 2 * sizeP * sizeH;

        Real* gWKK = gWKKWQQWVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gWQQ = gWKKWQQWVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gWVV = gWKKWQQWVV + 2 * sizeP * sizeH * sizeB * sizeS;

        nvtx_region_begin("aib");
        AttentionInputBiases<Real,
            dimH, dimP, dimB, dimJ,
            sdAIB_DV, sdAIB_DT,
            lWKKself, lWVVself, lWQQself,
            lBK, lBV, lBQ,
            lKKself, lVVself, lQQself
        >::run(gWKK, gWVV, gWQQ, gBK, gBV, gBQ, gKK, gVV, gQQ, stream);
        nvtx_region_end();

        nvtx_region_begin("QKT");
        Einsum<Real, lKK, lQQ, lBETA>::run(gKK, gQQ, gBETA, handle, stream, 1./sqrtf(dimP::value), algoBETA);
        nvtx_region_end();
        
        nvtx_region_begin("softmax");
        Softmax<Real, dimK, sdSM_DV, lBETA, lALPHA, lATTN_DROP_MASK, lATTN_DROP>::run(gBETA, gALPHA, gATTN_DROP_MASK, gATTN_DROP, softmaxDropoutProbability, grs, stream);
        nvtx_region_end();
        
        nvtx_region_begin("gamma");
        Einsum<Real, lVV, lATTN_DROP, lGAMMA>::run(gVV, gATTN_DROP, gGAMMA, handle, stream, 1, algoGAMMA);
        nvtx_region_end();
        
        nvtx_region_begin("out");
        Einsum<Real, lWO, lGAMMA, lATT>::run(gWO, gGAMMA, gATT, handle, stream, 1, algoATT);
        nvtx_region_end();

        // end of attention

        nvtx_region_begin("bdrln1");
        BiasDropoutResidualLinearNorm<Real, true, dimI, sdBDRLN1_DV, lSB1, lATT, lX, lLN1, lLN1DIFF, lLN1STD, lATT, lDROP1MASK>::run(
            gSB1, gATT, gX, gLN1, gS1, gB1, gBO, gATT, gLN1DIFF, gLN1STD, gDROP1MASK, residual1DropoutProbability, grs, stream);
        nvtx_region_end();
        
        nvtx_region_begin("lin1");
        Einsum<Real, lSB1, lLINW1, lSB1_LINW1>::run(gSB1, gLINW1, gSB1_LINW1, handle, stream, 1, algoSB1_LINW1);
        nvtx_region_end();
        
        nvtx_region_begin("bad");
        BiasActivationDropout<Real, dimB, dimJ, dimU, sdBAD_DV, sdBAD_DT, lSB1_LINW1, lDROP2, lLIN1, lDROP2MASK>::run(
            gSB1_LINW1, gDROP2, gLINB1, gLIN1, gDROP2MASK, activationDropoutProbability, grs, stream);
        nvtx_region_end();
        
        nvtx_region_begin("lin2");
        Einsum<Real, lDROP2, lLINW2, lDROP2_LINW2>::run(gDROP2, gLINW2, gDROP2_LINW2, handle, stream, 1, algoDROP2_LINW2);
        nvtx_region_end();
        
        nvtx_region_begin("bdrln2");
        BiasDropoutResidualLinearNorm<Real, true, dimI, sdBDRLN2_DV, lSB2, lDROP2_LINW2, lSB1, lLN2, lLN2DIFF, lLN2STD, lLIN2, lDROP3MASK>::run(
            gSB2, gDROP2_LINW2, gSB1, gLN2, gS2, gB2, gLINB2, gLIN2, gLN2DIFF, gLN2STD, gDROP3MASK, residual2DropoutProbability, grs, stream);
        nvtx_region_end();

        //CHECK(cudaStreamSynchronize(stream));
    }
    void encoder_backward(ENCODER_BACKWARD_DEF) {

        nvtx_region_begin("bsb");
        BackwardScaleBias<Real, dimB, dimJ, dimI, sdBSB_DV, sdBSB_DW, lLN2, lDLN2, lDSB2>::run(
            gLN2, gS2, gDSB2,
            gDLN2, gDS2, gDB2,
            stream);
        nvtx_region_end();

        nvtx_region_begin("blnrd1");
        BackwardLayerNormResidualDropout<Real, dimI, sdBLNRD1_DV, 
            lDLN2, lLN2STD, lLN2DIFF, lDROP3MASK,
            lDRESID2, lDLIN2
        >::run(
            gDLN2, gLN2STD, gLN2DIFF, gDROP3MASK,
            gDRESID2, gDLIN2, stream
        );
        nvtx_region_end();

        nvtx_region_begin("dXlin2");
        Einsum<Real, lDLIN2, lLINW2, lDDROP2>::run(gDLIN2, gLINW2, gDDROP2, handle, stream, 1, algoDDROP2);
        nvtx_region_end();
        
        nvtx_region_begin("dWlin2");
        Einsum<Real, lDLIN2, lDROP2, lDLINW2>::run(gDLIN2, gDROP2, gDLINW2, handle, stream, 1, algoDLINW2);
        nvtx_region_end();

        nvtx_region_begin("bdrlb");
        BackwardDropoutReluLinearBias<Real,
            dimB, dimJ, dimU,
            sdBDRLB_DV, sdBDRLB_DW,
            lDDROP2, lDROP2MASK, lLIN1, lDLIN1
        >::run(gDDROP2, gDROP2MASK, gLIN1,
            gDLIN1, gDLINB1, stream);
        nvtx_region_end();
        
        nvtx_region_begin("dXlin1");
        Einsum<Real, lDLIN1, lLINW1, lDLIN1_LINW1>::run(gDLIN1, gLINW1, gDLIN1_LINW1, handle, stream, 1, algoDLIN1_LINW1);
        nvtx_region_end();
        
        nvtx_region_begin("dWlin1");
        Einsum<Real, lDLIN1, lSB1, lDLINW1>::run(gDLIN1, gSB1, gDLINW1, handle, stream, 1, algoDLINW1);
        nvtx_region_end();
        
        nvtx_region_begin("ebsb");
        ExtendedBackwardScaleBias<Real,
            dimB, dimJ, dimI,
            sdEBSB_DV, sdEBSB_DW,
            lLN1, lDLN1, lDRESID2, lDLIN1_LINW1, lDLIN2
        >::run(gLN1, gS1, gDRESID2, gDLIN1_LINW1, gDLIN2,
            gDLN1, gDS1, gDB1, gDLINB2, stream);
        nvtx_region_end();
        
        nvtx_region_begin("blnrd2");
        BackwardLayerNormResidualDropout<Real,
            dimI, sdBLNRD2_DV, 
            lDLN1, lLN1STD, lLN1DIFF, lDROP1MASK,
            lDRESID1, lDATT
        >::run(
            gDLN1, gLN1STD, gLN1DIFF, gDROP1MASK,
            gDRESID1, gDATT, stream
        );
        nvtx_region_end();

        // printf("gDRESID1: %p\n", gDRESID1);
        
        // attention backward start
        // attention_backward();

        nvtx_region_begin("baob");
        BackwardAttnOutBias<Real,
            dimI, dimB, dimJ,
            sdBAOB_DV, sdBAOB_DW,
            lDATT>::run(gDATT, gDBO, stream);
        nvtx_region_end();

        nvtx_region_begin("dWout");
        Einsum<Real, lGAMMA, lDATT, lDWO>::run(gGAMMA, gDATT, gDWO, handle, stream, 1, algoDWO);
        nvtx_region_end();
        
        nvtx_region_begin("dXout");
        Einsum<Real, lWO, lDATT, lDGAMMA>::run(gWO, gDATT, gDGAMMA, handle, stream, 1, algoDGAMMA);
        nvtx_region_end();

        Real* gDKK = gDKKQQVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gDQQ = gDKKQQVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gDVV = gDKKQQVV + 2 * sizeP * sizeH * sizeB * sizeS;

        Real* gKK = gKKQQVV + 0 * sizeP * sizeH * sizeB * sizeS;
        Real* gQQ = gKKQQVV + 1 * sizeP * sizeH * sizeB * sizeS;
        Real* gVV = gKKQQVV + 2 * sizeP * sizeH * sizeB * sizeS;

        nvtx_region_begin("dX1gamma");
        Einsum<Real, lVV, lDGAMMA, lDATTN_DROP>::run(gVV, gDGAMMA, gDATTN_DROP, handle, stream, 1, algoDATTN_DROP);
        nvtx_region_end();

        nvtx_region_begin("dX2gamma");
        Einsum<Real, lDGAMMA, lATTN_DROP, lDVV>::run(gDGAMMA, gATTN_DROP, gDVV, handle, stream, 1, algoDVV);
        nvtx_region_end();

        nvtx_region_begin("bs");
        BackwardSoftmax<Real, dimH, dimB, dimJ, dimK, sdBS_DV,
            lALPHA, lDBETA, lATTN_DROP_MASK, lDATTN_DROP>::run(gALPHA, gDBETA, gATTN_DROP_MASK, gDATTN_DROP, stream);
        nvtx_region_end();

        nvtx_region_begin("dX1QKT");
        Einsum<Real, lKK, lDBETA, lDQQ>::run(gKK, gDBETA, gDQQ, handle, stream, 1./sqrtf(dimP::value), algoDQQ);
        nvtx_region_end();

        nvtx_region_begin("dX2QKT");
        Einsum<Real, lQQ, lDBETA, lDKK>::run(gQQ, gDBETA, gDKK, handle, stream, 1./sqrtf(dimP::value), algoDKK);
        nvtx_region_end();

        nvtx_region_begin("dXQKV-fused");
        Einsum<Real, lWKQV, lDKKQQVV, lDXATT>::run(gWKQV, gDKKQQVV, gDXATT, handle, stream, 1, algoDXATT);
        nvtx_region_end();

        nvtx_region_begin("dWQKV-fused");
        Einsum<Real, lX, lDKKQQVV, lDWKQV>::run(gX, gDKKQQVV, gDWKQV, handle, stream, 1, algoDWKQV);
        nvtx_region_end();

        Real* gDBK = gDBKQV + 0 * sizeP * sizeH;
        Real* gDBQ = gDBKQV + 1 * sizeP * sizeH;
        Real* gDBV = gDBKQV + 2 * sizeP * sizeH;

        nvtx_region_begin("baib");
        BackwardAttentionInputBiases<Real, 
            dimP, dimH, dimB, dimJ,
            sdBAIB_DV, sdBAIB_DW,
            lDKKself, lDVVself, lDQQself, 
            lDBK, lDBV, lDBQ>::run(
                gDKK, gDVV, gDQQ, gDBK, gDBV, gDBQ, stream);
        nvtx_region_end();

        // attention backward end
        
        nvtx_region_begin("bei");
        BackwardEncoderInput<Real, dimB, dimJ, dimI, sdBEI_DV, sdBEI_DT, lDXATT, lDRESID1, lDX>::run(gDXATT, gDRESID1, gDX, stream);
        nvtx_region_end();

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



