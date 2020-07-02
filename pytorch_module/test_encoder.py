import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer
from copy import deepcopy

from substation.transformer import encoder as ref_encoder, encoder_backward as ref_encoder_backward

sizes = dict(B=4, S=128, H=8, P=32)
sizes['N'] = sizes['H'] * sizes['P']
sizes['U'] = 4 * sizes['N']

dims = dict(B=sizes['B'],
            K=sizes['S'],
            J=sizes['S'],
            H=sizes['H'],
            P=sizes['P'],
            N=sizes['N'],
            U=sizes['U'],
            T=3)

layouts = dict(
    X='NBJ',
    K='NBK',
    Q='NBJ',
    V='NBK',
    WKQV='TPHN',
    WK='PHN',
    WQ='PHN',
    WV='PHN',
    KKQQVV='TPHBJ',
    KK='PHBK',
    QQ='PHBJ',
    VV='PHBK',
    BETA='HBJK',
    ALPHA='HBJK',
    GAMMA='PHBJ',
    WO='PHN',
    ATT='NBJ',
    DROP1MASK='BJN',
    SB1='BJN',
    S1='N',
    B1='N',
    LINB1='U',
    LINW1='UN',
    SB1_LINW1='BJU',
    DROP2='BJU',
    LIN1='BJU',
    LINB2='N',
    LINW2='NU',
    DROP2_LINW2='BJN',
    LIN2='BJN',
    S2='N',
    DS2='N',
    B2='N',
    DB2='N',
    SB2='BJN',
    DSB2='BJN',
    LN2='BJN',
    DLN2='BJN',
    LN2STD='BJ',
    LN2DIFF='BJN',
    DROP3MASK='BJN',
    DRESID2='BJN',
    DLIN2='BJN',
    DDROP2='BJU',
    DLINW2='NU',
    DROP2MASK='BJU',
    ACT='BJU',
    DLIN1='BJU',
    DLIN1_LINW1='BJN',
    DLINW1='UN',
    LN1='BJN',
    DLN1='BJN',
    DSB1='BJN',
    DS1='N',
    DB1='N',
    DLINB2='N',
    DLINB1='U',
    LN1STD='BJ',
    LN1DIFF='BJN',
    DRESID1='BJN',
    DATT='BJN',
    DXATT='BJN',
    DX='NBJ',
    DGAMMA='PHBJ',
    DVV='PHBK',
    DWV='PHN',
    DALPHA='HBJK',
    DBETA='HBJK',
    DQQ='PHBJ',
    DWQ='PHN',
    DKK='PHBK',
    DWK='PHN',
    DKKQQVV='TPHBJ',
    DWO='PHN'
    )

special_arrays = ['K', 'Q', 'V', 'WK', 'WQ', 'WV', 'KK', 'QQ', 'VV', 'DKK', 'DQQ', 'DVV']

def gpu_mem_helper():
    if not os.path.exists('gpu_mem_helper.so'):
        gpu_mem_source = """
            #include <cuda.h>
            #include <stdio.h>
        
            #define CHECK(expr) do {\
                auto err = (expr);\
                if (err != 0) {\
                    printf("ERROR %s %s:%d\\n", #expr, __FILE__, __LINE__); \
                    abort(); \
                }\
            } while(0)

            extern "C" {
                void* gpu_allocate(size_t size) {
                    void* ptr = nullptr;
                    CHECK(cudaMalloc(&ptr, size));
                    CHECK(cudaMemset(ptr, 0, size));
                    return ptr;
                }

                void gpu_free(void* ptr) {
                    CHECK(cudaFree(ptr));
                }

                void host_to_gpu(void* gpu, void* host, size_t size) {
                    CHECK(cudaMemcpy(gpu, host, size, cudaMemcpyHostToDevice));
                }

                void gpu_to_host(void* host, void* gpu, size_t size) {
                    CHECK(cudaMemcpy(host, gpu, size, cudaMemcpyDeviceToHost));
                }
            }
        """
        
        with open('gpu_mem_helper.cu', 'w') as f:
            f.write(gpu_mem_source)
            
        subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC gpu_mem_helper.cu -o gpu_mem_helper.o".split(' '))
        subprocess.run("nvcc -shared -o gpu_mem_helper.so gpu_mem_helper.o".split(' '))
    
    lib = ctypes.CDLL('./gpu_mem_helper.so')
    
    lib.gpu_allocate.argtypes = [ctypes.c_size_t]
    lib.gpu_allocate.restype = ctypes.c_void_p
    
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    
    lib.host_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    
    lib.gpu_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    
    return lib

def generate_encoder():
    if not os.path.exists('sample_encoder.so'):
        print("Start encoder compilation...")
        subprocess.run("nvcc -g -G -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC sample_encoder.cu -o sample_encoder.o".split(' '))
        subprocess.run("nvcc -lcublas -shared -o sample_encoder.so sample_encoder.o".split(' '))
        print("Done!")
    
    lib = ctypes.CDLL('./sample_encoder.so')
    
    lib.init.argtypes = [ctypes.c_void_p for _ in layouts]
    lib.init.restype = ctypes.c_void_p
    
    lib.encoder_forward.argtypes = [ctypes.c_void_p]
    lib.encoder_backward.argtypes = [ctypes.c_void_p]
    
    lib.destroy.argtypes = [ctypes.c_void_p]
    
    return lib

elem_size = 8 # bytes in double
data_type = np.float64

def array_size(name):
    return reduce(lambda x,y: x*y, [dims[d] for d in layouts[name]])

def allocate_gpu_memory(memhelper):
    
    # allocate GPU memory 
    arrays = {}
    for arr in layouts:
        if arr not in special_arrays:
            size = array_size(arr)
        arrays[arr] = memhelper.gpu_allocate(size * elem_size)
        
    
    arrays_and_views = {}
    for arr in layouts:
        if arr in special_arrays:
            if arr in 'KQV':
                arrays_and_views[arr] = arrays['X']
            elif arr == 'KK':
                arrays_and_views[arr] = arrays['KKQQVV'] + 0 * elem_size * array_size('KK')
            elif arr == 'QQ':
                arrays_and_views[arr] = arrays['KKQQVV'] + 1 * elem_size * array_size('KK')
            elif arr == 'VV':
                arrays_and_views[arr] = arrays['KKQQVV'] + 2 * elem_size * array_size('KK')
            elif arr == 'DKK':
                arrays_and_views[arr] = arrays['DKKQQVV'] + 0 * elem_size * array_size('DKK')
            elif arr == 'DQQ':
                arrays_and_views[arr] = arrays['DKKQQVV'] + 1 * elem_size * array_size('DKK')
            elif arr == 'DVV':
                arrays_and_views[arr] = arrays['DKKQQVV'] + 2 * elem_size * array_size('DKK')
            elif arr == 'WK':
                arrays_and_views[arr] = arrays['WKQV'] + 0 * elem_size * array_size('WK')
            elif arr == 'WQ':
                arrays_and_views[arr] = arrays['WKQV'] + 1 * elem_size * array_size('WK')
            elif arr == 'WV':
                arrays_and_views[arr] = arrays['WKQV'] + 2 * elem_size * array_size('WK')
            else:
                assert(0)
        else:
            arrays_and_views[arr] = arrays[arr]
    
    return arrays, arrays_and_views
    
def from_gpu(arrays_and_views, name):
    memhelper = gpu_mem_helper()
    
    res = np.zeros([dims[dim] for dim in layouts[name]], dtype=data_type)
    memhelper.gpu_to_host(res.ctypes.data, arrays_and_views[name], array_size(name) * elem_size)
    return res
    
def isclose(a, b, atol=1e-4, rtol=1e-4):
    if a.shape != b.shape:
        print("Shape missmatch:", a.shape, b.shape)
        return False
    pattern = np.absolute(a - b) < (atol + rtol * np.absolute(b))
    if pattern.all():
        return True
    else:
        a_bad = np.extract(1 - pattern, a)
        b_bad = np.extract(1 - pattern, b)
        bad_ratio = a_bad.size * 1. / pattern.size
        print("Bad ratio", bad_ratio)
        # print(a)
        # print(b)
        print(a_bad)
        print(b_bad)
        return False
        # if bad_ratio < 0.05:
        #     return True
        # else:
        #     print("Bad ratio", bad_ratio)
        #     print(a_bad)
        #     print(b_bad)
        #     return False
    
# def softmax_minitest(encoder):
#     memhelper = gpu_mem_helper()

#     A = np.random.rand(64, 4).astype(data_type)
#     B = np.zeros((64, 4), dtype=data_type)
    
#     gA = memhelper.gpu_allocate(A.size * elem_size)
#     gB = memhelper.gpu_allocate(B.size * elem_size)
    
#     memhelper.host_to_gpu(gA, A.ctypes.data, A.size * elem_size)
    
#     encoder.softmax_test.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
#     encoder.softmax_test(gA, gB)
    
#     memhelper.gpu_to_host(B.ctypes.data, gB, B.size * elem_size)
    
#     from scipy.special import softmax as sm
#     ref_B = sm(A, axis=0)
#     assert(isclose(B, ref_B))
    
def test_encoder():
    
    memhelper = gpu_mem_helper()
    
    encoder = generate_encoder()
    
    arrays, arrays_and_views = allocate_gpu_memory(memhelper)
    
    encoder_instance = encoder.init(*arrays_and_views.values())
    
    
    #A = np.array([[1,2,3],[4,5,6]], dtype=data_type)
    #B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=data_type)
    #C = np.zeros((2, 4), dtype=data_type)
    
    #gA = memhelper.gpu_allocate(A.size * elem_size)
    #gB = memhelper.gpu_allocate(B.size * elem_size)
    #gC = memhelper.gpu_allocate(C.size * elem_size)
    
    #memhelper.host_to_gpu(gA, A.ctypes.data, A.size * elem_size)
    #memhelper.host_to_gpu(gB, B.ctypes.data, B.size * elem_size)
    
    #encoder.einsum_test.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    #encoder.einsum_test(gA, gB, gC)
    
    #memhelper.gpu_to_host(C.ctypes.data, gC, C.size * elem_size)
    
    #print('gA: %x, gB: %x, gC: %x' % (gA, gB, gC))
    
    #ref_C = np.einsum('mk,kn->mn', A, B)
    #print(C)
    #print('----------------')
    #print(ref_C)
    #if not np.allclose(C, ref_C):
        #print("FAIL")
    #else:
        #print("OK")
    
    #return

    # softmax_minitest(encoder)
    # return
    
    # compute reference solution
    
    embed_size = sizes['N']
    num_heads = sizes['H']
    batch_size = sizes['B']
    max_seq_len = sizes['S']
    proj_size = embed_size // num_heads
    
    x = np.random.randn(batch_size, max_seq_len, embed_size)
    wq = np.random.randn(num_heads, proj_size, embed_size)
    wk = np.random.randn(num_heads, proj_size, embed_size)
    wv = np.random.randn(num_heads, proj_size, embed_size)
    in_b = np.zeros((3, num_heads, proj_size))
    wo = np.random.randn(embed_size, embed_size)
    out_b = np.zeros(embed_size)
    scale = 1.0 / np.sqrt(proj_size)
    norm1_scale = np.random.randn(embed_size)
    norm1_bias = np.random.randn(embed_size)
    norm2_scale = np.random.randn(embed_size)
    norm2_bias = np.random.randn(embed_size)
    linear1_w = np.random.randn(4*embed_size, embed_size)
    linear1_b = np.random.randn(4*embed_size)
    linear2_w = np.random.randn(embed_size, 4*embed_size)
    linear2_b = np.random.randn(embed_size)
    
    (y,
        i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
        i_attn_scaled_scores, i_attn_dropout_mask,
        i_norm1_mean, i_norm1_std, i_norm1_normed,
        i_linear1_dropout_mask, i_ff_dropout_mask,
        i_norm2_mean, i_norm2_std, i_norm2_normed,
        i_ff_resid, i_ff1, iff1_linear, i_normed1,
        i_attn_resid) = ref_encoder(
            x, wq, wk, wv, wo, in_b, out_b, scale,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            0, 0, 0, activation='relu')
        
    dy = np.random.randn(*y.shape)
    
    (dx,
        dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b,
        dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
        dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b) = ref_encoder_backward(
            x, dy,
            i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
            i_attn_scaled_scores, i_attn_dropout_mask,
            i_norm1_mean, i_norm1_std, i_norm1_normed,
            i_linear1_dropout_mask, i_ff_dropout_mask,
            i_norm2_mean, i_norm2_std, i_norm2_normed,
            i_ff_resid, i_ff1, iff1_linear, i_normed1,
            i_attn_resid,
            wq, wk, wv, wo, scale,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            0, 0, 0, activation='relu')
    
    # transform layouts for input params
    
    #x = np.random.randn(batch_size, max_seq_len, embed_size)
    myX = np.ascontiguousarray(np.einsum('BJN->%s' % layouts['X'], x), dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['X'], myX.ctypes.data, array_size('X') * elem_size)
    #wq = np.random.randn(num_heads, proj_size, embed_size)
    myWQ = np.ascontiguousarray(np.einsum('HPN->%s' % layouts['WQ'], wq), dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['WQ'], myWQ.ctypes.data, array_size('WQ') * elem_size)
    #wk = np.random.randn(num_heads, proj_size, embed_size)
    myWK = np.ascontiguousarray(np.einsum('HPN->%s' % layouts['WK'], wk), dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['WK'], myWK.ctypes.data, array_size('WK') * elem_size)
    #wv = np.random.randn(num_heads, proj_size, embed_size)
    myWV = np.ascontiguousarray(np.einsum('HPN->%s' % layouts['WV'], wv), dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['WV'], myWV.ctypes.data, array_size('WV') * elem_size)
    #in_b = np.random.randn(3, num_heads, proj_size)
    #wo = np.random.randn(embed_size, embed_size)
    myWO = np.ascontiguousarray(np.einsum('NHP->%s' % layouts['WO'], wo.reshape(sizes['N'], sizes['H'], sizes['P'])), dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['WO'], myWO.ctypes.data, array_size('WO') * elem_size)
    #out_b = np.random.randn(embed_size)
    #scale = 1.0 / np.sqrt(proj_size)
    #norm1_scale = np.random.randn(embed_size)
    myS1 = np.ascontiguousarray(norm1_scale, dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['S1'], myS1.ctypes.data, array_size('S1') * elem_size)
    #norm1_bias = np.random.randn(embed_size)
    myB1 = np.ascontiguousarray(norm1_bias, dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['B1'], myB1.ctypes.data, array_size('B1') * elem_size)
    #norm2_scale = np.random.randn(embed_size)
    myS2 = np.ascontiguousarray(norm2_scale, dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['S2'], myS2.ctypes.data, array_size('S2') * elem_size)
    #norm2_bias = np.random.randn(embed_size)
    myB2 = np.ascontiguousarray(norm2_bias, dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['B2'], myB2.ctypes.data, array_size('B2') * elem_size)
    #linear1_w = np.random.randn(4*embed_size, embed_size)
    myLINW1 = np.ascontiguousarray(np.einsum('UN->%s' % layouts['LINW1'], linear1_w), dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['LINW1'], myLINW1.ctypes.data, array_size('LINW1') * elem_size)
    #linear1_b = np.random.randn(4*embed_size)
    myLINB1 = np.ascontiguousarray(linear1_b, dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['LINB1'], myLINB1.ctypes.data, array_size('LINB1') * elem_size)
    #linear2_w = np.random.randn(embed_size, 4*embed_size)
    myLINW2 = np.ascontiguousarray(np.einsum('NU->%s' % layouts['LINW2'], linear2_w), dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['LINW2'], myLINW2.ctypes.data, array_size('LINW2') * elem_size)
    #linear2_b = np.random.randn(embed_size)
    myLINB2 = np.ascontiguousarray(linear2_b, dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['LINB2'], myLINB2.ctypes.data, array_size('LINB2') * elem_size)

    # transform layouts for output params
    
    myDSB2 = np.ascontiguousarray(dy, dtype=data_type)
    memhelper.host_to_gpu(arrays_and_views['DSB2'], myDSB2.ctypes.data, array_size('DSB2') * elem_size)

    
    # compute custom solution
    
    encoder.encoder_forward(encoder_instance)
    
    encoder.encoder_backward(encoder_instance)


    # print(np.stack(i_attn_concat).shape) #B, J, N (HP) = GAMMA
    # print(np.stack(i_attn_proj_q).shape) #B, H, J, P = QQ
    # print(np.stack(i_attn_proj_k).shape) #B, H, K, P = KK
    # print(np.stack(i_attn_proj_v).shape) #B, H, K, P = VV
    # print(np.stack(i_attn_scaled_scores).shape) # B, H, J, K = ALPHA
    
    # compare results
    
    ### START OF FORWARD PASS

    refQQ = np.einsum("BHJP->%s" % layouts['QQ'], np.stack(i_attn_proj_q))
    myQQ = from_gpu(arrays_and_views, 'QQ')
    assert(isclose(myQQ, refQQ))
    
    refKK = np.einsum("BHKP->%s" % layouts['KK'], np.stack(i_attn_proj_k))
    myKK = from_gpu(arrays_and_views, 'KK')
    assert(isclose(myKK, refKK))
    
    refVV = np.einsum("BHKP->%s" % layouts['VV'], np.stack(i_attn_proj_v))
    myVV = from_gpu(arrays_and_views, 'VV')
    assert(isclose(myVV, refVV))

    myBETA = from_gpu(arrays_and_views, 'BETA')
    refBETA = np.einsum("%s,%s->%s" % (layouts['KK'], layouts['QQ'], layouts['BETA']), myKK, myQQ) / np.sqrt(dims['P'])
    assert(isclose(myBETA, refBETA))
    
    refALPHA = np.einsum("BHJK->%s" % layouts['ALPHA'], np.stack(i_attn_scaled_scores))
    myALPHA = from_gpu(arrays_and_views, 'ALPHA')
    assert(isclose(myALPHA, refALPHA))
    
    refGAMMA = np.einsum("BJHP->%s" % layouts['GAMMA'], np.stack(i_attn_concat).reshape(dims['B'], dims['J'], dims['H'], dims['P']))
    myGAMMA = from_gpu(arrays_and_views, 'GAMMA')
    assert(isclose(myGAMMA, refGAMMA))
    
    refRESID1 = np.stack(i_attn_resid)
    myATT = from_gpu(arrays_and_views, 'ATT')
    myRESID1 = np.einsum("%s->BJN" % layouts['ATT'], myATT) + np.einsum("%s->BJN" % layouts['X'], myX)
    assert(isclose(myRESID1, refRESID1))
    
    #i_normed1 = SB1
    #i_norm1_mean = LN1MEAN = RESID1 - LN1DIFF
    #i_norm1_std = LN1STD
    #i_norm1_normed = LN1
    
    refLN1 = np.einsum("BJN->%s" % layouts['LN1'], i_norm1_normed)
    myLN1 = from_gpu(arrays_and_views, 'LN1')
    assert(isclose(myLN1, refLN1))
    
    refSB1 = np.einsum("BJN->%s" % layouts['SB1'], i_normed1)
    mySB1 = from_gpu(arrays_and_views, 'SB1')
    assert(isclose(mySB1, refSB1))
    
    refLN1DIFF = refRESID1 - i_norm1_mean
    myLN1DIFF = from_gpu(arrays_and_views, 'LN1DIFF')
    assert(isclose(myLN1DIFF, refLN1DIFF))

    refLN1STD = np.einsum("BJ->%s" % layouts['LN1STD'], i_norm1_std[:,:,0])
    myLN1STD = 1 / from_gpu(arrays_and_views, 'LN1STD')
    assert(isclose(myLN1STD, refLN1STD))

    # iff1_linear = LIN1
    # i_ff1 = DROP2

    refLIN1 = np.einsum("BJU->%s" % layouts['LIN1'], iff1_linear)
    myLIN1 = from_gpu(arrays_and_views, 'LIN1')
    assert(isclose(myLIN1, refLIN1))

    refDROP2 = np.einsum("BJU->%s" % layouts['DROP2'], i_ff1)
    myDROP2 = from_gpu(arrays_and_views, 'DROP2')
    assert(isclose(myDROP2, refDROP2))

    #i_ff_resid = RESID2
    #i_norm2_mean = LN2MEAN = RESID2 - LN2DIFF
    #i_norm2_std = LN1STD
    #i_norm2_normed = LN1
    #y = SB2

    refRESID2 = np.stack(i_ff_resid)
    myLIN2 = from_gpu(arrays_and_views, 'LIN2')
    myRESID2 = np.einsum("%s->BJN" % layouts['LIN2'], myLIN2) + np.einsum("%s->BJN" % layouts['SB1'], mySB1)
    assert(isclose(myRESID2, refRESID2))

    refLN2 = np.einsum("BJN->%s" % layouts['LN2'], i_norm2_normed)
    myLN2 = from_gpu(arrays_and_views, 'LN2')
    assert(isclose(myLN2, refLN2))
    
    refSB2 = np.einsum("BJN->%s" % layouts['SB2'], y)
    mySB2 = from_gpu(arrays_and_views, 'SB2')
    assert(isclose(mySB2, refSB2))
    
    refLN2DIFF = refRESID2 - i_norm2_mean
    myLN2DIFF = from_gpu(arrays_and_views, 'LN2DIFF')
    assert(isclose(myLN2DIFF, refLN2DIFF))

    refLN2STD = np.einsum("BJ->%s" % layouts['LN2STD'], i_norm2_std[:,:,0])
    myLN2STD = 1 / from_gpu(arrays_and_views, 'LN2STD')
    assert(isclose(myLN2STD, refLN2STD))

    ### END OF FORWARD PASS

    ### START BACKWARD PASS

    #dnorm2_scale = DS2
    #dnorm2_bias = DB2

    refDS2 = dnorm2_scale
    myDS2 = from_gpu(arrays_and_views, 'DS2')
    assert(isclose(myDS2, refDS2))

    refDB2 = dnorm2_bias
    myDB2 = from_gpu(arrays_and_views, 'DB2')
    assert(isclose(myDB2, refDB2))

    #dlinear2_w = DLINW2
    #dlinear2_b = DLINB2

    refDLINW2 = dlinear2_w
    myDLINW2 = from_gpu(arrays_and_views, 'DLINW2')
    assert(isclose(myDLINW2, refDLINW2))

    refDLINB2 = dlinear2_b
    myDLINB2 = from_gpu(arrays_and_views, 'DLINB2')
    assert(isclose(myDLINB2, refDLINB2))

    #dnorm1_scale = DS2
    #dnorm1_bias = DB2

    refDS1 = dnorm1_scale
    myDS1 = from_gpu(arrays_and_views, 'DS1')
    assert(isclose(myDS1, refDS1))

    refDB1 = dnorm1_bias
    myDB1 = from_gpu(arrays_and_views, 'DB1')
    assert(isclose(myDB1, refDB1))

    #dlinear1_w = DLINW1
    #dlinear1_b = DLINB1

    refDLINW1 = dlinear1_w
    myDLINW1 = from_gpu(arrays_and_views, 'DLINW1')
    assert(isclose(myDLINW1, refDLINW1))

    refDLINB1 = dlinear1_b
    myDLINB1 = from_gpu(arrays_and_views, 'DLINB1')
    assert(isclose(myDLINB1, refDLINB1))

    # 
    def ref_blnrd(dout, std, diff, drop_mask):
        from substation.transformer import layer_norm_backward_data, dropout_backward_data

        # only diff = x - mean is used in layer_norm_backward_data, x and mean themselves are not used
        x = diff
        mean = 0
        d_ln_in = layer_norm_backward_data(x, dout, mean, 1 / np.repeat(std[:, :, np.newaxis], x.shape[-1], axis=2))
        nonzero_magic_value = 0.12345
        d_drop_in = dropout_backward_data(d_ln_in, nonzero_magic_value, drop_mask)
        return d_ln_in, d_drop_in
    myDLN1 = from_gpu(arrays_and_views, 'DLN1')
    myLN1STD = from_gpu(arrays_and_views, 'LN1STD')
    myLN1DIFF = from_gpu(arrays_and_views, 'LN1DIFF')
    myDROP1MASK = from_gpu(arrays_and_views, 'DROP1MASK')
    
    refDRESID1, refDATT = ref_blnrd(myDLN1, myLN1STD, myLN1DIFF, myDROP1MASK)
    
    myDRESID1 = from_gpu(arrays_and_views, 'DRESID1')
    assert(isclose(myDRESID1, refDRESID1))

    myDATT = from_gpu(arrays_and_views, 'DATT')
    assert(isclose(myDATT, refDATT))



    # dattn_wq [HPN] = DWQ
    # dattn_wk [HPN] = DWK
    # dattn_wv [HPN] = DWV
    # dattn_wo [NHP] = DWO
    # dx [BJN] = DX

    refDWO = np.einsum("NHP->%s" % layouts['DWO'], dattn_wo.reshape(dims['N'], dims['H'], dims['P']))
    myDWO = from_gpu(arrays_and_views, 'DWO')
    assert(isclose(myDWO, refDWO))


    # myV = from_gpu(arrays_and_views, 'V')
    # myDVV = from_gpu(arrays_and_views, 'DVV')
    # refDWV1 = np.einsum("%s,%s->%s" % (layouts['V'], layouts['DVV'], layouts['DWV']), myV, myDVV)

    refDWV = np.einsum("HPN->%s" % layouts['DWV'], dattn_wv)
    myDWV = from_gpu(arrays_and_views, 'DWV')
    assert(isclose(myDWV, refDWV))

    refDWQ = np.einsum("HPN->%s" % layouts['DWQ'], dattn_wq)
    myDWQ = from_gpu(arrays_and_views, 'DWQ')
    assert(isclose(myDWQ, refDWQ))

    refDWK = np.einsum("HPN->%s" % layouts['DWK'], dattn_wk)
    myDWK = from_gpu(arrays_and_views, 'DWK') 
    assert(isclose(myDWK, refDWK))    

    refDX = np.einsum("BJN->%s" % layouts['DX'], dx)
    myDX = from_gpu(arrays_and_views, 'DX')
    assert(isclose(myDX, refDX))

    ### END BACKWARD PASS

    print("All tests passed")

    # deallocate resources
    
    encoder.destroy(encoder_instance)
            
    for arr in arrays:
        memhelper.gpu_free(arrays[arr])

if __name__ == '__main__':
    test_encoder()
