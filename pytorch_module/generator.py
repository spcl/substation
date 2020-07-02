import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer

from substation.attention import softmax as ref_softmax

def generate_softmax(dims, reduce_dim, libname, reps=1):
    size = reduce(lambda x, y: x * y, dims.values())
    
    temp_source = """
    #include "blocks.cuh"
    
    #include <chrono>
    
    """ + dims_declaration
    
    dims_declaration = "\n".join(["struct %s { enum { value = %d }; };" % (d, dims[d]) for d in dims])
    
    temp_source += dims_declaration
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_out in itertools.permutations(dims):
            in_label = "".join(dims_permutation_in)
            out_label = "".join(dims_permutation_out)
            
            in_layout = ", ".join(dims_permutation_in)
            out_layout = ", ".join(dims_permutation_out)
            
            layouts_declaration = """
                using lIN = metal::list<%s>;
                using lOUT = metal::list<%s>;
            """ % (in_layout, out_layout)
            
            temp_source += """
                extern "C" {{
                    double temp_{in_label}_{out_label}(half* IN, half* OUT) {{
                        
                        half* gIN = nullptr;
                        half* gOUT = nullptr;
                        
                        CHECK(cudaMalloc(&gIN, {size} * sizeof(half)));
                        CHECK(cudaMemcpy(gIN, IN, {size} * sizeof(half), cudaMemcpyHostToDevice));
                        
                        CHECK(cudaMalloc(&gOUT, {size} * sizeof(half)));

                        {layouts_declaration}
                        
                        typedef std::chrono::high_resolution_clock Clock;
                        auto t1 = Clock::now();
                        for (int i = 0; i < {reps}; i++) {{
                            Softmax<lIN, lOUT, {reduce_dim}>::run(gIN, gOUT, (cudaStream_t)0);
                            CHECK(cudaStreamSynchronize(0));
                        }}
                        auto t2 = Clock::now();
                        
                        CHECK(cudaMemcpy(OUT, gOUT, {size} * sizeof(half), cudaMemcpyDeviceToHost));
                        
                        CHECK(cudaFree(gIN));
                        CHECK(cudaFree(gOUT));
                        
                        return std::chrono::duration<double, std::micro>(t2 - t1).count() / {reps};
                    }}
                }}
            """.format(
                layouts_declaration=layouts_declaration,
                in_label=in_label,
                out_label=out_label,
                size=size,
                reduce_dim=reduce_dim,
                reps=reps)
    
    with open("temp.cu", "w") as f:
        f.write(temp_source)

    subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC temp.cu -o temp.o".split(' '))
    subprocess.run("nvcc -shared -o {libname} temp.o".format(libname=libname).split(' '))
    
def generate_drln(dims, reduce_dim, libname, reps=1):
    if os.path.exists(libname):
        return
    
    size = reduce(lambda x, y: x * y, dims.values())
    reduce_size = dims[reduce_dim]

    dims_declaration = "\n".join(["struct %s { enum { value = %d }; };" % (d, dims[d]) for d in dims])
    
    temp_source = """
    #include "blocks.cuh"
    
    #include <chrono>
    
    """ + dims_declaration
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_out in itertools.permutations(dims):
            for dims_permutation_resid in itertools.permutations(dims):
                in_label = "".join(dims_permutation_in)
                out_label = "".join(dims_permutation_out)
                resid_label = "".join(dims_permutation_resid)
                
                in_layout = ", ".join(dims_permutation_in)
                out_layout = ", ".join(dims_permutation_out)
                resid_layout = ", ".join(dims_permutation_resid)
                
                layouts_declaration = """
                    using lIN = metal::list<%s>;
                    using lOUT = metal::list<%s>;
                    using lRESID = metal::list<%s>;
                """ % (in_layout, out_layout, resid_layout)
                
                func_name =  'temp_%s_%s_%s' % (out_label, in_label, resid_label)
                
                temp_source += """
                    extern "C" {{
                        double {func_name}(half* OUT, half* IN, half* RESID, half* SCALE, half* BIAS) {{
                            
                            half* gIN = nullptr;
                            half* gOUT = nullptr;
                            half* gRESID = nullptr;
                            half* gSCALE = nullptr;
                            half* gBIAS = nullptr;
                            
                            CHECK(cudaMalloc(&gIN, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gIN, IN, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gOUT, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gOUT, OUT, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gRESID, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gRESID, RESID, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gSCALE, {reduce_size} * sizeof(half)));
                            CHECK(cudaMemcpy(gSCALE, SCALE, {reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gBIAS, {reduce_size} * sizeof(half)));
                            CHECK(cudaMemcpy(gBIAS, BIAS, {reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gOUT, {size} * sizeof(half)));

                            {layouts_declaration}
                            
                            float dropoutProbability = 0.;
                            
                            typedef std::chrono::high_resolution_clock Clock;
                            auto t1 = Clock::now();
                            for (int i = 0; i < {reps}; i++) {{
                                BiasDropoutResidualLinearNorm<false, lOUT, lIN, lRESID, {reduce_dim}>::run(gOUT, gIN, gRESID, gSCALE, gBIAS, nullptr, dropoutProbability, (cudaStream_t)0);
                                CHECK(cudaStreamSynchronize(0));
                            }}
                            auto t2 = Clock::now();
                            
                            CHECK(cudaMemcpy(OUT, gOUT, {size} * sizeof(half), cudaMemcpyDeviceToHost));
                            
                            CHECK(cudaFree(gIN));
                            CHECK(cudaFree(gOUT));
                            CHECK(cudaFree(gRESID));
                            CHECK(cudaFree(gSCALE));
                            CHECK(cudaFree(gBIAS));
                            
                            return std::chrono::duration<double, std::micro>(t2 - t1).count() / {reps};
                        }}
                    }}
                """.format(
                    layouts_declaration=layouts_declaration,
                    func_name=func_name,
                    size=size,
                    reduce_dim=reduce_dim,
                    reps=reps,
                    reduce_size=reduce_size)

    
    with open("temp.cu", "w") as f:
        f.write(temp_source)

    subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC temp.cu -o temp.o".split(' '))
    subprocess.run("nvcc -shared -o {libname} temp.o".format(libname=libname).split(' '))

def generate_bdrln(dims, reduce_dim, libname, reps=1):
    if os.path.exists(libname):
        return
    
    size = reduce(lambda x, y: x * y, dims.values())
    reduce_size = dims[reduce_dim]

    dims_declaration = "\n".join(["struct %s { enum { value = %d }; };" % (d, dims[d]) for d in dims])
    
    temp_source = """
    #include "blocks.cuh"
    
    #include <chrono>
    
    """ + dims_declaration
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_out in itertools.permutations(dims):
            for dims_permutation_resid in itertools.permutations(dims):
                in_label = "".join(dims_permutation_in)
                out_label = "".join(dims_permutation_out)
                resid_label = "".join(dims_permutation_resid)
                
                in_layout = ", ".join(dims_permutation_in)
                out_layout = ", ".join(dims_permutation_out)
                resid_layout = ", ".join(dims_permutation_resid)
                
                layouts_declaration = """
                    using lIN = metal::list<%s>;
                    using lOUT = metal::list<%s>;
                    using lRESID = metal::list<%s>;
                """ % (in_layout, out_layout, resid_layout)
                
                func_name =  'temp_%s_%s_%s' % (out_label, in_label, resid_label)
                
                temp_source += """
                    extern "C" {{
                        double {func_name}(half* OUT, half* IN, half* RESID, half* SCALE, half* BIAS, half* LINEAR_BIAS) {{
                            
                            half* gIN = nullptr;
                            half* gOUT = nullptr;
                            half* gRESID = nullptr;
                            half* gSCALE = nullptr;
                            half* gBIAS = nullptr;
                            half* gLINEAR_BIAS = nullptr;
                            
                            CHECK(cudaMalloc(&gIN, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gIN, IN, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gOUT, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gOUT, OUT, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gRESID, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gRESID, RESID, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gSCALE, {reduce_size} * sizeof(half)));
                            CHECK(cudaMemcpy(gSCALE, SCALE, {reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gBIAS, {reduce_size} * sizeof(half)));
                            CHECK(cudaMemcpy(gBIAS, BIAS, {reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gLINEAR_BIAS, {reduce_size} * sizeof(half)));
                            CHECK(cudaMemcpy(gLINEAR_BIAS, LINEAR_BIAS, {reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gOUT, {size} * sizeof(half)));

                            {layouts_declaration}
                            
                            float dropoutProbability = 0.;
                            
                            typedef std::chrono::high_resolution_clock Clock;
                            auto t1 = Clock::now();
                            for (int i = 0; i < {reps}; i++) {{
                                BiasDropoutResidualLinearNorm<true, lOUT, lIN, lRESID, {reduce_dim}>::run(gOUT, gIN, gRESID, gSCALE, gBIAS, gLINEAR_BIAS, dropoutProbability, (cudaStream_t)0);
                                CHECK(cudaStreamSynchronize(0));
                            }}
                            auto t2 = Clock::now();
                            
                            CHECK(cudaMemcpy(OUT, gOUT, {size} * sizeof(half), cudaMemcpyDeviceToHost));
                            
                            CHECK(cudaFree(gIN));
                            CHECK(cudaFree(gOUT));
                            CHECK(cudaFree(gRESID));
                            CHECK(cudaFree(gSCALE));
                            CHECK(cudaFree(gBIAS));
                            CHECK(cudaFree(gLINEAR_BIAS));
                            
                            return std::chrono::duration<double, std::micro>(t2 - t1).count() / {reps};
                        }}
                    }}
                """.format(
                    layouts_declaration=layouts_declaration,
                    func_name=func_name,
                    size=size,
                    reduce_dim=reduce_dim,
                    reps=reps,
                    reduce_size=reduce_size)

    
    with open("temp.cu", "w") as f:
        f.write(temp_source)

    subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC temp.cu -o temp.o".split(' '))
    subprocess.run("nvcc -shared -o {libname} temp.o".format(libname=libname).split(' '))

def generate_bad(dims, reduce_dim, libname, reps=1):
    if os.path.exists(libname):
        return
    
    size = reduce(lambda x, y: x * y, dims.values())
    reduce_size = dims[reduce_dim]

    dims_declaration = "\n".join(["struct %s { enum { value = %d }; };" % (d, dims[d]) for d in dims])
    
    temp_source = """
    #include "blocks.cuh"
    
    #include <chrono>
    
    """ + dims_declaration
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_out in itertools.permutations(dims):
                in_label = "".join(dims_permutation_in)
                out_label = "".join(dims_permutation_out)
                
                in_layout = ", ".join(dims_permutation_in)
                out_layout = ", ".join(dims_permutation_out)
                
                layouts_declaration = """
                    using lIN = metal::list<%s>;
                    using lOUT = metal::list<%s>;
                """ % (in_layout, out_layout)
                
                func_name =  'temp_%s_%s' % (in_label, out_label)
                
                temp_source += """
                    extern "C" {{
                        double {func_name}(half* IN, half* OUT, half* BIAS) {{
                            
                            half* gIN = nullptr;
                            half* gOUT = nullptr;
                            half* gBIAS = nullptr;
                            
                            CHECK(cudaMalloc(&gIN, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gIN, IN, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gBIAS, {reduce_size} * sizeof(half)));
                            CHECK(cudaMemcpy(gBIAS, BIAS, {reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gOUT, {size} * sizeof(half)));

                            {layouts_declaration}
                            
                            float dropoutProbability = 0.;
                            
                            typedef std::chrono::high_resolution_clock Clock;
                            auto t1 = Clock::now();
                            for (int i = 0; i < {reps}; i++) {{
                                BiasActivationDropout<lIN, lOUT, {reduce_dim}>::run(gIN, gOUT, gBIAS, dropoutProbability, (cudaStream_t)0);
                                CHECK(cudaStreamSynchronize(0));
                            }}
                            auto t2 = Clock::now();
                            
                            CHECK(cudaMemcpy(OUT, gOUT, {size} * sizeof(half), cudaMemcpyDeviceToHost));
                            
                            CHECK(cudaFree(gIN));
                            CHECK(cudaFree(gOUT));
                            CHECK(cudaFree(gBIAS));
                            
                            return std::chrono::duration<double, std::micro>(t2 - t1).count() / {reps};
                        }}
                    }}
                """.format(
                    layouts_declaration=layouts_declaration,
                    func_name=func_name,
                    size=size,
                    reduce_dim=reduce_dim,
                    reps=reps,
                    reduce_size=reduce_size)

    
    with open("temp.cu", "w") as f:
        f.write(temp_source)

    subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC temp.cu -o temp.o".split(' '))
    subprocess.run("nvcc -shared -o {libname} temp.o".format(libname=libname).split(' '))
    
    
def generate_bsb(dims, reduce_dim, warp_reduce_dim, libname, reps=1):
    if os.path.exists(libname):
        return
    
    size = reduce(lambda x, y: x * y, dims.values())
    
    for d in dims:
        if d != reduce_dim and d != warp_reduce_dim:
            non_reduce_dim = d
    
    non_reduce_size = dims[non_reduce_dim]

    dims_declaration = "\n".join(["struct %s { enum { value = %d }; };" % (d, dims[d]) for d in dims])
    
    temp_source = """
    #include "blocks.cuh"
    
    #include <chrono>
    
    """ + dims_declaration
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_din in itertools.permutations(dims):
            for dims_permutation_dout in itertools.permutations(dims):
                in_label = "".join(dims_permutation_in)
                din_label = "".join(dims_permutation_din)
                dout_label = "".join(dims_permutation_dout)
                
                in_layout = ", ".join(dims_permutation_in)
                din_layout = ", ".join(dims_permutation_din)
                dout_layout = ", ".join(dims_permutation_dout)
                
                layouts_declaration = """
                    using lIN = metal::list<%s>;
                    using lDIN = metal::list<%s>;
                    using lDOUT = metal::list<%s>;
                """ % (in_layout, din_layout, dout_layout)
                
                func_name =  'temp_%s_%s_%s' % (in_label, din_label, dout_label)
                
                temp_source += """
                    extern "C" {{
                        double {func_name}(half* IN, half* SCALE, half* DOUT, half* DIN, half* DSCALE, half* DBIAS) {{
                            
                            half* gIN = nullptr;
                            half* gSCALE = nullptr;
                            half* gDOUT = nullptr;
                            half* gDIN = nullptr;
                            half* gDSCALE = nullptr;
                            half* gDBIAS = nullptr;
                            
                            CHECK(cudaMalloc(&gIN, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gIN, IN, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gSCALE, {non_reduce_size} * sizeof(half)));
                            CHECK(cudaMemcpy(gSCALE, SCALE, {non_reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            CHECK(cudaMalloc(&gDOUT, {size} * sizeof(half)));
                            CHECK(cudaMemcpy(gDOUT, DOUT, {size} * sizeof(half), cudaMemcpyHostToDevice));
                            
                            
                            CHECK(cudaMalloc(&gDIN, {size} * sizeof(half)));
                            CHECK(cudaMalloc(&gDSCALE, {non_reduce_size} * sizeof(half)));
                            CHECK(cudaMalloc(&gDBIAS, {non_reduce_size} * sizeof(half)));

                            {layouts_declaration}
                            
                            typedef std::chrono::high_resolution_clock Clock;
                            auto t1 = Clock::now();
                            for (int i = 0; i < {reps}; i++) {{
                                BackwardScaleBias<lIN, lDIN, lDOUT, {reduce_dim}, {warp_reduce_dim}>::run(gIN, gSCALE, gDOUT, gDIN, gDSCALE, gDBIAS, (cudaStream_t)0);
                                CHECK(cudaStreamSynchronize(0));
                            }}
                            auto t2 = Clock::now();
                            
                            CHECK(cudaMemcpy(DIN, gDIN, {size} * sizeof(half), cudaMemcpyDeviceToHost));
                            CHECK(cudaMemcpy(DSCALE, gDSCALE, {non_reduce_size} * sizeof(half), cudaMemcpyDeviceToHost));
                            CHECK(cudaMemcpy(DBIAS, gDBIAS, {non_reduce_size} * sizeof(half), cudaMemcpyDeviceToHost));
                            
                            CHECK(cudaFree(gIN));
                            CHECK(cudaFree(gSCALE));
                            CHECK(cudaFree(gDOUT));
                            CHECK(cudaFree(gDIN));
                            CHECK(cudaFree(gDSCALE));
                            CHECK(cudaFree(gDBIAS));
                            
                            return std::chrono::duration<double, std::micro>(t2 - t1).count() / {reps};
                        }}
                    }}
                """.format(
                    layouts_declaration=layouts_declaration,
                    func_name=func_name,
                    size=size,
                    reduce_dim=reduce_dim,
                    warp_reduce_dim=warp_reduce_dim,
                    reps=reps,
                    non_reduce_size=non_reduce_size)

    
    with open("temp.cu", "w") as f:
        f.write(temp_source)

    subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC temp.cu -o temp.o".split(' '))
    subprocess.run("nvcc -shared -o {libname} temp.o".format(libname=libname).split(' '))
    

def generate_blnrd(dims, nothing_dim, vec_dim, reduce_dim, libname, reps=1):
    if os.path.exists(libname):
        return
    
    size = reduce(lambda x, y: x * y, dims.values())
    
    non_reduce_size = dims[nothing_dim] * dims[vec_dim]

    dims_declaration = "\n".join(["struct %s { enum { value = %d }; };" % (d, dims[d]) for d in dims])
    
    temp_source = """
    #include "blocks.cuh"
    
    #include <chrono>
    
    """ + dims_declaration
    
    for dims_permutation_dout in itertools.permutations(dims):
        for dims_permutation_std in itertools.permutations([nothing_dim, vec_dim]):
            for dims_permutation_diff in itertools.permutations(dims):
                for dims_permutation_drop_mask in itertools.permutations(dims):
                    for dims_permutation_d_ln_in in itertools.permutations(dims):
                        for dims_permutation_d_drop_in in itertools.permutations(dims):
                            dout_label = "".join(dims_permutation_dout)
                            std_label = "".join(dims_permutation_std)
                            diff_label = "".join(dims_permutation_diff)
                            drop_mask_label = "".join(dims_permutation_drop_mask)
                            d_ln_in_label = "".join(dims_permutation_d_ln_in)
                            d_drop_in_label = "".join(dims_permutation_d_drop_in)
                            
                            dout_layout = ", ".join(dims_permutation_dout)
                            std_layout = ", ".join(dims_permutation_std)
                            diff_layout = ", ".join(dims_permutation_diff)
                            drop_mask_layout = ", ".join(dims_permutation_drop_mask)
                            d_ln_in_layout = ", ".join(dims_permutation_d_ln_in)
                            d_drop_in_layout = ", ".join(dims_permutation_d_drop_in)
                            
                            layouts_declaration = """
                                using lDOUT = metal::list<%s>;
                                using lSTD = metal::list<%s>;
                                using lDIFF = metal::list<%s>;
                                using lDROP_MASK = metal::list<%s>;
                                using lD_LN_IN = metal::list<%s>;
                                using lD_DROP_IN = metal::list<%s>;
                            """ % (dout_layout, std_layout, diff_layout, drop_mask_layout, d_ln_in_layout, d_drop_in_layout)
                            
                            func_name =  'temp_%s_%s_%s_%s_%s_%s' % (dout_label, std_label, diff_label, drop_mask_label, d_ln_in_label, d_drop_in_label)
                            
                            temp_source += """
                                extern "C" {{
                                    double {func_name}(half* DOUT, half* STD, half* DIFF, half* DROP_MASK, half* D_LN_IN, half* D_DROP_IN) {{
                                        
                                        half* gDOUT = nullptr;
                                        half* gSTD = nullptr;
                                        half* gDIFF = nullptr;
                                        half* gDROP_MASK = nullptr;
                                        half* gD_LN_IN = nullptr;
                                        half* gD_DROP_IN = nullptr;
                                        
                                        CHECK(cudaMalloc(&gDOUT, {size} * sizeof(half)));
                                        CHECK(cudaMemcpy(gDOUT, DOUT, {size} * sizeof(half), cudaMemcpyHostToDevice));
                                        
                                        CHECK(cudaMalloc(&gSTD, {non_reduce_size} * sizeof(half)));
                                        CHECK(cudaMemcpy(gSTD, STD, {non_reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
                                        
                                        CHECK(cudaMalloc(&gDIFF, {size} * sizeof(half)));
                                        CHECK(cudaMemcpy(gDIFF, DIFF, {size} * sizeof(half), cudaMemcpyHostToDevice));
                                        
                                        CHECK(cudaMalloc(&gDROP_MASK, {size} * sizeof(half)));
                                        CHECK(cudaMemcpy(gDROP_MASK, DROP_MASK, {size} * sizeof(half), cudaMemcpyHostToDevice));
                                        
                                        CHECK(cudaMalloc(&gD_LN_IN, {size} * sizeof(half)));
                                        
                                        CHECK(cudaMalloc(&gD_DROP_IN, {size} * sizeof(half)));

                                        {layouts_declaration}
                                        
                                        typedef std::chrono::high_resolution_clock Clock;
                                        auto t1 = Clock::now();
                                        for (int i = 0; i < {reps}; i++) {{
                                            BackwardLayerNormResidualDropout<{nothing_dim}, {vec_dim}, {reduce_dim}, lDOUT, lSTD, lDIFF, lDROP_MASK, lD_LN_IN, lD_DROP_IN>
                                                ::run(gDOUT, gSTD, gDIFF, gDROP_MASK, gD_LN_IN, gD_DROP_IN, (cudaStream_t)0);
                                            CHECK(cudaStreamSynchronize(0));
                                        }}
                                        auto t2 = Clock::now();
                                        
                                        CHECK(cudaMemcpy(D_LN_IN, gD_LN_IN, {size} * sizeof(half), cudaMemcpyDeviceToHost));
                                        CHECK(cudaMemcpy(D_DROP_IN, gD_DROP_IN, {size} * sizeof(half), cudaMemcpyDeviceToHost));
                                        
                                        CHECK(cudaFree(gDOUT));
                                        CHECK(cudaFree(gSTD));
                                        CHECK(cudaFree(gDIFF));
                                        CHECK(cudaFree(gDROP_MASK));
                                        CHECK(cudaFree(gD_LN_IN));
                                        CHECK(cudaFree(gD_DROP_IN));
                                        
                                        return std::chrono::duration<double, std::micro>(t2 - t1).count() / {reps};
                                    }}
                                }}
                            """.format(
                                layouts_declaration=layouts_declaration,
                                func_name=func_name,
                                size=size,
                                nothing_dim=nothing_dim,
                                vec_dim=vec_dim,
                                reduce_dim=reduce_dim,
                                reps=reps,
                                non_reduce_size=non_reduce_size)
                            
                            break
                        break
                    break
                break
            break
        break

    
    with open("temp.cu", "w") as f:
        f.write(temp_source)

    subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC temp.cu -o temp.o".split(' '))
    subprocess.run("nvcc -shared -o {libname} temp.o".format(libname=libname).split(' '))
    
    
def generate_ebsb(dims, reduce_dim, warp_reduce_dim, non_reduce_dim,
                dims_permutation_in,
                dims_permutation_din,
                dims_permutation_dout1,
                dims_permutation_dout2,
                dims_permutation_dout12,
                dims_permutation_dlinear,
                libname, reps=1):
    
    size = reduce(lambda x, y: x * y, dims.values())
    
    non_reduce_size = dims[non_reduce_dim]

    dims_declaration = "\n".join(["struct %s { enum { value = %d }; };" % (d, dims[d]) for d in dims])
    
    temp_source = """
    #include "blocks.cuh"
    
    #include <chrono>
    
    """ + dims_declaration
    
    in_label = "".join(dims_permutation_in)
    din_label = "".join(dims_permutation_din)
    dout1_label = "".join(dims_permutation_dout1)
    dout2_label = "".join(dims_permutation_dout2)
    dout12_label = "".join(dims_permutation_dout12)
    dlinear_label = "".join(dims_permutation_dlinear)

    in_layout = ",".join(dims_permutation_in)
    din_layout = ",".join(dims_permutation_din)
    dout1_layout = ",".join(dims_permutation_dout1)
    dout2_layout = ",".join(dims_permutation_dout2)
    dout12_layout = ",".join(dims_permutation_dout12)
    dlinear_layout = ",".join(dims_permutation_dlinear)

    layouts_declaration = """
    using lIN = metal::list<%s>;
    using lDIN = metal::list<%s>;
    using lDOUT1 = metal::list<%s>;
    using lDOUT2 = metal::list<%s>;
    using lDOUT12 = metal::list<%s>;
    using lDLINEAR = metal::list<%s>;
    """ % (in_layout, din_layout, dout1_layout, dout2_layout, dout12_layout, dlinear_layout)

    func_name = 'temp_%s_%s_%s_%s_%s_%s' % (in_label, din_label, dout1_label, dout2_label, dout12_label, dlinear_label)

    temp_source += """
    extern "C" {{
    double {func_name}(half* IN, half* SCALE, half* DOUT1, half* DOUT2, half* DLINEAR,
                        half* DIN, half* DSCALE, half* DBIAS, half* DOUT12, half* DLINEAR_BIAS) {{
        
        half* gIN = nullptr;
        half* gSCALE = nullptr;
        half* gDOUT1 = nullptr;
        half* gDOUT2 = nullptr;
        half* gDLINEAR = nullptr;
        half* gDIN = nullptr;
        half* gDSCALE = nullptr;
        half* gDBIAS = nullptr;
        half* gDOUT12 = nullptr;
        half* gDLINEAR_BIAS = nullptr;
        
        CHECK(cudaMalloc(&gIN, {size} * sizeof(half)));
        CHECK(cudaMemcpy(gIN, IN, {size} * sizeof(half), cudaMemcpyHostToDevice));
        
        CHECK(cudaMalloc(&gSCALE, {non_reduce_size} * sizeof(half)));
        CHECK(cudaMemcpy(gSCALE, SCALE, {non_reduce_size} * sizeof(half), cudaMemcpyHostToDevice));
        
        CHECK(cudaMalloc(&gDOUT1, {size} * sizeof(half)));
        CHECK(cudaMemcpy(gDOUT1, DOUT1, {size} * sizeof(half), cudaMemcpyHostToDevice));
        
        CHECK(cudaMalloc(&gDOUT2, {size} * sizeof(half)));
        CHECK(cudaMemcpy(gDOUT2, DOUT2, {size} * sizeof(half), cudaMemcpyHostToDevice));
        
        CHECK(cudaMalloc(&gDLINEAR, {size} * sizeof(half)));
        CHECK(cudaMemcpy(gDLINEAR, DLINEAR, {size} * sizeof(half), cudaMemcpyHostToDevice));
        
        CHECK(cudaMalloc(&gDIN, {size} * sizeof(half)));
        
        CHECK(cudaMalloc(&gDSCALE, {non_reduce_size} * sizeof(half)));
        
        CHECK(cudaMalloc(&gDBIAS, {non_reduce_size} * sizeof(half)));
        
        CHECK(cudaMalloc(&gDOUT12, {size} * sizeof(half)));
        
        CHECK(cudaMalloc(&gDLINEAR_BIAS, {non_reduce_size} * sizeof(half)));

        {layouts_declaration}
        
        /*typename in_layout,
        typename din_layout,
        typename dout1_layout,
        typename dout2_layout,
        typename dout12_layout,
        typename dlinear_layout*/
        
        typedef std::chrono::high_resolution_clock Clock;
        auto t1 = Clock::now();
        for (int i = 0; i < {reps}; i++) {{
            ExtendedBackwardScaleBias<{reduce_dim}, {warp_reduce_dim}, {non_reduce_dim}, lIN, lDIN, lDOUT1, lDOUT2, lDOUT12, lDLINEAR>
                ::run(gIN, gSCALE, gDOUT1, gDOUT2, gDLINEAR,
                        gDIN, gDSCALE, gDBIAS, gDOUT12, gDLINEAR_BIAS, (cudaStream_t)0);
                
                
                //half* IN, half* SCALE, half* DOUT1, half* DOUT2, half* DLINEAR,
                //half* DIN, half* DSCALE, half* DBIAS, half* DOUT12, half* DLINEAR_BIAS
            CHECK(cudaStreamSynchronize(0));
        }}
        auto t2 = Clock::now();
        
        CHECK(cudaMemcpy(DIN, gDIN, {size} * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(DSCALE, gDSCALE, {non_reduce_size} * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(DBIAS, gDBIAS, {non_reduce_size} * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(DOUT12, gDOUT12, {size} * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(DLINEAR_BIAS, gDLINEAR_BIAS, {non_reduce_size} * sizeof(half), cudaMemcpyDeviceToHost));
        
        CHECK(cudaFree(gIN));
        CHECK(cudaFree(gSCALE));
        CHECK(cudaFree(gDOUT1));
        CHECK(cudaFree(gDOUT2));
        CHECK(cudaFree(gDLINEAR));
        CHECK(cudaFree(gDIN));
        CHECK(cudaFree(gDSCALE));
        CHECK(cudaFree(gDBIAS));
        CHECK(cudaFree(gDOUT12));
        CHECK(cudaFree(gDLINEAR_BIAS));
        
        return std::chrono::duration<double, std::micro>(t2 - t1).count() / {reps};
    }}
    }}
    """.format(
    layouts_declaration=layouts_declaration,
    func_name=func_name,
    size=size,
    reps=reps,
    reduce_dim=reduce_dim,
    warp_reduce_dim=warp_reduce_dim,
    non_reduce_dim=non_reduce_dim,
    non_reduce_size=non_reduce_size)
    
    with open("temp.cu", "w") as f:
        f.write(temp_source)
    
    subprocess.run("nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC temp.cu -o temp.o".split(' '))
    subprocess.run("nvcc -shared -o {libname} temp.o".format(libname=libname).split(' '))
    
