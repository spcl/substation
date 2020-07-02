
.PHONY: test
test:
	python correctness_test.py

.PHONY: all
all: test perf

.PHONY: perf
perf:
	python performance_test.py

encoder: encoder.cu
	nvcc --compiler-options -Wfatal-errors -O3 -o encoder encoder.cu -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -lcublas

sample_encoder: sample_encoder.cu
	nvcc --compiler-options -Wfatal-errors -O3 -o sample_encoder sample_encoder.cu sample_encoder_runner.cu -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -lcublas
	
prototype70: main.cu
	nvcc -o prototype70 main.cu -gencode arch=compute_70,code=sm_70 -lcublas

prototype61: main.cu
	nvcc --compiler-options -Wfatal-errors -g -o prototype61 main.cu -gencode arch=compute_61,code=sm_61 -lcublas
	
prof-time.nvprof: prototype70
	nvprof -fo prof-time.nvprof ./prototype70
	
# transformers.o: transformers.cu
# 	nvcc -G -gencode arch=compute_61,code=sm_61 -c --compiler-options -fPIC transformers.cu -o transformers.o
# 	
# libtransformers.so: transformers.o
# 	nvcc -shared -o libtransformers.so transformers.o
	
.PHONY: clean
clean:
	rm -f prototype70 prototype61 *\.nvprof *\.o *\.so temp*
