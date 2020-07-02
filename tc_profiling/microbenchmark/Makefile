.PHONY: all

all: tcores

tcores: tensorcores.cu
	nvcc -arch sm_70 -O3 tensorcores.cu -o tcores

clean:
	rm -f *~ tcores
