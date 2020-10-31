// compute.cu
//
// driver and kernel call

#include <stdio.h>

#define THREADS_PER_BLOCK 32
 
__global__ void compute_d (float *c_d, int arrSize, int timeStep, int timeSteps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x <= arrSize) {
        if (x % 2 == timeStep % 2 && x <= timeStep)
        {
            if (timeStep > timeSteps && x <= (timeStep - timeSteps - arrSize))
            {
                
            } else 
            {
                if (x == 0)
                {
                    c_d[x] = (100.0 + c_d[x + 1]) / 2.0;
                } else if (x == arrSize - 1)
                {
                    c_d[x] = (c_d[x - 1] + c_d[x]) / 2.0;
                } else
                {
                    c_d[x] = (c_d[x - 1] + c_d[x + 1]) / 2.0;
                }
            }
        }
        __syncthreads();
	}
		
}

extern "C" void computeArr (float *metalRod, int arrSize, int timeSteps)
{
    float *c_d;
    int i = 0;
    
    cudaMalloc ((void**) &c_d, sizeof(float) * arrSize);
    cudaMemcpy (c_d, metalRod, sizeof(float) * arrSize, cudaMemcpyHostToDevice);
    
    for (i = 0; i < (2*(timeSteps - 1)) + arrSize; i++)
    {
        compute_d <<< ceil((float) arrSize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (c_d, arrSize, i, timeSteps);
    }
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
		
	cudaMemcpy (metalRod, c_d, sizeof(float) * arrSize, cudaMemcpyDeviceToHost);
	cudaFree (c_d);
}

