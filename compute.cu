// compute.cu
// Collin Beaudoin November 2020
// driver and kernel call

#include <stdio.h>

/*********************************************************
This section is used to declare global variables
*********************************************************/

#define THREADS_PER_BLOCK 128

/*********************************************************
This is the kernel function of the code. This is where 
the GPU will be calculating the heat diffusion of the rod

@parameter c_d: This is the heat diffusion array
@parameter arrSize: This is the size of the array
@parameter timeStep: This is the current time step
@parameter timeSteps: This is the amount of steps to
calculate
*********************************************************/
 
__global__ void compute_d (float *c_d, int arrSize, int timeStep, int timeSteps)
{
    //DECLARE VARS
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    //CHECK THAT POSITION EXISTS
    if (x <= arrSize) {
        //CHECK IF THIS VALUE SHOULD BE CALCULATED CURRENTLY
        if (x % 2 == timeStep % 2 && x <= timeStep)
        {
            //SKIP IF THE CALCULATIONS ARE DONE FOR THIS SECTION
            if (timeStep > timeSteps && x <= (timeStep - timeSteps - arrSize))
            {
                
            } else 
            {
                //CALCULATE HEAT DIFFUSION BASED ON POSITION
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

/*********************************************************
This is the CUDA declaration of the GPU program. Here it
will set up the required memory for the GPU calculations
to run.

@parameter metalRod: This is rod to compute
@parameter arrSize: This is the size of the rod
@parameter timeSteps: This is the amount of steps to
calculate
*********************************************************/

extern "C" void computeArr (float *metalRod, int arrSize, int timeSteps)
{
    //DECLARE VARS
    float *c_d;
    int i = 0;
    
    //ALLOCATE MEMORY
    cudaMalloc ((void**) &c_d, sizeof(float) * arrSize);
    cudaMemcpy (c_d, metalRod, sizeof(float) * arrSize, cudaMemcpyHostToDevice);
    
    //RUN CALCULATIONS FOR REQUIRED AMOUNT OF STEPS
    for (i = 0; i < (2*(timeSteps - 1)) + arrSize; i++)
    {
        compute_d <<< ceil((float) arrSize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (c_d, arrSize, i, timeSteps);
    }
    
    //CHECK FOR ERRORS
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
    
    //RETURN ARRAY TO DEVICE
	cudaMemcpy (metalRod, c_d, sizeof(float) * arrSize, cudaMemcpyDeviceToHost);
	cudaFree (c_d);
}

