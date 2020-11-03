// compute.cu
//
// driver and kernel call

#include <stdio.h>

#define THREADS_PER_BLOCK 128
 
// __global__ void compute_2d (int secondArrSize, float *arr[])
__global__ void compute_2d ( int firstArrSize, int secondArrSize, float **arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x == 0 && x < firstArrSize && y == 0 && y < firstArrSize)
    {
        printf("Hello. I'm a thread %d in block %d \n", threadIdx.x, blockIdx.x);
        // printf("%lf \n", arr[x][y]);
    }
    // if (x <= arrSize) {
    //     if (x % 2 == timeStep % 2 && x <= timeStep)
    //     {
    //         if (timeStep > timeSteps && x <= (timeStep - timeSteps - arrSize))
    //         {
                
    //         } else 
    //         {
    //             if (x == 0)
    //             {
    //                 c_d[x] = (100.0 + c_d[x + 1]) / 2.0;
    //             } else if (x == arrSize - 1)
    //             {
    //                 c_d[x] = (c_d[x - 1] + c_d[x]) / 2.0;
    //             } else
    //             {
    //                 c_d[x] = (c_d[x - 1] + c_d[x + 1]) / 2.0;
    //             }
    //         }
    //     }
        __syncthreads();
	// }
		
}

extern "C" void compute2DArr (int firstArrSize, int secondArrSize, float *metalRod, int timeSteps)
{
    int i = 0, j = 0;
    int size=firstArrSize*secondArrSize*sizeof(float);
    //allocate resources
    float **cell=(float**)malloc(size * 2);
    float **cell2=(float**)malloc(size * 2);
    
    for (i = 0; i < firstArrSize; i ++)
    {
        cell[i] = (float*)malloc(size);
        cell2[i] = (float*)malloc(size);
        for (j = 0; j < secondArrSize; j ++)
        {
            cell[i][j] = 23.0;
        }
    }
    
    size_t pitch;
    float **d_cell;

    cudaMallocPitch((void**) &d_cell, &pitch, secondArrSize * sizeof(float), firstArrSize);
    cudaError_t tmp = cudaMemcpy2D(d_cell, pitch, cell, secondArrSize * sizeof(float), secondArrSize * sizeof(float), firstArrSize, cudaMemcpyHostToDevice);

    if (cudaSuccess != tmp)
    {
        printf("\n copy to GPU \n");
        printf(cudaGetErrorString(tmp));
    }
    
    dim3 dimBlock(8,8);
    dim3 dimGrid(1,1);
    
    compute_2d<<<dimGrid, dimBlock>>>( firstArrSize, secondArrSize, d_cell);

    if (cudaSuccess != tmp)
    {
        printf("\n compute \n");
        printf(cudaGetErrorString(tmp));
    }

    tmp = cudaMemcpy2D(cell2, secondArrSize * sizeof(float), d_cell, pitch, secondArrSize * sizeof(float), firstArrSize, cudaMemcpyDeviceToHost);
    
    if (cudaSuccess != tmp)
    {
        printf("\n copy to CPU \n");
        printf(cudaGetErrorString(tmp));
    }

    for (i = 0; i < firstArrSize; i++)
    {
        for (j = 0; j < secondArrSize; j ++)
        {
            printf("\n %lf ", cell2[i][j]);
        }
    }
    
    // for (i = 0; i < (2*(timeSteps - 1)) + secondArrSize; i ++)
    // {
    //     //compute_2d <<< ceil((float) secondArrSize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (c_d, secondArrSize, i, timeSteps);
    // }
    
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
}

