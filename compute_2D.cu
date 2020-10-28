// compute.cu
//
// driver and kernel call

#include <stdio.h>

#define THREADS_PER_BLOCK 128
 
// __global__ void compute_2d (int secondArrSize, float *arr[])
__global__ void compute_2d (int secondArrSize, float *arr[])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (x <= secondArrSize && y <= secondArrSize)
    // {
    //     arr[x][y] *= 2;
    // }
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
    //     __syncthreads();
	// }
		
}

extern "C" void compute2DArr (int firstArrSize, int secondArrSize, float *metalRod, int timeSteps)
{
    float *c_d = NULL;
    int i = 0, j = 0;

    int size=firstArrSize*secondArrSize*sizeof(float);
    //allocate resources
    float **cell=(float**)malloc(size);
    //float *d_cell; cudaMalloc(&d_cell,sizeof(float) * size * size);
    //float **d_cell;

    for (i = 0; i < size; i ++)
    {
        cell[i] = (float*)malloc(size);
    }
    
    for (i = 0; i < size; i ++)
    {
        for (j = 0; j < size; j ++)
        {
            cell[i][j] = 23.0;
        }
    }

    //initializeArray(node,N);
    //cudaMemcpy(d_cell,cell,size * size,cudaMemcpyHostToDevice);
    
    size_t pitch;
    float **d_cell;
    printf("\n \n \nhello \n \n \n");
    cudaMallocPitch((void**) &d_cell, &pitch, 128 * sizeof(float), size);
    printf("\n \n \nhello \n \n \n");
    cudaMemcpy2D(d_cell, pitch, cell, 256 * sizeof(float), secondArrSize * sizeof(float), firstArrSize, cudaMemcpyHostToDevice);
    printf("hello");
    //cudaMemcpy(d_node,node,size,cudaMemcpyHostToDevice);

    //compute_win2D<<<nblocks, nthreads>>>(d_node,d_cell);
    compute_2d<<<ceil((float) secondArrSize/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(secondArrSize, d_cell);

    // for (i = 0; i < size; i++)
    // {
    //     for (j = 0; j < size; j ++)
    //     {
    //         printf("%lf \n", d_cell[i][j]);
    //     }
    // }
    //free resources
    free(cell);
    //free(node);

    cudaFree(d_cell);
    //cudaFree(d_node);

    // printf("%d %d \n", firstArrSize, secondArrSize);

    // cudaMalloc (&tmpRod, sizeof(float) * firstArrSize );

    // for (i = 0; i < firstArrSize; i ++)
    //     cudaMalloc (&tmpRod[i], sizeof(float) * firstArrSize );

    // printf("hello");
    
    // //cudaMallocPitch((void**)&d_arr, &pitch, 256, 1024);
    // //cudaMallocPitch((void**) &c_d, &pitch, secondArrSize, firstArrSize);

    // // --- Copy array to device
    // //cudaMemcpy2D(c_d, pitch, tmpRod, 256 * sizeof(float), secondArrSize * sizeof(float), firstArrSize, cudaMemcpyHostToDevice);
    
    // //cudaMalloc ((void**) &c_d, sizeof(float) * arrSize);
    // //cudaMemcpy (c_d, metalRod, sizeof(float) * secondArrSize, cudaMemcpyHostToDevice);
    
    // for (i = 0; i < (2*(timeSteps - 1)) + secondArrSize; i ++)
    // {
    //     //compute_2d <<< ceil((float) secondArrSize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (c_d, secondArrSize, i, timeSteps);
    // }
    
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
    
    // //cudaMemcpy2D(metalRod, pitch, c_d, secondArrSize * sizeof(float), secondArrSize * sizeof(float), firstArrSize, cudaMemcpyDeviceToHost);
	// //cudaMemcpy (metalRod, c_d, sizeof(float) * arrSize, cudaMemcpyDeviceToHost);
    // //cudaFree (c_d);
    // cudaFree(tmpRod);
}

