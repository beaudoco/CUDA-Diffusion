#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void processArr();
void process2DArr();
//extern void computeArr(int * calcArr, int arrSize);
void createFile(float *calcArr, int arrSize);
extern void computeArr(float * metalRod, int arrSize, int timeSteps);
//extern void compute2DArr(float * metalRod, int firstArrSize, int secondArrSize, int timeSteps);
extern void compute2DArr(int firstArrSize, int secondArrSize, float *metalRod, int timeSteps);
//extern void compute2DArr(int firstArrSize, int secondArrSize, float **metalRod, int timeSteps);

int main()
{
    int firstArrSize = 0, secondArrSize = 0;
    int timeSteps = 0;
    int arrType = 1;
    float location = 0;

    //FLUSH INPUT AND READ FILE NAME
    fflush(stdin);
    printf("Enter the array type (1 == 1D, 2 == 2D): ");
    scanf("%d", &arrType);

    if (arrType == 1)
    {
        //FLUSH INPUT AND READ FILE NAME
        fflush(stdin);
        printf("Enter the amount of slices: ");
        scanf("%d", &secondArrSize);

        //FLUSH INPUT AND READ FILE NAME
        fflush(stdin);
        printf("Enter the amount of time steps: ");
        scanf("%d", &timeSteps);

        //FLUSH INPUT AND READ FILE NAME
        fflush(stdin);
        printf("Enter the location to measure: ");
        scanf("%f", &location);

        //BEGIN ARRAY CREATION
        processArr(secondArrSize, timeSteps, location);
    } else
    {
        //FLUSH INPUT AND READ FILE NAME
        fflush(stdin);
        printf("Enter the height amount of slices: ");
        scanf("%d", &firstArrSize);

        //FLUSH INPUT AND READ FILE NAME
        fflush(stdin);
        printf("Enter the length amount of slices: ");
        scanf("%d", &secondArrSize);

        //FLUSH INPUT AND READ FILE NAME
        fflush(stdin);
        printf("Enter the amount of time steps: ");
        scanf("%d", &timeSteps);

        //FLUSH INPUT AND READ FILE NAME
        fflush(stdin);
        printf("Enter the location to measure: ");
        scanf("%f", &location);

        //BEGIN ARRAY CREATION
        process2DArr(firstArrSize, secondArrSize, timeSteps, location);
    }

    return 0;
}

void processArr(int arrSize, int timeSteps, double location)
{
    float *metalRod = NULL;
    float *metalRodCUDA = NULL;
    float *heatMap = NULL;
    int i = 0;
    int j = 0;
    float stepSize = 0.0;
    int arrPos = 0;    

    metalRod = malloc(sizeof(float) * arrSize );
    metalRodCUDA = malloc(sizeof(float) * arrSize);
    heatMap = malloc(sizeof(float) * (timeSteps + 1));

    stepSize = 1.0 / arrSize;
    arrPos = (location / stepSize);

    //SETUP TIMER FOR FILE
    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, &begin);

    for (i = 0; i < timeSteps + 1; i ++)
    {
        for (j = 0; j < arrSize; j ++)
        {
            if (i == 0)
            {
                metalRod[j] = 23.0;
                metalRodCUDA[j] = 23.0;
            } else
            {
                if (j == 0)
                {
                    metalRod[j] = (100.0 + metalRod[j + 1]) / 2.0;
                } else if (j == arrSize - 1)
                {
                    metalRod[j] = (metalRod[j - 1] + metalRod[j]) / 2.0;
                } else
                {
                    metalRod[j] = (metalRod[j - 1] + metalRod[j + 1]) / 2.0;
                }
            }

            if(j == arrPos)
            {
                heatMap[i] = metalRod[j];
            }
        }
    }

    //END CLOCK AND GET TIME
    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    printf("%lf \n", metalRod[arrPos]);

    free(metalRod);

    //SETUP TIMER FOR FILE
    struct timespec begin2, end2;
    clock_gettime(CLOCK_REALTIME, &begin2);

    computeArr(metalRodCUDA, arrSize, timeSteps);

    //END CLOCK AND GET TIME
    clock_gettime(CLOCK_REALTIME, &end2);
    long seconds2 = end2.tv_sec - begin2.tv_sec;
    long nanoseconds2 = end2.tv_nsec - begin2.tv_nsec;
    double elapsed2 = seconds2 + nanoseconds2*1e-9;

    printf("%lf \n", metalRodCUDA[arrPos]);

    printf("time taken for CPU: %f\n",elapsed);
    printf("time taken for GPU: %f\n",elapsed2);

    free(metalRodCUDA);

    createFile(heatMap, timeSteps);
}

void process2DArr(int firstArrSize, int secondArrSize, int timeSteps, double location)
{
    float metalRod[firstArrSize][secondArrSize];
    float metalRodCUDA[firstArrSize][secondArrSize];
    int i = 0, j = 0, k = 0;
    float stepSize = 0.0;
    int firsArrPos = 0, secondArrPos = 0;    

    //SETUP TIMER FOR FILE
    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, &begin);

    for (i = 0; i < timeSteps + 1; i ++)
    {
        if (firstArrSize > 1)
        {
            for (j = 0; j < firstArrSize; j ++)
            {
                float tmpArr[secondArrSize];
                for (k = 0; k < secondArrSize; k ++)
                {
                    if (i == 0)
                    {
                        metalRod[j][k] = 23.0;
                        metalRodCUDA[j][k] = 23.0;
                    } else
                    {
                        if (j == 0 && k == 0)
                        {
                            tmpArr[k] = (100.0 + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                            //metalRod[j][k] = (100.0 + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                        } else if (j == 0 && k == secondArrSize - 1)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                            //metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                        } else if (j == 0)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                            //metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                        } else if (j == firstArrSize - 1 && k == 0)
                        {
                            tmpArr[k] = (100.0 + metalRod[j][k + 1] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                            //metalRod[j][k] = (100.0 + metalRod[j][k + 1] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                        } else if (j == firstArrSize - 1 && k == secondArrSize - 1)
                        {
                            // printf("current: %lf \n", metalRod[j][k]);
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                            //metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                            // printf("left: %lf \n", metalRod[j][k - 1]);
                            // printf("top: %lf \n", metalRod[j - 1][k]);
                        } else if (j == firstArrSize - 1)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                            //metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                        } else if (k == 0)
                        {
                            tmpArr[k] = (100.0 + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                            //metalRod[j][k] = (100.0 + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                        } else if (k == secondArrSize - 1)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                            //metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                        } else
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                            //metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                        }
                    }
                }
                if (i > 0)
                {
                    for (k = 0; k < secondArrSize; k ++)
                    {
                        metalRod[j][k] = tmpArr[k];
                    }
                }
            }
        } else
        {
            if (k == 0)
            {
                metalRod[j][k] = (100.0 + metalRod[j][k + 1]) / 2.0;
            } else if (k == secondArrSize - 1)
            {
                metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k]) / 2.0;
            } else
            {
                metalRod[j][k] = (metalRod[j][k - 1] + metalRod[j][k + 1]) / 2.0;
            }
        }
    }

    //END CLOCK AND GET TIME
    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    stepSize = 1.0 / firstArrSize;
    firsArrPos = (location / stepSize);

    stepSize = 1.0 / secondArrSize;
    secondArrPos = (location / stepSize);
    
    // for (i = 0; i < firstArrSize; i ++)
    // {
    //     printf("\n");
    //     for (j = 0; j < secondArrSize; j ++)
    //     {
    //         printf("%lf \n", metalRod[i][j]);
    //     }
    // }

    printf("%lf \n", metalRod[firsArrPos][secondArrPos]);

    //free(metalRod);

    //SETUP TIMER FOR FILE
    // struct timespec begin2, end2;
    // clock_gettime(CLOCK_REALTIME, &begin2);

    float *tmpCUDARod = NULL;
    tmpCUDARod = malloc(sizeof(float) * firstArrSize * secondArrSize );

    compute2DArr(firstArrSize, secondArrSize, tmpCUDARod, timeSteps);
    //compute2DArr(firstArrSize, secondArrSize, metalRodCUDA, timeSteps);

    //END CLOCK AND GET TIME
    // clock_gettime(CLOCK_REALTIME, &end2);
    // long seconds2 = end2.tv_sec - begin2.tv_sec;
    // long nanoseconds2 = end2.tv_nsec - begin2.tv_nsec;
    // double elapsed2 = seconds2 + nanoseconds2*1e-9;

    //printf("%lf \n", tmpCUDARod[firsArrPos][secondArrPos]);

    printf("time taken for CPU: %f\n",elapsed);
    //printf("time taken for GPU: %f\n",elapsed2);
    //free(tmpCUDARod);
    //free(metalRodCUDA);
}

void createFile(float *calcArr, int arrSize)
{
    //DECLARE VARS
    FILE *filep;
    int i = 0;

    //OPEN FILE
    filep = fopen("heat.txt", "w+");

    //WRITE RESULTS TO FILE
    for (i = 0; i < arrSize; i ++)
    {
        fprintf(filep, "%lf,", calcArr[i]);
    }

    //CLOSE FILE
    fclose(filep);

    //LET USER KNOW PROGRAM IS DONE
    printf("file created \n");
}
