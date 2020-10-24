#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void processArr();
//extern void computeArr(int * calcArr, int arrSize);
void createFile(int *calcArr, int arrSize);
extern void computeArr(float * metalRod, int arrSize, int timeSteps);

int main()
{
    int arrSize = 0;
    int timeSteps = 0;
    float location = 0;

    //FLUSH INPUT AND READ FILE NAME
    fflush(stdin);
    printf("Enter the amount of slices: ");
    scanf("%d", &arrSize);

    //FLUSH INPUT AND READ FILE NAME
    fflush(stdin);
    printf("Enter the amount of time steps: ");
    scanf("%d", &timeSteps);

    //FLUSH INPUT AND READ FILE NAME
    fflush(stdin);
    printf("Enter the location to measure: ");
    scanf("%f", &location);

    //BEGIN ARRAY CREATION
    processArr(arrSize, timeSteps, location);

    return 0;
}

void processArr(int arrSize, int timeSteps, double location)
{
    float *metalRod = NULL;
    float *metalRodCUDA = NULL;
    int i = 0;
    int j = 0;
    float stepSize = 0.0;
    int arrPos = 0;    

    metalRod = malloc(sizeof(float) * arrSize );
    metalRodCUDA = malloc(sizeof(float) * arrSize);

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
        }
    }

    //END CLOCK AND GET TIME
    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    

    stepSize = 1.0 / arrSize;
    arrPos = (location / stepSize);

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
}

void createFile(int *calcArr, int arrSize)
{
    //DECLARE VARS
    FILE *filep;
    int i = 0;

    //OPEN FILE
    filep = fopen("force.txt", "w+");

    //WRITE RESULTS TO FILE
    for (i = 0; i < arrSize; i ++)
    {
        fprintf(filep, "%d,", calcArr[i]);
    }

    //CLOSE FILE
    fclose(filep);

    //LET USER KNOW PROGRAM IS DONE
    printf("file created \n");
}

