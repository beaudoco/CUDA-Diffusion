#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void processArr();
//extern void computeArr(int * calcArr, int arrSize);
void createFile(int *calcArr, int arrSize);

int main()
{
    int arrSize = 0;
    int timeSteps = 0;
    double location = 0;

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
    printf("Enter the location of the: ");
    scanf("%lf", &location);

    //BEGIN ARRAY CREATION
    processArr(arrSize, timeSteps, location);

    return 0;
}

void processArr(int arrSize, int timeSteps, double location)
{
    float *metalRod = NULL;
    int i = 0;
    int j = 0;
    float stepSize = 0.0;
    int arrPos = 0;

    metalRod = malloc(sizeof(float) * arrSize);

    for (i = 0; i < timeSteps + 1; i ++)
    {
        for (j = 0; j < arrSize; j ++)
        {
            if (i == 0)
            {
                metalRod[j] = 23.0;
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

    stepSize = 1.0 / arrSize;

    arrPos = (location / stepSize);

    printf("%lf \n", metalRod[arrPos]);
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

