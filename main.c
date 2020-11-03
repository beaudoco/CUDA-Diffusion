/*********************************************************
This program takes a given dimension, partition size and
and ammount of time steps to run for. These values are 
used to calculate the heat diffusion of a 1 meter rod
over time.

@author: Collin Beaudoin
@version: November 2020
*********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*********************************************************
This section is used to declare the methods that shall be
used throughout the program.
*********************************************************/

void processArr();
void process2DArr();
void createFile(float *calcArr, int arrSize);
extern void computeArr(float * metalRod, int arrSize, int timeSteps);
extern void compute2DArr(int firstArrSize, int secondArrSize, float *metalRod, int timeSteps);

/*********************************************************
This is the main function of the code, it is used to
accept the parameters that shall be used throughout the
program.

@parameter arrType: This is used to select what 
dimension the calculations will be ran in
@parameter firstArrSize: This is used to figure out the
length of the rod.
@parameter secondArrSize: This is used to figure out the
height of the rod.
@parameter timeSteps: This is used to figure out the
length of time to run the program for
@parameter location: This is used to locate an exact 
position and its temperature
*********************************************************/

int main()
{
    //DECLARE VARS
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
        //FLUSH INPUT AND READ SLICES
        fflush(stdin);
        printf("Enter the amount of slices: ");
        scanf("%d", &secondArrSize);

        //FLUSH INPUT AND READ STEPS
        fflush(stdin);
        printf("Enter the amount of time steps: ");
        scanf("%d", &timeSteps);

        //FLUSH INPUT AND READ LOCATION
        fflush(stdin);
        printf("Enter the location to measure: ");
        scanf("%f", &location);

        //BEGIN ARRAY CREATION
        processArr(secondArrSize, timeSteps, location);
    } else
    {
        //FLUSH INPUT AND READ SLICES
        fflush(stdin);
        printf("Enter the height amount of slices: ");
        scanf("%d", &firstArrSize);

        //FLUSH INPUT AND READ LENGTH
        fflush(stdin);
        printf("Enter the length amount of slices: ");
        scanf("%d", &secondArrSize);

        //FLUSH INPUT AND READ TIME STEPS
        fflush(stdin);
        printf("Enter the amount of time steps: ");
        scanf("%d", &timeSteps);

        //FLUSH INPUT AND READ LOCATION
        fflush(stdin);
        printf("Enter the location to measure: ");
        scanf("%f", &location);

        //BEGIN ARRAY CREATION
        process2DArr(firstArrSize, secondArrSize, timeSteps, location);
    }

    return 0;
}

/*********************************************************
This function is used to iterate over the entire rod for
the given amount of time. This is where the computation
of the heat diffusion occurs

@parameter timeSteps: The time to run calculations for
@parameter arrSize: The overall size of the 1D array
@parameter location: The location to measure
@return: none
*********************************************************/

void processArr(int arrSize, int timeSteps, double location)
{
    //DECLARE VARS
    float *metalRod = NULL;
    float *metalRodCUDA = NULL;
    float *heatMap = NULL;
    int i = 0;
    int j = 0;
    int k = 0;
    float stepSize = 0.0;
    int arrPos = 0;    

    //ALLOCATE MEMORY FOR ARRAYS
    metalRod = malloc(sizeof(float) * arrSize );
    metalRodCUDA = malloc(sizeof(float) * arrSize);
    heatMap = malloc(sizeof(float) * (timeSteps + 1));

    //CALCULATE POSITION TO MEASURE
    stepSize = 1.0 / arrSize;
    arrPos = (location / stepSize);

    //SETUP TIMER FOR FILE
    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, &begin);

    //DO FOR THE AMOUNT OF REQUIRED TIME STEPS
    for (i = 0; i < timeSteps + 1; i ++)
    {
        //DO FOR THE ENTIRE ARRAY
        for (j = 0; j < arrSize; j ++)
        {
            //CHECK IF INITIAL ARRAY AND SET TEMPERATURE
            if (i == 0)
            {
                metalRod[j] = 23.0;
                metalRodCUDA[j] = 23.0;
            } else
            {
                //CHECK WHERE ON THE ARRAY THE VALUE IS
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

            //CREATING A SMALL SNAP SHOT OF DATA FOR IMAGES
            if(j == arrPos && i % 10000 == 0)
            {
                heatMap[k++] = metalRod[j];
            }
        }
    }

    //END CLOCK AND GET TIME
    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    //DISPLAY RESULTS
    printf("%lf \n", metalRod[arrPos]);

    free(metalRod);

    //SETUP TIMER FOR FILE
    struct timespec begin2, end2;
    clock_gettime(CLOCK_REALTIME, &begin2);

    //CALL CUDA PROGRAM
    computeArr(metalRodCUDA, arrSize, timeSteps);

    //END CLOCK AND GET TIME
    clock_gettime(CLOCK_REALTIME, &end2);
    long seconds2 = end2.tv_sec - begin2.tv_sec;
    long nanoseconds2 = end2.tv_nsec - begin2.tv_nsec;
    double elapsed2 = seconds2 + nanoseconds2*1e-9;

    //DISPLAY RESULTS
    printf("%lf \n", metalRodCUDA[arrPos]);
    printf("time taken for CPU: %f\n",elapsed);
    printf("time taken for GPU: %f\n",elapsed2);

    free(metalRodCUDA);

    //CREATE FILE THAT IS USED TO CREATE HEATMAP
    createFile(heatMap, timeSteps/10000);
}

/*********************************************************
This function is used to iterate over the entire rod for
the given amount of time. This is where the computation
of the heat diffusion occurs

@parameter timeSteps: The time to run calculations for
@parameter firstArrSize: The overall height of the rod
@parameter secondArrSize: The overall length of the rod
@parameter location: The location to measure
@return: none
*********************************************************/

void process2DArr(int firstArrSize, int secondArrSize, int timeSteps, double location)
{
    //DECLARE VARS
    float metalRod[firstArrSize][secondArrSize];
    float metalRodCUDA[firstArrSize][secondArrSize];
    int i = 0, j = 0, k = 0;
    float stepSize = 0.0;
    int firsArrPos = 0, secondArrPos = 0;    

    //SETUP TIMER FOR FILE
    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, &begin);

    //DO UNTIL THE END OF TIME LIMIT
    for (i = 0; i < timeSteps + 1; i ++)
    {
        //CHECK HEIGHT
        if (firstArrSize > 1)
        {
            //DO FOR THE HEIGHT OF THE ARRAY
            for (j = 0; j < firstArrSize; j ++)
            {
                //CREATE HOLDER FOR ARRAY
                float tmpArr[secondArrSize];

                //DO FOR THE LENGTH OF THE ARRAY
                for (k = 0; k < secondArrSize; k ++)
                {
                    //CHECK POSITION OF ARRAY AND SET VALUES
                    if (i == 0)
                    {
                        metalRod[j][k] = 23.0;
                        metalRodCUDA[j][k] = 23.0;
                    } else
                    {
                        //ASSIGN CORRECT VALUE BASED ON POSITION
                        if (j == 0 && k == 0)
                        {
                            tmpArr[k] = (100.0 + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                        } else if (j == 0 && k == secondArrSize - 1)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                        } else if (j == 0)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j][k]) / 4.0;
                        } else if (j == firstArrSize - 1 && k == 0)
                        {
                            tmpArr[k] = (100.0 + metalRod[j][k + 1] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                        } else if (j == firstArrSize - 1 && k == secondArrSize - 1)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                        } else if (j == firstArrSize - 1)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j][k] + metalRod[j - 1][k]) / 4.0;
                        } else if (k == 0)
                        {
                            tmpArr[k] = (100.0 + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                        } else if (k == secondArrSize - 1)
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                        } else
                        {
                            tmpArr[k] = (metalRod[j][k - 1] + metalRod[j][k + 1] + metalRod[j + 1][k] + metalRod[j - 1][k]) / 4.0;
                        }
                    }
                }
                if (i > 0)
                {
                    //ADD VALUES TO THE ROD
                    for (k = 0; k < secondArrSize; k ++)
                    {
                        metalRod[j][k] = tmpArr[k];
                    }
                }
            }
        } else
        {
            //THEY MADE A 1D ARRAY
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

    //CALCULATE WHERE TO MEASURE
    stepSize = 1.0 / firstArrSize;
    firsArrPos = (location / stepSize);

    //CALCULATE WHERE TO MEASURE
    stepSize = 1.0 / secondArrSize;
    secondArrPos = (location / stepSize);

    //PRINT RESULTS
    printf("%lf \n", metalRod[firsArrPos][secondArrPos]);

    //SETUP TIMER FOR FILE
    // struct timespec begin2, end2;
    // clock_gettime(CLOCK_REALTIME, &begin2);

    float *tmpCUDARod = NULL;
    tmpCUDARod = malloc(sizeof(float) * firstArrSize * secondArrSize );

    compute2DArr(firstArrSize, secondArrSize, tmpCUDARod, timeSteps);

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

/*********************************************************
This function is used to output the calculated heat of the
given position

@parameter complexityArr: The file data to be output.
@parameter arrLen: The overall size of the file.
@return: none
*********************************************************/

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
