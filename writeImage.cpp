// writeImage.cpp
// Collin Beaudoin November 2020
// program that uses a function to write a simple 2D image file

#include <iostream>
#include <fstream>
#include <cstring>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <iomanip>
using namespace std;

/*********************************************************
This function is used to take a given array and process it
to an image

@parameter fileName: The name of the file to create
@parameter dimX: used to denote the length of the image
@parameter dimY: used to denote the height of the image
@parameter data: The actual data used to create the image
@return: none
*********************************************************/

void writeImage(const char *fileName, int dimX, int dimY, unsigned char *data)
{
	//OPEN FILE
	ofstream f;
	f.open(fileName, fstream::out | fstream::binary);

	//DECLARE PPM VARS
	f << "P6" << endl;
	f << dimX << " " << dimY << endl;
	f << "255" << endl;

	//ITERATE OVER LENGTH OF ARRAY
	for (int x = 0; x < dimX; x++) {
		//ITERATE OVER HEIGHT OF ARRAY
		for (int y = 0; y < dimY * 3; y++) {
			f << data[(x * 3 * dimY) + y];	
		}
	}

	f.close();
}

/*********************************************************
This is the main function of the code, it is used to
upload the required information to create the image
*********************************************************/

int main()
{
	//DECLARE VARS
	int size = 768;
	unsigned char color[3];
    vector<float> v2;

	//OPEN HEAT FILE
	std::ifstream file("heat.txt");

	//CHECK THAT THE FILE OPENED
	if (file.is_open()) {
		std::string line;
		//RUN UNTIL FILE READ COMPLETELY
		while (std::getline(file, line)) {
            stringstream ss(line.c_str());
			//CLEAN STRING
            while (ss.good())
            {
                //DECLARE TEMP VARIABLE FOR STRING
                std::string substr;

                getline(ss, substr, ',');
				
				//CHECK FOR VALID VALUE AND SAVE IT
                if(!substr.empty())
                {
                    float tmp = std::stof(substr);
                    v2.push_back(tmp);
                }
            }
		}

		file.close();
	}

	//ALLOCATE MEMORY
    unsigned char *test = new unsigned char[size * v2.size() * 3];

	// generates data (a stream of bytes) representing a series of RED gradients
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < v2.size() * 3; j++) {
			color[0] = 255 * (v2[j] / 86);
			color[1] = 0;
			color[2] = 0;
			test[(i * 3 * size) + j] = color[j % 3];
		}
	}

	// pass data to function for writing to file
	writeImage("test.ppm", size, v2.size(), test);
	delete test;

	return 0;
}
