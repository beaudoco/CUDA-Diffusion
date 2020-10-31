// program that uses a function to write a simple 2D image file
// demonstrates structure of the '.ppm' RGB file format

#include <iostream>
#include <fstream>
#include <cstring>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <iomanip>
using namespace std;

void writeImage(const char *fileName, int dimX, int dimY, unsigned char *data)
{
	ofstream f;
	f.open(fileName, fstream::out | fstream::binary);

	f << "P6" << endl;
	f << dimX << " " << dimX << endl;
	f << "255" << endl;

	for (int x = 0; x < dimX; x++) {
		for (int y = 0; y < dimY * 3; y++) {
			f << data[(x * 3 * dimY) + y];	
		}
	}

	f.close();
}

int main()
{
	int size = 768;
	//unsigned char *test = new unsigned char[size * size * 3];
	unsigned char color[3];
    vector<float> v2;

	std::ifstream file("heat.txt");
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
            stringstream ss(line.c_str());

            while (ss.good())
            {
                /* code */
                std::string substr;

                getline(ss, substr, ',');

                if(!substr.empty())
                {
                    float tmp = std::stof(substr);
                    v2.push_back(tmp);
                }

                // cout << substr << endl;
            }

            // for(size_t i = 0; i < v2.size(); i++)
            // {
            //     cout << v2[i] << endl;
            // }
		}

		file.close();
	}

    unsigned char *test = new unsigned char[size * v2.size() * 3];

	// generates data (a stream of bytes) representing a series of RED gradients
	for (int i = 0; i < size; i++) {
		//for (int j = 0; j < size * 3; j++) {
        for (int j = 0; j < v2.size() * 3; j++) {
			color[0] = j % 256;
			color[1] = 0;
			color[2] = 0;
			test[(i * 3 * v2.size()) + j] = color[j % 3];
		}
	}

	// pass data to function for writing to file
	writeImage("test.ppm", size, v2.size(), test);
	delete test;

	return 0;
}
