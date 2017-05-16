#include<iostream>
#include<math.h>
#include<string>
#include<iomanip>
#include<fstream>
#include<cstdlib>

using namespace std;

const int maxsize = 1000;
double stdev(double data[], int size);

/*takes as input two files (vectors), and number of items evaluated
first containing annotator ratings in binary form (avg or any other function)
second is the eval metric scores
computes the biserial correlation between the two 
*/
int main(int argc, char *argv[])
{
	
	if ( argc < 2 ){
    		cerr << "enter two file names and number of ratings.\n";
		return 1;
	}
	int size  = atoi(argv[3]); 
	int ratings[size];
	double metric[size];

	double ypopsdev; //=  0.3233224; //0.4313579;
	ifstream inFile1 ( argv[1] );
	ifstream inFile2 ( argv[2] );
	

	int val, v2;
	for (int i = 0; i < size; i++)
		inFile1 >> ratings[i];
	inFile1.close();
	
	for (int i = 0; i < size; i++)
		inFile2 >> metric[i] ;
	inFile2.close();
	
	//compute standard dev of metric values
	ypopsdev = stdev(metric,size);
	

	double p=0;
	double q=0;
	double y0=0;
	double y1=0;
	for (int i = 0; i < size; i++)
	{
		if (ratings[i] == 0){
			p++;
			y0 += metric[i];
		}
		else{
			q++;	
			y1 += metric[i];
		}
	}
	double y0mean = y0 / size;
	double y1mean = y1 / size;
	
	double result = (y0mean - y1mean)  / ypopsdev * sqrt((p*q)/(size*size)); 
	cout << result<< endl;
	return 0;
		
	
}

double stdev(double data[], int size)
{
    double sum = 0.0;
    double mean = 0.0;
    double standardDeviation = 0.0;

    int i;

    for(i = 0; i < size; ++i)
    {
        sum += data[i];
    }

    mean = sum/size;

    for(i = 0; i < size; ++i)
        standardDeviation += pow(data[i] - mean, 2);

    return sqrt(standardDeviation / size);
}
