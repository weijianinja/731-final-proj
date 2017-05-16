#include<iostream>
#include<math.h>
#include<string>
#include<iomanip>
#include<fstream>
#include<cstdlib>

using namespace std;

const int maxsize = 1000;

/*converts 5-point ordinal scale ratings to binary classes*/
int main(int argc, char *argv[])
{
	
	if ( argc < 2 ){
    	cerr << "enter two file names (input and output)\n";
	cerr << "number of ratings and number of annotators.\n";
	return 1;
	}
	int size  = atoi(argv[3]); 
	int annotator = atoi(argv[4]); 

	int ratings[size][annotator];
	

	ifstream inFile ( argv[1] );
	ofstream outFile ( argv[2] );
	ofstream outFile2;
	outFile2.open("avgratings.txt");
	float rowsum, avg;
	int val;
	int count = 0;
	for (int i = 0; i < size; i++){
	
	rowsum = 0;
	for (int j = 0; j < annotator; j++){
		count++;
		inFile >> val;
		rowsum += val;
	}
		avg = rowsum / annotator;
		outFile2 << avg << endl;
		if ( (int)((rowsum / annotator) + 0.5) < 3)
			outFile << "0" << endl;
		else
			outFile << "1" << endl;
			
	}
	
	cout << count << " values read and processed.\n";
	
	inFile.close();
	outFile.close();

	return 0;
}