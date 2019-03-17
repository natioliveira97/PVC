#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include "imageClass.hpp"

using namespace std;
using namespace cv;

#define boardHeight 6
#define boardWidth 8
#define squareSize 0.028 //Tamanho em m do lado do quadrado


void CallBackFunc(int event, int x, int y, int flags, void* userdata){
	imageClass* classe = (imageClass*) userdata;
    if  ( event == EVENT_LBUTTONDOWN ){
    	classe->click++;

        if((classe->click)%2){						//se for um clique ímpar, coloca as coordenadas obtidas nos pontos de início
        	classe->xi = x;
        	classe->yi = y;
        }
        else{										//se for um clique par, coloca as coordenadas obtidas nos pontos de final
        	classe->xf = x;
        	classe->yf = y;
        	classe->distanciaReal = true;
        	float D = (classe->xi-classe->xf)*(classe->xi-classe->xf) + (classe->yi-classe->yf)*(classe->yi-classe->yf);	//calcula a distância

        	D = sqrt(D);
        	classe->pixelDistance = D;
        	cout << "Distância em pixel entre os pontos em " << classe->windowsName << ": " << D << endl;
        }       	
    }
}

//Desenha linha na imagem
void desenhaLinha(string windowsName, imageClass* classe){

	setMouseCallback(windowsName, CallBackFunc, classe); 

	if(classe->click%2 == 0 && classe->click>0){

		Point inicio(classe->xi,classe->yi), final(classe->xf,classe->yf);
		if(windowsName == "Raw"){		
			line(classe->image, inicio, final, CV_RGB(0,0,255), 1, 8, 0);		
		}
		else{		
			line(classe->image, inicio, final, CV_RGB(255,0,0), 1, 8, 0);		
		}
	}
	imshow(windowsName, classe->image);
}

//Calcula a média do vetor de coeficiente de distorção.
vector<float> calculaMediaDistCoeff(vector<vector<float> > distCoeffArray, vector<float> distCoeff){
	for(int i = 0; i<distCoeff.size(); ++i){
		float sum = 0;
		for(int j = 0; j < distCoeffArray.size(); ++j){
			sum = sum + distCoeffArray[j][i];
		}
		distCoeff[i] = sum/distCoeffArray.size();
	}
	return distCoeff;
}

//Calcula o desvio padrão do vetor de coeficiente de distorção.
vector<float> calculaDesvioDistCoeff(vector<vector<float> > distCoeffArray, vector<float> distCoeff){
	vector<float> distCoeffDesvio;
	for(int i = 0; i<distCoeff.size(); ++i){
		float sum = 0;
		for(int j = 0; j<distCoeffArray.size(); ++j){
			sum = sum + pow(distCoeffArray[j][i]-distCoeff[i],2);
		}
		sum = sqrt(sum/distCoeffArray.size());
		distCoeffDesvio.push_back(sum);
	}
	return distCoeffDesvio;
}

//Calcula média da matriz de intrínsecos.
Mat calculaMedeaIntrinsics(vector<Mat> intrinsicArray, Mat intrinsic){
	Mat sum = Mat::zeros(3,3, CV_64FC1);
	for(int i = 0; i<intrinsicArray.size(); ++i){
		sum = sum + intrinsicArray[i];
	}
	sum = sum/intrinsicArray.size();
	intrinsic = sum;
	return intrinsic;
}

//Calcula desvio padrão da matriz de intrínsecos.
Mat calculaDesvioIntrisics(Mat intrinsic, vector<Mat> intrinsicArray, Mat intrinsicDesvio){
	Mat sum = Mat::zeros(3,3, CV_64FC1);
	for(int i = 0; i<intrinsicArray.size(); ++i){
		sum = sum + ((intrinsicArray[i]-intrinsic)*(intrinsicArray[i]-intrinsic));
	}
	sum = sum/intrinsicArray.size();
	sqrt(sum, intrinsicDesvio);
	return intrinsicDesvio;
}

//Calcula media de vetor tvecs e rvecs
Mat calculaMediaVect(vector<Mat> vetor){
	Mat sum = Mat::zeros(3,1,CV_64FC1);
	for(int i=0; i<vetor.size(); ++i){
		sum = sum+vetor[i];
	}
	sum = sum/vetor.size();
	return sum;
}

//Calcula desvio padrão de vetor tvecs e rvecs
Mat calculaDesvioVect(vector<Mat> vetor, Mat media, Mat tvecsDesvio){
	Mat sum = Mat::zeros(3,1,CV_64FC1);
	double data[3] = {0,0,0}, element;

	for(int i=0; i<vetor.size(); ++i){
		sum = vetor[i]-media;

		for(int j=0; j<3; ++j){
			element =  (double)sum.at<double>(j,0);
			element = element*element/vetor.size();
			data[j] = data[j]+element;
		}
	}

	sum = Mat(3,1,CV_64FC1, data);
	sqrt(sum, tvecsDesvio);
	return tvecsDesvio;
}

//Encontra a matriz de parametros intrínsecos e extrínsecos.
void findIntrinsic(Mat intrinsic, vector<float> *distCoeff){	

	VideoCapture capture(0);
	if(!capture.isOpened()){
        cout << "Não foi possível abrir a webcam!" << endl;
        return;
    }

	Mat frame;
	imageClass realFrame, undistortedFrame;
	int spanspots = 5;
	bool shot = false;
	int countSpanspots = 0;                                                
	Size patternSize = Size(boardWidth, boardHeight);
	vector<Point2f> corners;                      	//Vetor de pontos 2D na imagem referentes aos cantos dos quadrados
	vector<Point3f> realCorners;                  	//Vetor de pontos 3D no mundo referentes aos cantos dos quadrados
	Mat rvecs;
	Mat tvecs;
	bool patternWasFound;
	vector<vector<Point2f> > imagePoints;       	//Vetor de cantos na imagem
	vector<vector<Point3f> > objectPoints;			//Vetor de cantos no mundo

	//Preenche o vertor de coordenadas dos cantos no mundo, admitindo que o tabuleiro está no eixo z=0 no mundo e seus pontos são coplanares.
	for(int y = 0; y < boardHeight; ++y){
		for(int x = 0; x < boardWidth; ++x){
			realCorners.push_back(Point3f(x*squareSize, y*squareSize ,0));
		}
	}

	realFrame.windowsName = "Raw";
	undistortedFrame.windowsName = "Undistorted";
	namedWindow(realFrame.windowsName, WINDOW_AUTOSIZE);
	namedWindow(undistortedFrame.windowsName, WINDOW_AUTOSIZE);
	
	while(countSpanspots < 5){
		capture >> frame;
		realFrame.image = frame.clone();

		if(!distCoeff->empty()){
			undistort(frame, undistortedFrame.image, intrinsic, *distCoeff);
		}
		else{
			undistortedFrame.image = frame;
		}
		desenhaLinha(undistortedFrame.windowsName, &undistortedFrame);
		patternWasFound = findChessboardCorners(frame, patternSize, corners, CALIB_CB_FAST_CHECK);

		if(patternWasFound && shot){
			shot = false;
			++countSpanspots;
			cout <<"spanspot "<< countSpanspots<< endl;
			imagePoints.push_back(corners);
			objectPoints.push_back(realCorners);
		}

		drawChessboardCorners(realFrame.image, patternSize, corners, patternWasFound);
		desenhaLinha(realFrame.windowsName, &realFrame);

		if(waitKey(30)>=0){
			shot = true;
		}
	}
	calibrateCamera(objectPoints, imagePoints, realFrame.image.size(), intrinsic, *distCoeff, rvecs, tvecs, CV_CALIB_FIX_PRINCIPAL_POINT);
}

//Abre o streaming de video, desenha uma linha entre os pontos desejados e mostra a distância em pixels.
void requisito1(){
	system("clear");
	cout << "DISTÂNCIA ENTRE DOIS PONTOS" << endl << endl;

	VideoCapture capture(0);
	if(!capture.isOpened()){
		cout << "Error opening video stream or file" << endl;
	    return;
	}
	
	imageClass classe;
	classe.windowsName = "Camera";
	namedWindow(classe.windowsName, WINDOW_AUTOSIZE);

	while(1){
		capture >> classe.image;

		if(classe.image.empty()){
        	break;
		}

    	desenhaLinha(classe.windowsName, &classe);
    	imshow(classe.windowsName, classe.image); 

	    if(waitKey(25)==27){
	    	break;
	    }
	}

	capture.release();
	destroyAllWindows();
}

//Abre a câmera com as imagens 'Raw' e 'Undistorted', realiza a calibração dos parâmetros intrínsecos, calcula a média e o desvio padrão e escreve em arquivos.
void requisito2(){
	system("clear");
	cout << "CALIBRAÇÃO DOS PARÂMETROS INTRÍNSECOS" << endl << endl;
	cout << "Para encontrar a matriz de parâmetros intrinsecos é necessário tirar 5 fotos (spanspots). ";
	cout << "Mas para conseguir uma melhor calibração é necessário repetir esse procedimento 5 vezes." << endl << endl << endl;

	Mat intrinsic = Mat(3,3,CV_64FC1);
	Mat intrinsicDesvio;
	vector<Mat> intrinsicArray;
	vector<float> distCoeff;
	vector<vector<float> > distCoeffArray;
	vector<float> distCoeffDesvio;

	for(int i = 1; i <= 5; ++i){
		cout << "Procedimento " << i << endl;
		findIntrinsic(intrinsic, &distCoeff);
		intrinsicArray.push_back(intrinsic);
		distCoeffArray.push_back(distCoeff);
		intrinsic = calculaMedeaIntrinsics(intrinsicArray, intrinsic);
		distCoeff = calculaMediaDistCoeff(distCoeffArray, distCoeff);
	}
	distCoeffDesvio = calculaDesvioDistCoeff(distCoeffArray, distCoeff);
	intrinsicDesvio = calculaDesvioIntrisics(intrinsic, intrinsicArray, intrinsicDesvio);	

	//Escreve no arquivo
	FileStorage fs("intrinsics.xml", FileStorage::WRITE);
	if(!fs.isOpened()){
		cout << "Ocorreu um problema, não foi possível abrir o arquivo." << endl;
		return;
	}
	fs << "intrinsic" << intrinsic;
	fs << "desvioPadrao" << intrinsicDesvio;
	fs.release();

	fs.open("distortion.xml", FileStorage::WRITE);
	if(!fs.isOpened()){
		cout << "Ocorreu um problema, não foi possível abrir o arquivo." << endl;
		return;
	}
	fs << "distortion" << distCoeff;
	fs << "desvioPadrao" << distCoeffDesvio;
	fs.release();

	destroyAllWindows();
}

//Abre a câmera com as imagens 'Raw' e 'Undistorted' realiza a calibração dos parâmetros extrínsecos.
void requisito3(){
	int medidas;
	system("clear");
	cout << "CALIBRAÇÃO DOS PARÂMETROS EXTRÍNSECOS" << endl << endl;
	cout << "Para fazer uma medição, pressione qualquer tecla." << endl << endl;
	cout << "Quantas medições deseja fazer:";
	cin >> medidas;
	while (cin.get() != '\n');

	VideoCapture capture(0);
	if(!capture.isOpened()){
		cout << "Error opening video stream or file" << endl;
	    return;
	}

	Mat intrinsic;
	vector<float> distCoeff;
	Size patternSize = Size(boardWidth, boardHeight);
	vector<Point2f> corners;                      	//Vetor de pontos 2D na imagem referentes aos cantos dos quadrados
	vector<Point3f> realCorners;                 	//Vetor de pontos 3D no mundo referentes aos cantos dos quadrados
	vector<Mat> rvecsArray;
	vector<Mat> tvecsArray;
	Mat tvecsDesvio, rvecsDesvio;
	bool patternWasFound;
	bool shot = false;
	Mat frame, undistortedFrame;

	//Ler do arquivo
	FileStorage fs("intrinsics.xml", FileStorage::READ);
	if(!fs.isOpened()){
		cout << "Não foi possível abrir o arquivo, faça a calibração de intrínsecos primeiro" << endl;
		return;
	}
	fs["intrinsic"] >> intrinsic;
	fs.release();

	fs.open("distortion.xml", FileStorage::READ);
	if(!fs.isOpened()){
		cout << "Não foi possível abrir o arquivo, faça a calibração de intrínsecos primeiro" << endl;
		return;
	}
	fs["distortion"] >> distCoeff;
	fs.release();

	namedWindow("Undistorted", WINDOW_AUTOSIZE);
	namedWindow("Raw", WINDOW_AUTOSIZE);

	//Preenche o vetor de coordenadas dos cantos no mundo, admitindo que o tabuleiro está no eixo z=0 no mundo e seus pontos são coplanares.
	for(int y = 0; y < boardHeight; ++y){
		for(int x = 0; x < boardWidth; ++x){
			realCorners.push_back(Point3f(x*squareSize, y*squareSize ,0));
		}
	}

	for(int i=0; i<medidas;){
		Mat rvecs, tvecs;
		capture >> frame;
		undistort(frame, undistortedFrame, intrinsic, distCoeff);

		patternWasFound = findChessboardCorners(undistortedFrame, patternSize, corners, CALIB_CB_FAST_CHECK);
		drawChessboardCorners(undistortedFrame, patternSize, corners, patternWasFound);

		imshow("Raw", frame);
		imshow("Undistorted", undistortedFrame);

		if(patternWasFound && shot){
			shot = false;
			solvePnP(realCorners, corners, intrinsic, distCoeff, rvecs, tvecs, false, CV_ITERATIVE);
			double N;
			N = norm(tvecs, NORM_L2);
			cout << "Vetor de translação: " << endl << tvecs << endl; 
			cout << "Distância: " << N << "m" << endl << endl;
			++i;

			tvecsArray.push_back(tvecs);
			rvecsArray.push_back(rvecs);
			string filename;
		}

		if(waitKey(30)>=0){
			shot = true;
		}
	}

	Mat tvecs = calculaMediaVect(tvecsArray);
	Mat rvecs = calculaMediaVect(rvecsArray);
	tvecsDesvio = calculaDesvioVect(tvecsArray, tvecs, tvecsDesvio);
	rvecsDesvio = calculaDesvioVect(rvecsArray, rvecs, rvecsDesvio);

	//Escreve no arquivo.
	fs.open("extrinsics.xml", FileStorage::WRITE);
	if(!fs.isOpened()){
		cout << "Ocorreu um problema, não foi possível abrir o arquivo" << endl;
		return;
	}
	fs << "tvecs" << tvecs;
	fs << "rvecs" << rvecs;
	fs << "tvecsDesvio" << tvecsDesvio;
	fs << "rvecsDesvio" << rvecsDesvio;
	fs.release();

	destroyAllWindows();
	capture.release();
	cout << "Pressione 'enter' para sair.";
	while (cin.get() != '\n');
}

//Calcula a coordenada do ponto real, usando um ponto na imagem.
Mat calculaPontoReal(Mat rotation, Mat tvecs, Mat intrinsic, int x, int y){

	double data[3] = {x*1.0, y*1.0, 1.0};
	Mat q = Mat(3,1,CV_64FC1, data);
	
	Mat real, C, z;
	double lambda;

	Mat rotation_invertida = rotation.inv(DECOMP_LU);
	Mat intrinsic_invertida = intrinsic.inv(DECOMP_LU);

	C = -(rotation_invertida * tvecs);

	z = (rotation_invertida*intrinsic_invertida)*q;

	lambda = ( -(double)C.at<double>(2,0) )/( (double)z.at<double>(2,0) );

	real = C + (lambda*z);

	return real;
}

//Calcula o tamanho real do objeto
void calculaTamanhoReal(Mat rotation, Mat tvecs, Mat intrinsic, imageClass* classe){
	if(classe->distanciaReal){
		classe->distanciaReal = false;
		Mat Ri = calculaPontoReal(rotation, tvecs, intrinsic, classe->xi, classe->yi);
		Mat Rf = calculaPontoReal(rotation, tvecs, intrinsic, classe->xf, classe->yf);

		double Xi =  (double)Ri.at<double>(0,0);
		double Yi =  (double)Ri.at<double>(1,0);
		double Xf =  (double)Rf.at<double>(0,0);
		double Yf =  (double)Rf.at<double>(1,0);

		double D = (Xi-Xf)*(Xi-Xf) + (Yi-Yf)*(Yi-Yf);	//calcula a distância
        D = sqrt(D);

        cout << "Tamanho real em " << classe->windowsName<< ": " << D <<"m" << endl;
	}
}

void requisito4(){
	imageClass realFrame;
	imageClass undistortedFrame;
	Mat intrinsic;
	Mat rotation;
	Mat rvecs;
	Mat tvecs;
	Mat frame,undistortframe;
	vector<float> distCoeff;

	realFrame.windowsName = "Raw";
	undistortedFrame.windowsName = "Undistorted";

	//Ler do arquivo
	FileStorage fs("intrinsics.xml", FileStorage::READ);
	if(!fs.isOpened()){
		cout << "Não foi possível abrir o arquivo, faça a calibração de intrínsecos primeiro" << endl;
		return;
	}
	fs["intrinsic"] >> intrinsic;
	fs.release();

	fs.open("distortion.xml", FileStorage::READ);
	if(!fs.isOpened()){
		cout << "Não foi possível abrir o arquivo, faça a calibração de intrínsecos primeiro" << endl;
		return;
	}
	fs["distortion"] >> distCoeff;
	fs.release();

	fs.open("extrinsics.xml", FileStorage::READ);
	if(!fs.isOpened()){
		cout << "Não foi possível abrir o arquivo, faça a calibração de extrínsecos primeiro" << endl;
		return;
	}
	fs["tvecs"] >> tvecs;
	fs["rvecs"] >> rvecs;
	fs.release();

	Rodrigues(rvecs, rotation);

	system("clear");
	cout << "RÉGUA VISUAL" << endl << endl;

	VideoCapture capture(0);

	namedWindow("Raw", WINDOW_AUTOSIZE);
	namedWindow("Undistorted", WINDOW_AUTOSIZE);

	while(1){
		capture >> frame;
		realFrame.image = frame.clone();
		undistort(frame, undistortframe, intrinsic, distCoeff);
		undistortedFrame.image = undistortframe;
		desenhaLinha("Raw", &realFrame);
		desenhaLinha("Undistorted", &undistortedFrame);
		calculaTamanhoReal(rotation, tvecs, intrinsic, &realFrame);
		calculaTamanhoReal(rotation, tvecs, intrinsic, &undistortedFrame);

		if(waitKey(30)==27){
			break;
		}
	}
	destroyAllWindows();
}


int main(){
	string choice;

	while(choice != "5"){
		system("clear");
		cout << "***************************************************************************************" << endl;
		cout << "*                                CALIBRAÇÂO DE CÂMERA                                 *" << endl;
		cout << "***************************************************************************************" << endl << endl << endl;

		cout << "1- REQUISITO 1" << endl;
		cout << "Abre a câmera, recebe do mouse as coordenadas de dois cliques, desenha uma linha ligando os dois cliques e calcula a sua distância em pixels." << endl << endl;
		cout << "2- REQUISITO 2" << endl;
		cout << "Realiza a calibração dos parâmetros intrínsecos da câmera e faz o requisito 1 nas saídas de vídeo 'raw' e 'undistorted'." << endl << endl;
		cout << "3- REQUISITO 3" << endl;
		cout << "Realiza a calibração dos parâmetros extrínsecos e retorna a norma do vetor de translação." << endl << endl;
		cout << "4- REQUISITO 4" << endl;
		cout << "Depois de realizar a calibração de intrínsecos e extrinsecos, para uma determinada distância, realiza a medição de um objeto."<< endl<<endl;
		cout << "5- Sair" << endl;

		cout << endl << "Sua escolha:";

		cin >> choice;
		while (cin.get() != '\n');

		if(choice == "1"){
			requisito1();
		}
		if(choice == "2"){
			requisito2();
		}
		if(choice == "3"){
			requisito3();
		}
		if(choice == "4"){
			requisito4();
		}
	}

}