#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <time.h>
#include <Windows.h>
using namespace std;

#define LearningRate 10
int Layer[3] = { 15,12,10 };
double neuron[3][15];
double error[3][15];
double weight[3][15][12];
double DWeight[3][15];
int NNAnswer,Expected;//,int bias = 1;

double sigm(double x) {
    double e = exp(1.0);
    return 1 / (1 + pow(e,-x));
}

/*double predict(double x) {
    if (x > 0.8) {
        return 1;
    }
    else {
        return 0;
    }
}*/

void train() {
    cout << "\n\nLayer 2:";
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 15; ++j) neuron[1][i] += neuron[0][j] * weight[0][j][i];
        neuron[1][i] = sigm(neuron[1][i]);
        cout << "\nneuron[" << i << "]= " << neuron[1][i];
    }
    cout << "\nLayer 3:";
    for (int i = 0; i < 10; ++i){
        for (int j = 0; j < 12; ++j) neuron[2][i] += neuron[1][j] * weight[1][j][i];
        neuron[2][i] = sigm(neuron[2][i]);
        cout << "\nneuron[" << i << "]= " << neuron[2][i];
    }
    double a = neuron[2][0];
    NNAnswer = 0;
    for (int i = 0; i < 10; ++i) if (a < neuron[2][i]) {
        a = neuron[2][i];
        NNAnswer = i;
    }
    cout << "\nActual= " << NNAnswer;
    if (NNAnswer != Expected) {
        error[2][NNAnswer] = neuron[2][NNAnswer] - neuron[2][Expected];
        DWeight[2][NNAnswer] = error[2][NNAnswer]*neuron[2][NNAnswer]*(1-neuron[2][NNAnswer]);
        for (int j = 0; j < Layer[0]; ++j) {
            for (int i = 0; i < Layer[1]; ++i) {
                weight[1][i][NNAnswer] = weight[1][i][NNAnswer] - neuron[1][i] * DWeight[2][NNAnswer] * LearningRate;
                error[1][i] = weight[1][i][NNAnswer] * DWeight[2][NNAnswer];
                DWeight[1][i] = error[1][i] * neuron[1][i] * (1 - neuron[1][i]);
                weight[0][j][i] = weight[0][j][i] - neuron[0][j] * DWeight[1][i] * LearningRate;
            }
        }
        train();
    }
}

int main(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(1, 99);
    for (int l = 0; l < 2; ++l) for (int n = 0; n < Layer[l]; ++n) for (int nln = 0; nln < Layer[l + 1]; ++nln) {
        weight[l][n][nln] = dist(gen);
        weight[l][n][nln] = weight[l][n][nln] / 100;
    }
    cout << "Expected= ";
    cin >> Expected;
    cout << "\nLayer 1:\n";
    for (int i = 0; i < 15; ++i) {
        cout << "neuron[" << i << "]= ";
        cin >> neuron[0][i];
    }
    train();
}
/*
NNError= actual-expected
deltaWeight= NNError*sigmoid(x)dx= NNError*sigmoid(x)*(1-sigmoid(x))
weight= weight-output*deltaWeight*learningRate
NError= weight*deltaWeight
nextLayerNeuronValue= sigmoid(neuron1*weight1+...+neuron[n]*weight[n]+bias)
*/