#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Simple N.N to learn xor
#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

double sigmoid(double x){return 1/(1+ exp(-x));}
double dSigmoid(double x){return x*(1-x);}
double init_weights(){return (double)rand()/(double)RAND_MAX;}
void shuffle(int *array, size_t n){
    if(n>1){
        for(size_t i=0; i<n-1; i++){
            size_t j = i + ( rand() / (1+(RAND_MAX/(n-i))) );
            int aux = array[j];
            array[j] = array[i];
            array[i] = aux;
        }
    }
}

int main(void){
	srand(time(NULL)); 
	const double lr = 0.1f; // Learning Rate

	double hiddenLayer[numHiddenNodes];
	double outputLayer[numOutputs];

	double hiddenLayerBias[numHiddenNodes];
	double outputLayerBias[numOutputs];

	double hiddenWeights[numInputs][numHiddenNodes];
	double outputWeights[numHiddenNodes][numOutputs];

	double training_inputs[numTrainingSets][numInputs]   = {{0.0f,0.0f}, {1.0f,0.0f}, {0.0f,1.0f}, {1.0f,1.0f}};
	double training_outputs[numTrainingSets][numOutputs] = {  {0.0f},       {1.0f},      {1.0f},      {0.0f}  };

	// Inicializar Pesos
	for(int i=0; i<=numInputs-1; i++){
		for(int j=0; j<numHiddenNodes; j++){
			hiddenWeights[i][j] = init_weights();
		}
	}
	for(int i=0; i<=numHiddenNodes-1; i++){
		for(int j=0; j<numOutputs; j++){
			outputWeights[i][j] = init_weights();
		}
	}
	for(int i=0; i<=numOutputs-1; i++){
		outputLayerBias[i] = init_weights();
	}	

	// Training Set
	int trainingSetOrder[] = {0,1,2,3};
	int numberOfEpochs = 100000;

	// Train the N.N. for a num of epochs
	for(int epochs=0; epochs <= numberOfEpochs-1; epochs++){
		shuffle(trainingSetOrder, numTrainingSets);

		for(int x=0; x<=numTrainingSets-1; x++){
			int i = trainingSetOrder[x];

			// Forward pass
				// Compute hidden layer activation
				for(int j=0; j<=numHiddenNodes-1; j++){
					double activation = hiddenLayerBias[j];
					for(int k=0; k<=numInputs-1; k++){
						activation += training_inputs[i][k] * hiddenWeights[k][j];
					}
					hiddenLayer[j] = sigmoid(activation);
				}
				// Compute output layer activation
				for(int j=0; j<=numOutputs-1; j++){
					double activation = outputLayerBias[j];
					for(int k=0; k<=numHiddenNodes-1; k++){
						activation += hiddenLayer[k] * outputWeights[k][j];
					}
					outputLayer[j] = sigmoid(activation);
				}

			if(epochs%(numberOfEpochs/10)==0){
				printf ("Input:%g %g  Output:%g    Expected Output: %g\n",
                    training_inputs[i][0], training_inputs[i][1],
                    outputLayer[0], training_outputs[i][0]);	
			}	
			

			// Backpropagation			
				// Change in Output weights
				double deltaOutput[numOutputs];
	            for (int j=0; j<numOutputs; j++) {
	                double errorOutput = (training_outputs[i][j] - outputLayer[j]);
	                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
	            }

				// Compute Change in hidden weights
				double deltaHidden[numHiddenNodes];
	            for (int j=0; j<numHiddenNodes; j++) {
	                double errorHidden = 0.0f;
	                for(int k=0; k<numOutputs; k++) {
	                    errorHidden += deltaOutput[k] * outputWeights[j][k];
	                }
	                deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
	            }

				// Apply changes in output weights
				for(int j=0; j<=numOutputs-1; j++){
					outputLayerBias[j] += deltaOutput[j] * lr;
					for(int k=0; k<=numHiddenNodes-1; k++){
						outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
					}
				}
				// Apply changes in hidden weights
				for(int j=0; j<=numHiddenNodes-1; j++){
					hiddenLayerBias[j] += deltaHidden[j] * lr;
					for(int k=0; k<=numInputs-1; k++){
						hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
					}
				}

		} if(epochs%(numberOfEpochs/10)==0){printf("\n");}

	}

	// Print final weights after training
    printf("Final Hidden Weights\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf("[ ");
        for(int k=0; k<numInputs; k++) {
            printf ("%f ", hiddenWeights[k][j]);
        }
        printf("] ");
    }
    
    printf("]\nFinal Hidden Biases\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf ("%f ", hiddenLayerBias[j]);
    }

    printf("]\nFinal Output Weights");
    for (int j=0; j<numOutputs; j++) {
        printf("[ ");
        for (int k=0; k<numHiddenNodes; k++) {
            printf ("%f ", outputWeights[k][j]);
        }
        printf("]\n");
    }

    printf("Final Output Biases\n[ ");
    for (int j=0; j<numOutputs; j++) {
        printf ("%f ", outputLayerBias[j]);
        
    }
    
    printf("]\n");

	return 0;
}