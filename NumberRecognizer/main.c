#define PROJECT "Number Recognizer"
#define AUTHOR  " Victor Martinez "
#define VERSION "  Version NRv0   "

#define numOutputs 10
#define LR 0.1f

#include <iostream> // In/Out
#include <fstream>  // File stream
#include <vector>
#include <cstdint>  // uint8_t

#include <stdlib.h>
#include <time.h>
// Readme
// uint8_t -> Armazena um inteiro de 8bits sem sinal 0-255;

// MACROS
double rand_uniform(){return (double)rand()/(double)RAND_MAX;}

const std::string train_images_file_name = "MNIST/train-images-idx3-ubyte";
const std::string train_labels_file_name = "MNIST/train-labels-idx1-ubyte";
const std::string  test_images_file_name = "MNIST/t10k-images-idx3-ubyte";
const std::string  test_labels_file_name = "MNIST/t10k-labels-idx1-ubyte";

std::vector<std::vector<uint8_t>> loadMNISTImages(const std::string& filename){
	std::ifstream file(filename, std::ios::binary);
	if(!file.is_open()){throw std::runtime_error("Cannot open file: " + filename);}

	// Read the header
	uint32_t magic_number = 0;
	uint32_t num_images = 0;
	uint32_t num_rows = 0;
	uint32_t num_cols = 0;

	file.read(reinterpret_cast<char*>(&magic_number), 4);
	file.read(reinterpret_cast<char*>(&num_images), 4);
	file.read(reinterpret_cast<char*>(&num_rows), 4);
	file.read(reinterpret_cast<char*>(&num_cols), 4);

	// Convert from big-endian to little-endian if needed
	magic_number = __builtin_bswap32(magic_number);
	num_images   = __builtin_bswap32(num_images);
	num_rows     = __builtin_bswap32(num_rows);
	num_cols     = __builtin_bswap32(num_cols);

	// 2051 is the magic number for image files
	if(magic_number != 2051){throw std::runtime_error("Invalid MNIST image file!");}

	// Prepare storage for images
	std::vector<std::vector<uint8_t>> images (num_images, std::vector<uint8_t>(num_rows * num_cols));

	// Read image data
	for(uint32_t i = 0; i <= num_images-1; i++){
		file.read(reinterpret_cast<char*>(images[i].data()), num_rows*num_cols);
	}

	file.close();
	return images;
}

std::vector<uint8_t> loadMNISTLabels(const std::string& filename){
	std::ifstream file(filename, std::ios::binary);
	if(!file.is_open()){throw std::runtime_error("Cannot open file: " + filename);}

	uint32_t magic_number = 0;
	uint32_t num_labels   = 0;
	file.read(reinterpret_cast<char*>(&magic_number), 4);
	file.read(reinterpret_cast<char*>(&num_labels),   4);

	// Ajustar a ordem dos bytes se necessário (converter de big-endian para a ordem do sistema)
	magic_number = __builtin_bswap32(magic_number);
	num_labels   = __builtin_bswap32(num_labels);

	if(magic_number!=2049){throw std::runtime_error("Erro Leitura no Arquivo de Labels - Magic number incorreto." + filename);}

	std::vector<uint8_t> labels(num_labels);
	file.read(reinterpret_cast<char*>(labels.data()), num_labels);


	file.close();
	return labels;
}

int main(void){
	srand(time(NULL));
	// READ DATA ------------------------------------------------------------------------------------
	auto train_images = loadMNISTImages(train_images_file_name);
	auto train_labels = loadMNISTLabels(train_labels_file_name);
	auto  test_images = loadMNISTImages(test_images_file_name);
	auto  test_labels = loadMNISTLabels(test_labels_file_name);
	int   img_size = train_images[0].size();

	std::cout << "|--------------- " << PROJECT << " ---------------|" << std::endl;
	std::cout << "|--------------- " << AUTHOR  << " ---------------|" << std::endl;
	std::cout << "|--------------- " << VERSION << " ---------------|" << std::endl << std::endl;

	std::cout << "Loaded "    << train_images.size() << " training images!" << std::endl;
	std::cout << "Loaded "    << train_labels.size() << " training labels!" << std::endl;
	std::cout << "Loaded "    <<  test_images.size() << " testing images!"  << std::endl;
	std::cout << "Loaded "    <<  test_labels.size() << " testing labels!"  << std::endl;
	std::cout << "Img Size: " <<      img_size       << " pixels (28x28)."  << std::endl << std::endl;

	// User Input --------------------------------------------------------------------------------
    std::cout << "Defina Num de Layers/Neuronios (L N1 N2 N3): ";

    int numHiddenLayers;
    std::cin >> numHiddenLayers;
    std::vector<int> numNeuronios(numHiddenLayers);
    for(int i=0; i <=numHiddenLayers-1; i++){std::cin >> numNeuronios[i];}

    // Algebra Linear --------------------------------------------------------------------------
	// Layers
	std::vector<double> outputLayer[numOutputs];
	std::vector<std::vector<double>> hiddenLayers(numHiddenLayers);
	for(int i=0; i <=numHiddenLayers-1; i++){
		hiddenLayers[i].resize(numNeuronios[i]);
	}

	// Bias
	std::vector<double> outputBias[numOutputs];
	std::vector<std::vector<double>> hiddenBias(numHiddenLayers);
	for(int i=0; i <=numHiddenLayers-1; i++){
		hiddenBias[i].resize(numNeuronios[i]);
	}

	std::vector<std::vector<double>> hiddenBias(numHiddenLayers);
	std::vector<std::vector<double>> hiddenWeights(numHiddenLayers);
	for(int i=0; i <=numHiddenLayers-1; i++){
		hiddenLayers[i].resize(numNeuronios[i]);
		hiddenBias[i].resize(numNeuronios[i]);
		hiddenWeights[i].resize(numNeuronios[i]);
	} std::vector<double> outputWeights[numOutputs];

	
		
	

	// Inicializar Pesos e Bias -------------------------------------------------------------------
	for(int i=0; i<=hiddenLayers.size()-1; i++){
		for(int j=0; j<=hiddenLayers[i].size()-1; j++){
			hiddenBias[i][j]    = rand_uniform();
			hiddenWeights[i][j] = rand_uniform();
		}
	}



	// Example: Print the first image as ASCII art

	printf("%d", train_labels[150]);
	for (size_t i = 0; i < 28; ++i){
		for (size_t j = 0; j < 28; ++j){
			uint8_t pixel = train_images[150][i * 28 + j];
			std::cout << (pixel > 128 ? '#' : ' ');
		}
		std::cout << std::endl;
	}


	return 0;
}
