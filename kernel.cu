
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <vector>
#include <ctime>
#include <iostream>

#include "kernels.cuh"
#include "edge.cuh"
#include "neuron.cuh"


		 int main()
{
	int numNeurons = 1000;
	int numExcit = 800;
	int T = 2000;
	int equilizationTime = 100;
	int transientTime = 300;
	int maxDelay = 15;

	/* CUDA Parameters */
	int numThreads = 512;

	/* Neurons */

	Neuron *h_neurons, *d_neurons;

	float *h_I, *d_I, *h_driven, *d_driven;
	bool *d_cf, *h_cf;

	float *d_inputs;

	float *d_v;

	h_neurons = new Neuron[numNeurons];
	h_I = new float[numNeurons*maxDelay];
	h_cf = new bool[numNeurons];
	h_driven = new float[numNeurons];

	float *h_drivenZero = new float[numNeurons];
	float *d_drivenZero;

	bool **SpikeTrainYard = new bool*[T];
	float **VoltageTrace = new float *[T];
	float **InputTrace = new float*[T];

	for (int i = 0; i < numNeurons; i++)
	{
		for (int j = 0; j < maxDelay; j++)
		{
			h_I[i*maxDelay + j] = 0;
		}

		if (i < numExcit)
		{
			h_neurons[i].setRegSpike();
		}
		else
		{
			h_neurons[i].setIntBurst();
		}

		h_cf[i] = false;

		if (i < 100)
		{
			h_driven[i] = 100;
			h_drivenZero[i] = 0;
		}
		else
		{
			h_driven[i] = 0;
			h_drivenZero[i] = 0;
		}
	}

	for (int t = 0; t < T; t++)
	{
		SpikeTrainYard[t] = new bool[numNeurons];
		VoltageTrace[t] = new float[numNeurons];
		InputTrace[t] = new float[numNeurons];
	}


	/* Edges */

	std::vector<Edge> h_edges; Edge *d_edges;

	std::mt19937 rd(time(NULL));
	std::uniform_real_distribution<float> dist(0.0, 1.0);
	std::uniform_int_distribution<int> intDist(1, maxDelay);

	int d;
	float w;

	for (int n = 0; n < numNeurons; n++)
	{
		for (int m = 0; m < numNeurons; m++)
		{
			if (n != m)
			{
				if (n < numExcit)
				{
					if (dist(rd) < .15)
					{
						d = intDist(rd);
						w = dist(rd) * 150;
						Edge e(n, m, d, w);
						h_edges.push_back(e);
					}
				}
				else
				{
					if (dist(rd) < .3)
					{
						d = intDist(rd);
						w = dist(rd) * -400;
						Edge e(n, m, d, w);
						h_edges.push_back(e);
					}
				}
			}
		}
	}

	int numEdges = h_edges.size();

	/* CUDA Memory Functions */

	cudaMalloc((void**)&d_neurons, numNeurons * sizeof(Neuron));
	cudaMalloc((void**)&d_I, numNeurons * maxDelay * sizeof(float));
	cudaMalloc((void**)&d_driven, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_cf, numNeurons * sizeof(bool));
	cudaMalloc((void**)&d_v, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_inputs, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_drivenZero, numNeurons * sizeof(float));


	cudaMalloc((void**)&d_edges, numEdges * sizeof(Edge));


	cudaMemcpy(d_neurons, h_neurons, numNeurons * sizeof(Neuron), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, numNeurons * maxDelay * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_driven, h_driven, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_drivenZero, h_drivenZero, numNeurons * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_edges, h_edges.data(), numEdges * sizeof(Edge), cudaMemcpyHostToDevice);


	/* Run Simulation */

	for (int t = 0; t < equilizationTime; t++)
	{
		/* Run Timesteps, No Communication */
		NeuronTimestep << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons,
			numExcit,
			d_neurons,
			d_I,
			d_cf,
			d_drivenZero,
			t,
			maxDelay,
			d_v,
			d_inputs);
	}

	for (int t = 0; t < transientTime; t++)
	{
		/* Run Timesteps, Communication, No Writing */
		NeuronTimestep << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons,
			numExcit,
			d_neurons,
			d_I,
			d_cf,
			d_drivenZero,
			t,
			maxDelay,
			d_v,
			d_inputs);

		CommunicationPhase << <(numEdges + numThreads - 1) / numThreads, numThreads >> >(
			numEdges,
			d_cf,
			d_edges,
			d_I,
			t,
			maxDelay);

	}

	for (int t = 0; t < T; t++)
	{
		/* Run Timesteps, Communication, Write Results*/
		NeuronTimestep << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons,
			numExcit,
			d_neurons,
			d_I,
			d_cf,
			d_driven,
			t,
			maxDelay,
			d_v,
			d_inputs);

		CommunicationPhase << <(numEdges + numThreads - 1) / numThreads, numThreads >> >(
			numEdges,
			d_cf,
			d_edges,
			d_I,
			t,
			maxDelay);

		cudaMemcpy(SpikeTrainYard[t], d_cf, numNeurons * sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(VoltageTrace[t], d_v, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(InputTrace[t], d_inputs, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	}


	/* Analyzing Run */

	std::vector<std::vector<int>> Firings;

	for (int t = 0; t < T; t++)
	{
		//std::cout << VoltageTrace[t][0] << "," << InputTrace[t][0] <<  std::endl;
		for (int n = 0; n < numNeurons; n++)
		{
			if (SpikeTrainYard[t][n] == true)
			{
				std::vector<int> v;
				v.push_back(t);
				v.push_back(n);
				Firings.push_back(v);
			}
		}
	}

	std::cout << "There were " << Firings.size() << " firings." << std::endl;


	/* Clean Up Code */

	cudaDeviceReset();

	for (int t = 0; t < T; t++)
	{
		delete[] SpikeTrainYard[t];
		delete[] VoltageTrace[t];
		delete[] InputTrace[t];
	}

	delete[] h_neurons; delete[] h_I; delete[] h_cf; delete[] SpikeTrainYard; delete[] h_driven;
	delete[] VoltageTrace;
	delete[] InputTrace;
	return 0;
}
