#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "edge.cuh"
#include "neuron.cuh"

__global__ void NeuronTimestep(
	int numNeur,
	int numExcit,
	Neuron *d_neuron,
	float *d_I,
	bool *d_cf,
	float *d_driven,
	int t,
	int maxDelay,
	float *d_v,
	float *d_input)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int index = (i * maxDelay) + (t % maxDelay);
	if (i < numNeur)
	{
		d_cf[i] = false;
		d_neuron[i].updateInput(d_driven[i]);
		d_neuron[i].updateInput(d_I[index]);


		d_I[index] = 0;
		d_input[i] = d_neuron[i].I;
		d_neuron[i].timestep();

		d_v[i] = d_neuron[i].returnVoltage();
		if (d_neuron[i].fired)
		{
			d_cf[i] = true;
		}
		d_neuron[i].I = 0;
	}
}
 

__global__ void CommunicationPhase(
	int numEdge,
	bool *d_cf,
	Edge *d_edges,
	float *d_I,
	int t,
	int maxDelay)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numEdge)
	{
		int s = d_edges[i].giveSource();
		if (d_cf[s])
		{
			int tar = d_edges[i].giveTarget();
			int delay = d_edges[i].giveDelay();
			int index = (tar * maxDelay) + ((t + delay) % maxDelay);
			atomicAdd(&d_I[index], d_edges[i].giveWeight());
		}
	}
}


#endif