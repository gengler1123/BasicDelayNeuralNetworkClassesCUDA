#ifndef __NEURON_CUH__
#define __NEURON_CUH__


class Neuron
{
private:
	float v;
	float u;

	float k;
	float v_r, v_t, v_peak, C;
	float a, b, c, d;
	float dv, du;

public:
	bool fired;
	float I;
	__host__ Neuron();
	__host__ void setRegSpike();
	__host__ void setIntBurst();
	__device__ void timestep();
	__device__ void updateInput(float input); // Need
	__device__ bool returnFired();  // Need
	__device__ void resetInput();
	__device__ float returnVoltage();
};


__device__ float Neuron::returnVoltage()
{
	return v;
}


__host__ Neuron::Neuron()
{

}


__device__ void Neuron::updateInput(float input)
{
	I += input;
}


__device__ void Neuron::resetInput()
{
	I = 0;
}


__device__ bool Neuron::returnFired()
{
	return fired;
}


__host__ void Neuron::setRegSpike()
{
	k = 0.7f;
	v_r = -60.0f;
	v_t = -40.0f;
	a = 0.03f;
	b = -2.0f;
	c = -50.0f;
	d = 100.0f;
	C = 100.0f;
	v_peak = 35.0f;
	v = -60.0f;
	u = 100.0f;
	fired = false;
	I = 0;
}

__host__ void Neuron::setIntBurst()
{
	k = 1.2f;
	v_r = -75.0f;
	v_t = -45.0f;
	a = 0.01f;
	b = 5.0f;
	c = -56.0f;
	d = 130.0f;
	C = 150.0f;
	v_peak = 50.0f;
	v = -75.0f;
	u = 0.0f;
	fired = false;
	I = 0;
}

__device__ void Neuron::timestep()
{
	fired = false;
	for (int dt = 0; dt < 4; dt++)
	{
		dv = (k * (v - v_r)*(v - v_t) - u + I) / C;
		du = (a * (b * (v - v_r) - u));
		v += 0.25*dv;
		u += 0.25*du;

		if (v > v_peak)
		{
			fired = true;
			v = c;
			u += d;
			break;
		}
		
	}
	I = 0;
}

#endif