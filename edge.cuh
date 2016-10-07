#ifndef __EDGE_CUH__
#define __EDGE_CUH__


class Edge
{
private:
	int source;
	int target;
	int delay;
	float weight;
public:
	__host__ Edge(
		int s,
		int t,
		int d,
		float w);
	__device__ int giveSource();
	__device__ int giveTarget();
	__device__ int giveDelay();
	__device__ float giveWeight();
};
__host__ Edge::Edge(
	int s,
	int t,
	int d,
	float w)
{
	source = s;
	target = t;
	delay = d;
	weight = w;
}

__device__ int Edge::giveSource()
{
	return source;
}

__device__ int Edge::giveTarget()
{
	return target;
}

__device__ int Edge::giveDelay()
{
	return delay;
}

__device__ float Edge::giveWeight()
{
	return weight;
}

#endif