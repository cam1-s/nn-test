#include "nn.hh"

#include <cinttypes>

// Simple example of nn.hh

int main() {
	// network with an input layer with size=2
	nn net(2);
	// hidden layer size=4
	net.add(new conv(4));
	// output layer size=1
	net.add(new conv(1, true));
	net.build();

	// training data, just results of A xor B for the example
	std::vector<std::pair<std::vector<float>,std::vector<float>>> train = {
		{ { 0.f, 0.f }, { 0.f } },
		{ { 0.f, 1.f }, { 1.f } },
		{ { 1.f, 0.f }, { 1.f } },
		{ { 1.f, 1.f }, { 0.f } }
	};

	size_t epoch_count = 10000;

	printf("training %" PRIu64 " epochs.\n", epoch_count);

	for (size_t E = 0; E < epoch_count; ++E) {
		std::random_shuffle(train.begin(), train.end());

		for (auto &i : train)
			net.train(i.first, i.second);
	}

	// Invoking the model for all examples in the training set.

	for (auto &i : train) {
		float res = net.invoke(i.first)[0];
		printf("IN = { %f, %f }  OUT = %f  EXPECTED = %f\n", i.first[0], i.first[1], res, i.second[0]);
	}

	return 0;
}
