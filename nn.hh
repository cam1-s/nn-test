#pragma once

#define LEARNING_RATE .1f
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <ctime>

inline uint32_t triple32(uint32_t x) {
	x ^= x >> 17;
	x *= 0xed5ad4bbU;
	x ^= x >> 11;
	x *= 0xac4c1b51U;
	x ^= x >> 15;
	x *= 0x31848babU;
	x ^= x >> 14;
	return x;
}

uint32_t rand_state;

inline uint32_t rng() {
	return rand_state = triple32(rand_state);
}

inline float sigmoid(float x) {
	return 1.f / (1.f + expf(-x));
}

// delta sigmoid
inline float dsigmoid(float x) {
	return x * (1.f - x);
}

// random weight between 0 and 1.
inline float init_weight() {
	union { float f; uint32_t u; } u;
	u.u = 0x3f800000 | (rng() >> 9);
	return u.f - 1.f;
}

class layer {
public:
	layer(size_t w) : width(w), values(w), prev(0), next(0) {
	}

	virtual void set_prev(layer *c) {
		prev = c;
	}

	virtual void set_next(layer *c) {
		next = c;
	}

	virtual void set(std::vector<float> const &v) {
		if (v.size() != values.size())
			throw std::runtime_error("layer::set called with vector of mismatching size.");
		values = v;
	}

	virtual void invoke() {
	}
	virtual void diff() {
	}
	virtual void diff_apply() {
	}

	size_t width;
	std::vector<float> values;
	layer *prev, *next;
};

class conv : public layer {
public:
	conv(size_t w, bool out=false) : layer(w), biases(w), diffs(w), output(out) {
	}

	void set_prev(layer *c) {
		layer::set_prev(c);
		
		weights.resize(c->width);
		for (auto &i : weights)
			i.resize(width);
		
		init();
	}

	void init() {
		for (auto &i : biases)
			i = init_weight();

		for (auto &i : weights)
			for (auto &j : i)
				j = init_weight();
	}

	void invoke() {
		for (size_t i = 0; i < width; ++i) {
			float a = biases[i];

			for (size_t j = 0; j < prev->width; ++j) {
				a += prev->values[j] * weights[j][i];
			}

			values[i] = sigmoid(a);
		}
	}

	void diff() {
		if (output) {
			for (size_t i = 0; i < width; ++i) {
				diffs[i] = (next->values[i] - values[i]) * dsigmoid(values[i]);
			}
		}
		else {
			conv *n = dynamic_cast<conv *>(next);

			for (size_t i = 0; i < width; ++i) {
				float d = 0.f;
				for (size_t j = 0; j < n->width; ++j) {
					d += n->diffs[j] * n->weights[i][j];
				}
				diffs[i] = d * dsigmoid(values[i]);
			}
		}
	}

	void diff_apply() {
		for (int i = 0; i < width; ++i) {
			biases[i] += diffs[i] * LEARNING_RATE;
			for (int j = 0; j < prev->width; ++j) {
				weights[j][i] += prev->values[j] * diffs[i] * LEARNING_RATE;
			}
		}
	}

	bool output;
	std::vector<float> diffs;
	std::vector<float> biases;
	std::vector<std::vector<float>> weights;
};

class nn {
public:
	nn(size_t input_width) {
		rand_state = time(0);
		layers.push_back(new layer(input_width));
	}
	~nn() {
		for (auto *i : layers)
			delete i;
	}

	void add(layer *l) {
		layers.push_back(l);
		if (layers.size() > 1) {
			l->set_prev(layers[layers.size() - 2]);
			layers[layers.size() - 2]->set_next(layers[layers.size() - 1]);
		}
	}

	void build() {
		// add "training output" layer
		add(new layer(layers.back()->width));
	}

	std::vector<float> const &invoke(std::vector<float> const &input) {
		layers[0]->set(input);
		for (size_t i = 1; i < layers.size() - 1; ++i) {
			layers[i]->invoke();
		}
		return layers[layers.size() - 2]->values;
	}

	void train(std::vector<float> const &input, std::vector<float> const &output) {
		layers.back()->set(output);
		invoke(input);

		for (size_t i = layers.size() - 2; i >= 1; --i) {
			layers[i]->diff();
		}
		for (size_t i = layers.size() - 2; i >= 1; --i) {
			layers[i]->diff_apply();
		}
	}

	layer &operator[](size_t idx) {
		return *layers[idx];
	}

	std::vector<layer *> layers;
};
