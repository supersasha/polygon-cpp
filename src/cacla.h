#ifndef __POLYGON_CACLA_H
#define __POLYGON_CACLA_H

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <iostream>

#include <doublefann.h>
#include "geom.h"

struct Range
{
	Float lo = 0;
	Float hi = 0;

	constexpr Range() {}
	constexpr Range(Float alo, Float ahi)
		:lo(alo), hi(ahi) {}
};

template<std::size_t NI, std::size_t NO>
struct Approx
{
	std::unique_ptr<fann, decltype(&fann_destroy)> net;
	std::array<Range, NI> ranges;

	const std::size_t n_hidden;

private:
	mutable std::array<Float, NI> tmp_in;
	mutable std::array<Float, NO> tmp_out;

public:
	template<typename T>
	Approx(const T& aranges,
		int ahidden, double learning_rate)
		: n_hidden(ahidden),
		  net(nullptr, fann_destroy)
	{
		std::copy(aranges.begin(), aranges.begin() + NI, ranges.begin());
		auto _net = std::unique_ptr<fann, decltype(&fann_destroy)>(
			fann_create_standard(4, NI, n_hidden, 10, NO)
			, fann_destroy
		);
		net = std::move(_net);
		fann_set_activation_function_hidden(net.get(), FANN_SIGMOID_SYMMETRIC);
		fann_set_activation_function_output(net.get(), FANN_LINEAR);
		fann_set_training_algorithm(net.get(), FANN_TRAIN_INCREMENTAL);
		fann_set_learning_rate(net.get(), learning_rate);
		fann_set_learning_momentum(net.get(), 0.0);
	}

	template<typename X, typename R>
	void call(const X& x, R& res) const
	{
		std::copy(x.cbegin(), x.begin() + NI, tmp_in.begin());
		auto p = fann_run(net.get(), const_cast<Float*>(tmp_in.data()));
		std::copy(p, p + NO, res.begin());
	}

	template<typename X>
	std::array<Float, NO> call(const X& x) const
	{
		std::array<Float, NO> res;
		std::copy(x.cbegin(), x.begin() + NI, tmp_in.begin());
		auto p = fann_run(net.get(), const_cast<Float*>(tmp_in.data()));
		std::copy(p, p + NO, res.begin());
		return res;
	} 

	template<typename T, typename X>
	void update(const T& target, const X& x)
	{
		std::copy(x.begin(), x.begin() + NI, tmp_in.begin());
		std::copy(std::cbegin(target), std::cbegin(target) + NO, tmp_out.begin()); 
		fann_train(net.get(), const_cast<Float*>(tmp_in.data()),
			const_cast<Float*>(tmp_out.data()));
	}

	//TODO: save, load, print
};

template <std::size_t NA>
struct CaclaState
{
	mutable std::array<Float, NA> action;
	double alpha;
	double beta;
	double gamma;
	mutable double sigma;
	double var;
};

template <std::size_t NS, std::size_t NA>
struct Cacla
{
	Approx<NS, 1> V;
	Approx<NS, NA> Ac;
	CaclaState<NA> state;
	std::mt19937 gen;

	template <typename T>
	Cacla(const T& state_ranges,
		int hidden,
		double gamma, double alpha, double beta, double sigma)
		: V(state_ranges, hidden, alpha),
		  Ac(state_ranges, hidden, alpha),
		  gen(time(0)),
		  state{ std::array<Float, NA>(),
			alpha, beta, gamma, sigma, 1.0 /*=var*/}
	{}

	template <typename T>
	std::array<Float, NA> get_action(const T& st)
	{
		const auto mu = Ac.call(st);
		for(auto i = 0; i < mu.size(); i++) {
			std::normal_distribution<Float> nd(mu[i], state.sigma);
			state.action[i] = nd(gen);	
		}
		/*
		if state.sigma > 0.1 {
			state.sigma *= 0.99999993068528434627048314517621;
		}
		*/
		return state.action;
	}

	template <typename OST, typename NST, typename A>
	void step(const OST& old_state,
			const NST& new_state,
			const A& action,
			double reward)
	{
		auto old_state_v = V.call(old_state);
		auto new_state_v = V.call(new_state);
		auto target = std::array<Float, 1>{{reward + state.gamma * new_state_v[0]}}; // ??? why ...[0] ???
		auto td_error = target[0] - old_state_v[0];
		V.update(target, old_state);
		if(td_error > 0) {
			state.var = (1 - state.beta) * state.var
				+ state.beta * td_error * td_error;
			auto n = std::ceil(td_error / std::sqrt(state.var));
			if(n > 12) {
				std::cout << "n = " << n << "\n";
			}
			for(auto i = 0; i < n; i++) {
				Ac.update(action, old_state);
			}
		}
	}

	// TODO: save, load, print, v_fn, ac_fn
};

#endif