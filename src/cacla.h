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

#include "geom.h"
#include "approx.h"

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
	ApproxTiny<NS, 1> V;
	ApproxTiny<NS, NA> Ac;
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