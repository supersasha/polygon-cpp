#ifndef __POLYGON_APPROX_H
#define __POLYGON_APPROX_H

#define CNN_USE_DOUBLE
#define CNN_SINGLE_THREAD

#include <vector>
#include <iostream>
#include <doublefann.h>
#include <tiny_dnn/tiny_dnn.h>

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
		std::copy(x.cbegin(), x.cbegin() + NI, tmp_in.begin());
		auto p = fann_run(net.get(), const_cast<Float*>(tmp_in.data()));
		std::copy(p, p + NO, res.begin());
		return res;
	} 

	template<typename T, typename X>
	void update(const T& target, const X& x)
	{
		std::copy(x.cbegin(), x.cbegin() + NI, tmp_in.begin());
		std::copy(std::cbegin(target), std::cbegin(target) + NO, tmp_out.begin()); 
		fann_train(net.get(), const_cast<Float*>(tmp_in.data()),
			const_cast<Float*>(tmp_out.data()));
	}

	//TODO: save, load, print
};

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;


template <std::size_t ...LS>
class MLPNet
{
	typedef activation::tanh actfun_t;
	std::array<std::size_t, sizeof...(LS)> layer_sizes = {{LS...}};
	std::array<std::shared_ptr<fc>, sizeof...(LS) - 1> layers;
	std::array<std::shared_ptr<actfun_t>, sizeof...(LS) - 2> acs;
	network<graph> m_net;
public:
	MLPNet()
	{
		for(auto i = 0; i < layer_sizes.size() - 1; i++) {
			auto layer = std::make_shared<fc>(layer_sizes[i], layer_sizes[i+1]);
			layers[i] = layer;
			if(i < layer_sizes.size() - 2) {
				auto ac = std::make_shared<actfun_t>();
				acs[i] = ac;
				*layer << *ac;
			}
			if(i > 0) {
				*(acs[i-1]) << *layer;
			}
		}

		construct_graph(m_net, {layers[0]}, {layers[layer_sizes.size() - 2]});
	}

	network<graph>& net()
	{
		return m_net;
	}
};


struct ResidualUnit
{
	input in;
	fc fc1, fc2;
	activation::tanh ac1, out;
	add plus;
	dropout do1, do2;

	ResidualUnit(std::size_t inout_dim,
					std::size_t hidden_dim)
		: in(inout_dim),
		  //intermediate(inout_dim),
		  fc1(inout_dim, hidden_dim),
		  fc2(hidden_dim, inout_dim),
		  plus(2, inout_dim),
		  do1(inout_dim, 0.5),
		  do2(hidden_dim, 0.5)
	{
		fc1 << ac1 << fc2;
		in << fc1;
		(fc2, in) << plus << out;
	}
};

template <std::size_t NI, std::size_t NH, std::size_t NO>
class ResidualNet
{
	fc fc1, fc2;
	activation::tanh
	ac1; 
	ResidualUnit ru1, ru2;
	network<graph> m_net;

public:
	ResidualNet()
	: fc1(NI, NH),
	  ru1(NH, NH),
	  ru2(NH, NH),
	  fc2(NH, NO)
	{
		fc1 << ac1 << ru1.in;
		ru1.out << ru2.in;
		ru2.out << fc2;
		construct_graph(m_net, {&fc1}, {&fc2});
	}

	network<graph>& net()
	{
		return m_net;
	}
};


template<std::size_t NI, std::size_t NO>
struct ApproxTiny
{
	std::array<Range, NI> ranges;
	const std::size_t n_hidden;

	//mutable ResidualNet<NI, 12, NO> arch;
	mutable MLPNet<NI, 18, 10, NO> arch;
	momentum opt;

private:
	mutable std::vector<vec_t> tmp_in;
	mutable std::vector<vec_t> tmp_out;
public:
	template<typename T>
	ApproxTiny(const T& aranges,
		int ahidden, double learning_rate)
		: n_hidden(ahidden),
		  tmp_in(1, vec_t(NI)),
		  tmp_out(1, vec_t(NO))
	{
		std::copy(aranges.begin(), aranges.begin() + NI, ranges.begin());
		
		std::ofstream ofs("graph_net_example.txt");
		graph_visualizer viz(arch.net(), "graph");
		viz.generate(ofs);

		opt.alpha = learning_rate;
		opt.mu = 0.95;
	}

	template<typename X, typename R>
	void call(const X& x, R& res) const
	{
		// TODO: consider optimization
		std::copy(x.cbegin(), x.begin() + NI, tmp_in[0].begin());
		//auto p = fann_run(net.get(), const_cast<Float*>(tmp_in.data()));
		arch.net().set_netphase(net_phase::test);
		//do1.set_context(net_phase::test);
		auto p = arch.net().predict(tmp_in[0]);
		std::copy(p.cbegin(), p.cend(), res.begin());
	}

	template<typename X>
	std::array<Float, NO> call(const X& x) const
	{
		// TODO: consider optimization
		std::array<Float, NO> res;
		std::copy(x.cbegin(), x.cbegin() + NI, tmp_in[0].begin());
		//auto p = fann_run(net.get(), const_cast<Float*>(tmp_in.data()));
		arch.net().set_netphase(net_phase::test);
		//do1.set_context(net_phase::test);
		auto p = arch.net().predict(tmp_in[0]);
		std::copy(p.cbegin(), p.cend(), res.begin());
		return res;
	} 

	template<typename T, typename X>
	void update(const T& target, const X& x)
	{
		std::copy(x.begin(), x.begin() + NI, tmp_in[0].begin());
		std::copy(std::cbegin(target), std::cbegin(target) + NO, tmp_out[0].begin()); 

		arch.net().set_netphase(net_phase::train);

		network<graph>& net = arch.net();
		net.fit<mse>(opt, tmp_in, tmp_out,
						1, //batch
						1 //epochs
					);
	}

	double max_q() const
	{
		auto max_w = 0.0;
		for(auto * l: arch.net()) {
			auto wcs = l->weights();
			for(auto wc: wcs) {
				for(auto w: *wc) {
					if(std::fabs(w) > max_w) {
						max_w = std::fabs(w);
					} 
				}
			}
		}
		return max_w;
	}

	//TODO: save, load, print
};

#endif