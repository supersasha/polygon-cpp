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

template<std::size_t NI, std::size_t NO>
struct ApproxTiny
{
	std::array<Range, NI> ranges;
	const std::size_t n_hidden;

	mutable network<graph> net;
	gradient_descent opt;

	fc fc1, fc2, fc3;
	activation::tanh tanh1, tanh2;


private:
	//mutable std::array<Float, NI> tmp_in;
	//mutable std::array<Float, NO> tmp_out;

	mutable std::vector<vec_t> tmp_in;
	mutable std::vector<vec_t> tmp_out;
public:
	template<typename T>
	ApproxTiny(const T& aranges,
		int ahidden, double learning_rate)
		: n_hidden(ahidden),
			tmp_in(1, vec_t(NI)),
			tmp_out(1, vec_t(NO)),

			fc1(NI, n_hidden),
			fc2(n_hidden, 10),
			fc3(10, NO)
	{
		std::copy(aranges.begin(), aranges.begin() + NI, ranges.begin());

		/*
		auto fc1 = fc(NI, n_hidden);
		auto tanh1 = activation::tanh();
		auto fc2 = fc(n_hidden, 10);
		auto tanh2 = activation::tanh();
		auto fc3 = fc(10, NO);
		*/

		fc1 << tanh1 << fc2 << tanh2 << fc3;
		std::cout << "1\n";
		construct_graph(net, {&fc1}, {&fc3});
		std::cout << "2\n";

/*		
		net << fc(NI, n_hidden) << activation::tanh()
			<< fc(n_hidden, 10) << activation::tanh()
			<< fc(10, NO);
*/
		opt.alpha = learning_rate;
		//opt.lambda = 0.0001;
	}

	template<typename X, typename R>
	void call(const X& x, R& res) const
	{
		// TODO: consider optimization
		std::copy(x.cbegin(), x.begin() + NI, tmp_in[0].begin());
		//auto p = fann_run(net.get(), const_cast<Float*>(tmp_in.data()));
		auto p = net.predict(tmp_in[0]);
		std::copy(p.cbegin(), p.cend(), res.begin());
	}

	template<typename X>
	std::array<Float, NO> call(const X& x) const
	{
		// TODO: consider optimization
		std::array<Float, NO> res;
		std::copy(x.cbegin(), x.cbegin() + NI, tmp_in[0].begin());
		//auto p = fann_run(net.get(), const_cast<Float*>(tmp_in.data()));
		auto p = net.predict(tmp_in[0]);
		std::copy(p.cbegin(), p.cend(), res.begin());
		return res;
	} 

	template<typename T, typename X>
	void update(const T& target, const X& x)
	{
		std::copy(x.begin(), x.begin() + NI, tmp_in[0].begin());
		std::copy(std::cbegin(target), std::cbegin(target) + NO, tmp_out[0].begin()); 

		net.fit<mse>(opt, tmp_in, tmp_out,
						1, //batch
						1 //epochs
					);
	}

	double max_q() const
	{
		auto max_w = 0.0;
		for(auto * l: net) {
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