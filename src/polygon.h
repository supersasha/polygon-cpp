#ifndef __POLYGON_POLYGON_H
#define __POLYGON_POLYGON_H

#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "cacla.h"
#include "track.h"

constexpr Range TRANGE = Range{-1, 1};

Float normalize(const Range& from, Float x, const Range& to)
{
	return to.lo + (x - from.lo) * (to.hi - to.lo) / (from.hi - from.lo);
}

template <std::size_t N>
struct MinMax
{
	std::array<Range, N> ranges;

	template <typename T>
	MinMax(const T& aranges)
	{
		std::copy(aranges.cbegin(), aranges.cbegin() + N, ranges.begin());
	}

	template <typename C1, typename C2>
	void norm(const C1& c1, C2& c2) const
	{
		// TODO: use std::transform
		/*
		const auto& rit = ranges.begin();
		const auto& it1 = c1.cbegin();
		auto& it2 = c2.begin();
		for(; it1 != c1.cend();) {
			*(c2++) = normalize(*(rit++), (*c1++), TRANGE);			
		}
		*/
		std::transform(ranges.cbegin(), ranges.cend(),
			c1.cbegin(), c2.begin(),
			[](const Range& r, Float x)
			{
				return normalize(r, x, TRANGE);
			});
	}
};

template <std::size_t NRAYS, std::size_t NA>
struct World {
	Car<NRAYS, NA> car;
	std::shared_ptr<Figure> walls;
	std::shared_ptr<Way> way;

	World(std::shared_ptr<Figure> awalls, std::shared_ptr<Way> away)
		: walls(awalls), way(away),
		  car({-110, 0}, {0, 1}, awalls)
	{}

	template <typename A>
	void act(const A& action)
	{
		car.act(action);
	}

	double reward() const
	{
		return car.reward();
	}

	std::array<Float, NRAYS>& state() { return car.state; }
	const std::array<Float, NRAYS>& state() const { return car.state; }

	constexpr std::size_t nrays() const noexcept
	{
		return NRAYS;
	}
};

template <std::size_t NRAYS, std::size_t NA>
struct Polygon
{
	std::vector<World<NRAYS, NA>> worlds;
	std::shared_ptr<Figure> walls;

	double last_reward = 0;
	Cacla<NRAYS, NA> learner;
	MinMax<NRAYS> minmax;
	Range reward_range = {-100, 100};
	unsigned stopped_cycles = 0;
	unsigned wander_cycles = 0;
	unsigned epoch = 0;

	std::string ws_dir;
	unsigned current_index = 0;

	Polygon(std::string dir)
		: ws_dir(dir),
			minmax(mk_state_ranges()),
			learner(mk_state_ranges(),
				18,    // hidden 
				0.99,  // gamma
				0.1,   // alpha
				0.001, // beta
				0.1   // sigma
			)
	{
		auto scale = 10.0;
		walls = std::make_shared<Figure>(clover(4.0, scale));
		auto way = std::make_shared<Way>(clover_data, scale);
		auto world = World<NRAYS, NA>(walls, way);
		worlds = std::vector<World<NRAYS, NA>>(10, world);
	}

	// TODO: save, load

	double run(unsigned ncycles)
	{
		std::array<Float, NRAYS> s, new_s;
		auto N = worlds.size();
		auto sum_reward = 0.0;
		for(auto i = 0; i < ncycles; i++) {
			sum_reward += run_once_for_world(0, s, new_s);
			for(auto j = 1; j < N; j++) {
				run_once_for_world(j, s, new_s);
			}
		}
		return sum_reward;
	}

	template <typename T1, typename T2>
	double run_once_for_world(unsigned index, T1& s, T2& new_s)
	{
		auto& world = worlds[index];

		minmax.norm(world.state(), s);
		auto a = learner.get_action(s);
		world.act(a);
		auto r = world.reward();

		minmax.norm(world.state(), new_s);
		for(auto x: new_s) {
			if(x > 0.9 || x < -0.9) {
				std::cout << "new_s[.]: " << x << "\n";
				throw "normalized value out of range";
			}
		}

		learner.step(s, new_s, a, normalize(reward_range, r, TRANGE));
		last_reward = r;
		return r;		
	}

	const World<NRAYS, NA>& current_world() const
	{ 
		return worlds[current_index];
	}

	const World<NRAYS, NA>& get_world(std::size_t index) const
	{
		return worlds[index];
	}

	std::size_t get_worlds_size() const
	{
		return worlds.size();
	}

	// TODO: v_fn, ac_fn	

	static std::array<Range, NRAYS> mk_state_ranges()
	{
		std::array<Range, NRAYS> state_ranges;
		state_ranges.fill({-5, 20});
		return state_ranges; 
	}
};

#endif