#ifndef __POLYGON_CAR_H
#define __POLYGON_CAR_H

#include <vector>
#include <cmath>
#include <memory>
#include <iostream>

#include "geom.h"

constexpr double powi(double x, int n)
{
	auto res = 1.0;
	for(auto i = 0; i < n; i++) {
		res *= x;
	}
	return res;
}

template <std::size_t NRAYS, std::size_t NA>
struct Car 
{
private:
	Pt center;
	Pt course;

	double base, length, width;

	std::array<Sect, NRAYS> rays;

	std::shared_ptr<Figure> walls;
	std::array<Isx, NRAYS> isxs;
	std::array<Isx, NRAYS> self_isxs;

public:
	double wheels_angle = 0;
	double speed = 0;

	std::array<Float, NRAYS> state;
	std::array<Float, NA> last_action;

	Figure path;

	Car(const Pt& acenter, const Pt& acourse,
		std::shared_ptr<Figure> awalls,
		double alength = 3.0, double awidth = 1.6)
		: center(acenter), course(acourse), walls(awalls),
		  length(alength), width(awidth)
	{
		base = alength;
		last_action.fill(0.0);
		recalc_rays();
		recalc_path();
		calc_self_isxs();
	}

	template <typename A>
	void act(const A& action)
	{
		last_action[0] = action[0];
		last_action[1] = action[1];

		speed = action[0];
		wheels_angle = M_PI / 4.0 * action[1];
		move_or_stop(0.1);
		recalc_state();
	}
	
	double reward() const
	{		
		auto speed_reward = speed;
		if(speed < 0) {
			speed_reward = -speed/2.0;
		}

		auto dist_reward = 0.0;
		for(const auto& s: state) {
			auto k = s*(1 - 0.0099*s);
			if(k < dist_reward)
				dist_reward = k;
		}

		auto wheels_reward = -wheels_angle*wheels_angle;

		auto action_penalty1 = action_penalty(last_action);
		auto action_reward = -action_penalty1*action_penalty1;

		auto speed_penalty = -speed*speed;

		return 10.0*speed_reward
			+ 20.0*dist_reward // TODO: dist_reward seems not obligatory
			+ 5.0*wheels_reward
			+ action_reward
			+ 10.0*speed_penalty;
	}

private:
	void set_pos(const Pt& acenter, const Pt& acourse)
	{
		center = acenter;
		course = acourse;
		recalc_rays();
		recalc_path();
	}

	template <typename A>
	double action_penalty(const A& action) const
	{
		return std::fabs(speed - action[0]);
	}

	void calc_self_isxs()
	{
		intersect(rays, path, -1.0, self_isxs);
	}

	void recalc_rays()
	{
		recalc_rays_a(rays, center, course);
	}

	void recalc_path()
	{
		const auto l = 0.5 * length * course;
		const auto w = 0.5 * course.rperp() * width;
		path = Figure::closed_path(std::vector<Pt>{center + l - w,
									center + l + w,
									center - l + w,
									center - l - w});
	}

	void move_or_stop(double dt)
	{
		const auto stored_center = center;
		const auto stored_course = course;
		mv(dt);
		recalc_path();
		if(intersected(path, *walls)) {
			center = stored_center;
			course = stored_course;
			speed = 0.0;
            // TODO: check, probably we forgot to recalculate back rays
            // (surely, we should restore saved rays, not recalculate them)
            // TODO: the same with path
			recalc_path();
		} else {
			recalc_rays();
			intersect(rays, *walls, -1.0, isxs);
			for(auto i = 0; i < isxs.size(); i ++) {
				if(isxs[i].dist >= 0) {
					isxs[i].dist -= self_isxs[i].dist;
				}
			}
		}
	}

	void mv(double dt)
	{
		if(std::fabs(wheels_angle) < 0.0001) {
			center = center + speed * dt * course;
		} else {
			move_with_turn(dt);
		}
		//std::cout << "center: " << center << "\n";
	}

	void move_with_turn(double dt)
	{
		const auto tn = std::tan(wheels_angle);
		const auto beta = -speed * dt * tn / base;
		const auto pg = wheels_angle > 0 ?
					course.rperp() : course.lperp();
		const auto rot_center = center - 0.5 * base * course
			+ base / std::fabs(tn) * pg;
		const auto s = std::sin(beta);
		const auto c = std::cos(beta);
		const auto m = Mtx2(Pt(c, -s),
							Pt(s,  c));
		center = rot_center + m * (center - rot_center);
		course = m * course;
	}

	void recalc_state()
	{
		std::transform(isxs.cbegin(), isxs.cend(),
			state.begin(),
			[](const auto& isx) {
				return (isx.dist < 10) ? isx.dist : 10;
			}
		);
	}
};

#endif