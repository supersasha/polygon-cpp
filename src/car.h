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

template <std::size_t NRAYS>
struct Car 
{
	Pt center;
	Pt course;

	double base, length, width;
	double wheels_angle = 0;
	double speed = 0;

	std::array<Sect, NRAYS> rays;

	Figure path;

	std::shared_ptr<Figure> walls;
	std::array<Isx, NRAYS> isxs;
	std::array<Isx, NRAYS> self_isxs;

	Car(const Pt& acenter, const Pt& acourse,
		std::shared_ptr<Figure> awalls,
		double alength = 3.0, double awidth = 1.6)
		: center(acenter), course(acourse), walls(awalls),
		  length(alength), width(awidth)
	{
		base = alength;
		recalc_rays();
		recalc_path();
		calc_self_isxs();
	}

	void set_pos(const Pt& acenter, const Pt& acourse)
	{
		center = acenter;
		course = acourse;
		recalc_rays();
		recalc_path();
	}

	// TODO: unused function?
	/*
	double action_penalty(const std::vector<Float>& action) const
	{
		const auto h = 0.1;
		const auto m = 8;
		const auto c = 5.0;
		const auto a = h / powi(c, m);
		const auto la = std::fabs(action[0]);
		const auto p = a * powi(la, m);
		auto pp = 0.0;
		if(la > 20.0) {
			pp = 1.0;
		}
		return p / (1 + std::fabs(p)) + pp; // TODO: p is always positive?
	}
	*/

	double val_of_action(double a) const
	{
		return a; // TODO: useless function?
	}

	template <typename A>
	double action_penalty3(const A& action) const
	{
		return std::fabs(speed - action[0]);
	}

	template <typename A>
	void act(const A& action)
	{
		speed = val_of_action(action[0]);
		wheels_angle = M_PI / 4.0 * val_of_action(action[1]);
		move_or_stop(0.1);
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
};

#endif