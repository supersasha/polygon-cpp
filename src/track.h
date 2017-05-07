#ifndef __POLYGON_TRACK_H
#define __POLYGON_TRACK_H

#include <cmath>
#include <iostream>
#include <random>
#include "geom.h"

std::vector<Pt> clover_data = {
    {-11.0, 1.0},
    {-9.0, 3.0},
    {-7.0, 3.0},
    {-5.0, 1.0},
    {-3.0, 1.0},
    {-1.0, 3.0},
    {-1.0, 5.0},
    {-3.0, 7.0},
    {-3.0, 9.0},
    {-1.0, 11.0},
    {1.0, 11.0},
    {3.0, 9.0},
    {3.0, 7.0},
    {1.0, 5.0},
    {1.0, 3.0},
    {3.0, 1.0},
    {5.0, 1.0},
    {7.0, 3.0},
    {9.0, 3.0},
    {11.0, 1.0},
    {11.0, -1.0},
    {9.0, -3.0},
    {7.0, -3.0},
    {5.0, -1.0},
    {3.0, -1.0},
    {1.0, -3.0},
    {1.0, -5.0},
    {3.0, -7.0},
    {3.0, -9.0},
    {1.0, -11.0},
    {-1.0, -11.0},
    {-3.0, -9.0},
    {-3.0, -7.0},
    {-1.0, -5.0},
    {-1.0, -3.0},
    {-3.0, -1.0},
    {-5.0, -1.0},
    {-7.0, -3.0},
    {-9.0, -3.0},
    {-11.0, -1.0}
};

Pt normalized(const Pt& v) 
{
	return 1.0 / v.norm() * v;
}

Float vec_prod_sign(const Pt& v1, const Pt& v2)
{
	auto v = v1.x*v2.y - v1.y*v2.x;
	if(v < 0)
		return -1.0;
	return 1.0;
}

Figure make_track(const std::vector<Pt>& points0,
	double d, double scale)
{
	auto n = points0.size();
	auto points = std::vector<Pt>(points0.size());
	for(auto i = 0; i < n; i++) {
		points[i] = scale * points0[i];
	}

/*
	std::mt19937 gen;
	std::normal_distribution<Float> nd(0, 0.15 * scale);	
*/

	std::vector<Pt> ps1, ps2;
	for(auto i = 0; i < n; i++) {
		auto x0 = (i > 1) ? points[i-2] : points[n-2+i];
		auto x1 = (i > 0) ? points[i-1] : points[n-1+i];
		auto x2 = points[i];
		auto y1 = normalized(x1 - x0);
		auto y2 = normalized(x1 - x2);
		auto y = normalized(y1 + y2);
		auto s = vec_prod_sign(y1, y);
		auto z1 = x1 + s*d*y;
		auto z2 = x1 - s*d*y;
/*
		z1 = z1 + Pt(nd(gen), nd(gen));
		z2 = z2 + Pt(nd(gen), nd(gen));
*/
		ps1.emplace_back(z1);
		ps2.emplace_back(z2);
	}
	auto f1 = Figure::closed_path(ps1);
	auto f2 = Figure::closed_path(ps2);

	//return Figure::closed_path(points);
	return Figure::compound({f1, f2});
}

Figure clover(Float d, Float scale)
{
	return make_track(clover_data, d, scale);
}

Figure obstacle(const Pt& p, double size)
{
	const auto d = 0.5 * size;
	std::vector<Pt> pts = {
		p + Pt(d, d),
		p + Pt(d, -d),
		p + Pt(-d, -d),
		p + Pt(-d, d)
	};
	Figure::closed_path(pts);
}

// WayPoint

struct WayPoint
{
	int segment = 0;
	double offset = 0.0;
};

// Projection

struct Projection
{
	WayPoint wp;
	double distance;
};

Projection project(const Pt& a,
	const Pt& b, const Pt& p,
	int segment)
{
	const auto d = a - b;
	auto lambda = dot(a - p, d) / dot(d, d);
	if(lambda < 0) {
		lambda = 0;
	} else if(lambda > 1) {
		lambda = 1;
	}
	auto x = a + lambda * (b - a);
	auto dist = (p-x).norm();
	auto length = (b - a).norm();

	Projection prj{{segment, lambda*length}, dist};
	return prj;
}

// Way
struct Way
{
	std::vector<double> segment_len;
	std::vector<Pt> points;
	int count = 0;

	Way() {};

	Way(const std::vector<Pt>& points0, double scale)
	{
		for(const auto& p: points0) {
			points.emplace_back(scale * p);
		}
		for(auto i = 0; i < points0.size(); i++) {
			segment_len.emplace_back((points[i+1] - points[i]).norm());
		}
		segment_len.emplace_back((points.back() - points.front()).norm());
		count = points0.size();
	}

	WayPoint where_is(const Pt& p)
	{
		auto min_pr = Projection{WayPoint(), 1.0e20};
		for(auto i = 0; i < count; i++) {
			const auto& a = points[i];
			const auto& b = points[(i+1 == count) ? 0 : i+1];
			auto pr = project(a, b, p, i);
			if(pr.distance < min_pr.distance) {
				min_pr = pr;
			}
		}
		return min_pr.wp;
	}

	double offset(const WayPoint& old, const WayPoint& nw)
	{
		if(nw.segment == old.segment) {
			return nw.offset - old.offset;
		} else if ((nw.segment - old.segment == 1)
			|| ((old.segment == count - 1) && (nw.segment == 0))) {
			return segment_len[old.segment] - old.offset + nw.offset;
		} else if ((old.segment - nw.segment == 1)
			|| ((nw.segment == count - 1) && (old.segment == 0))) {
			return segment_len[nw.segment] - nw.offset + old.offset;
		} else {
			// Should not be here
			std::cout << "Should not be here!!! FIXME\n";
		}
	}
};

#endif