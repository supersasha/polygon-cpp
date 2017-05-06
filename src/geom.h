#ifndef __POLYGON_GEOM_H
#define __POLYGON_GEOM_H

#include <array>
#include <vector>
#include <cmath>

typedef double Float;

// Point

struct Pt {
	Float x = 0.0;
	Float y = 0.0;

	constexpr Pt() {};
	constexpr Pt(Float ax, Float ay): x(ax), y(ay) {};

	constexpr Float norm() const {
		return std::sqrt(x*x + y*y);
	}

	constexpr Pt lperp() const {
		return Pt(-y, x);
	}

	constexpr Pt rperp() const {
		return Pt(y, -x);
	}
};

constexpr Pt operator+(const Pt& p1, const Pt& p2)
{
	return Pt(p1.x + p2.x, p1.y + p2.y);
}

constexpr Pt operator-(const Pt& p1, const Pt& p2)
{
	return Pt(p1.x - p2.x, p1.y - p2.y);
}

constexpr Pt operator*(Float k, const Pt& p)
{
	return Pt(k*p.x, k*p.y);
}

constexpr Pt operator*(const Pt& p, Float k) 
{
	return k*p;
}

constexpr Float dot(const Pt& p0, const Pt& p1)
{
	return p0.x * p1.x + p0.y * p1.y;
}

constexpr Float vdot(const Pt& p0, const Pt& p1)
{
	return p0.x * p1.y - p0.y * p1.x;
}

std::ostream& operator<<(std::ostream& os, const Pt& p)
{
	return os << "Pt{" << p.x << ", " << p.y << "}";
}

// Matrix

struct Mtx2
{
	std::array<Pt, 2> rows = { {{0, 0}, {0, 0}} };

	constexpr Mtx2() {};
	
	template<typename T>
	constexpr Mtx2(T&& p1, T&& p2)
		:rows({std::forward<T>(p1), std::forward<T>(p2)}) {};
};

constexpr Pt operator*(const Mtx2& m, const Pt& p)
{
	return Pt(m.rows[0].x*p.x + m.rows[0].y*p.y,
			  m.rows[1].x*p.x + m.rows[1].y*p.y);	
}

std::ostream& operator<<(std::ostream& os, const Mtx2& m)
{
	return os << "M{" << m.rows[0] << ", " << m.rows[1] << "}";
}

// Section

struct Sect
{
	Pt p0;
	Pt p1; 

	constexpr Sect() {};

	template <typename T1, typename T2>
	constexpr Sect(T1&& ap0, T2&& ap1)
		:p0(std::forward<T1>(ap0)),
		 p1(std::forward<T2>(ap1)) {}; 
};

std::ostream& operator<<(std::ostream& os, const Sect& s)
{
	return os << "Sect[" << s.p0 << "--" << s.p1 << "]";
}

// Intersection

struct Isx
{
	Pt point;
	Float dist = -1;

	constexpr Isx() {}

	template <typename T>
	constexpr Isx(T&& p, Float d)
		:point(std::forward<T>(p)),
		 dist(d) {}
};

constexpr Isx intersect(const Sect& subj, const Sect& obj, bool is_ray)
{
	Isx isx;
	auto a1 = subj.p1;
	if(!is_ray) {
		a1 = subj.p1 - subj.p0;
	}
	auto a2 = obj.p0 - obj.p1;
	auto b = obj.p0 - subj.p0;
	auto det = vdot(a1, a2);
	if(std::fabs(det) > 1e-8) {
		auto x0 = vdot(b, a2) / det;
		auto x1 = vdot(a1, b) / det;
		if(x0 >= 0.0 && 0.0 <= x1 && x1 <= 1.0) {
			if(is_ray || x0 < 1.0) {
				isx.dist = x0 * a1.norm();
				isx.point = subj.p0 + x0 * a1;
			}
		}
	}
	return isx;
}

std::ostream& operator<<(std::ostream& os, const Isx& isx)
{
	return os << "Isx[" << isx.point << "; " << isx.dist << "]";
}

// Path

struct Path
{
	std::vector<Sect> sects;

	Path() {}

	template <typename T>
	explicit Path(T&& ss)
		:sects(std::forward<T>(ss)) {}
};

std::ostream& operator<<(std::ostream& os, const Path& p)
{
	os << "Path[";
	for (const Sect& s: p.sects) {
		os << s << ",";
	}
	os << "]";
	return os;
}

// Figure

struct Figure
{
	std::vector<Path> paths;

	Figure(){}

	static Figure closed_path(const std::vector<Pt>& points)
	{
		Path path;
		auto n = points.size();
		for(auto i = 0; i < n - 1; i++) {
			auto s = Sect(points[i], points[i+1]);
			path.sects.emplace_back(std::move(s));
		}
		path.sects.emplace_back(Sect(points[n-1], points[0]));
		Figure f;
		f.paths.emplace_back(std::move(path));
		return f;
	}

	static Figure compound(const std::vector<Figure>& figs)
	{
		Figure fig;
		for(const Figure& f: figs) {
			for(const Path& p: f.paths) {
				fig.paths.emplace_back(p);
			}
		}
		return fig;
	}
};

bool intersected(const Figure& subjs, const Figure& objs)
{
	for(const auto& p1: subjs.paths) {
		for(const auto& s: p1.sects) {
			for(const auto& p2: objs.paths) {
				for(const auto& o: p2.sects) {
					auto isx = intersect(s, o, false);
					if(isx.dist >= 0) {
						return true;
					}
				}
			}
		}
	}
	return false;
}

template <typename Rays, typename Isxs>
void intersect(const Rays& rays,
	const Figure& figure,
	Float infinity,
	Isxs& intersections)
{
	auto i = 0;
	for (const auto& r: rays) {
		auto min_isx = Isx(Pt(), 1.0e20);
		for(const auto& p: figure.paths) {
			for(const auto& s: p.sects) {
				auto isx = intersect(r, s, true);
				if(isx.dist >= 0.0 && isx.dist < min_isx.dist) {
					min_isx = std::move(isx);
				}
			}
		}
		if(min_isx.dist < 0.0) { // TODO: check: should be > infinity
			intersections[i].dist = infinity;
		} else {
			intersections[i] = std::move(min_isx);
		}
		i++;
	}
}

template <typename T>
void recalc_rays_a(T& rays,
	const Pt& center, const Pt& course)
{
	auto k = 2.0 * M_PI / rays.size();
	auto i = 0;
	for(auto& r: rays) {
		auto angle = k * i;
		auto s = std::sin(angle);
		auto c = std::cos(angle);
		r = Sect(center, Pt(c*course.x - s*course.y,
							s*course.x + c*course.y));
		i++;
	}
}

#endif