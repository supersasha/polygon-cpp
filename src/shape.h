#ifndef __POLYGON_SHAPE_H
#define __POLYGON_SHAPE_H

#include <SFML/Graphics.hpp>

#include "geom.h"

class PathShape: public sf::Drawable
{
public:
	PathShape(const Path& p, const sf::Color& c = sf::Color::Black)
		: m_path(p), m_color(c)
	{}

	void draw(sf::RenderTarget &target, sf::RenderStates states) const override
	{
		sf::VertexArray vs(sf::LinesStrip);
		for(const auto& s: m_path.sects) {
			sf::Vertex v(sf::Vector2f(s.p0.x, s.p0.y), m_color);
			vs.append(v);
		}
		const auto p = m_path.sects.back().p1;
		vs.append(sf::Vertex(sf::Vector2f(p.x, p.y), m_color));
		target.draw(vs);
	}

private:
	const Path& m_path;
	sf::Color m_color;
};


class FigureShape: public sf::Drawable
{
public:
	FigureShape(const Figure& f, const sf::Color color = sf::Color::Black)
		: m_figure(f), m_color(color)
	{}

	void draw(sf::RenderTarget &target, sf::RenderStates states) const override
	{
		for(const auto& p: m_figure.paths) {
			PathShape(p, m_color).draw(target, states);
		}
	}
private:
	const Figure& m_figure;
	sf::Color m_color;
};

template <std::size_t NRAYS, std::size_t NA>
class CarShape: public sf::Drawable
{
public:
	CarShape(const Car<NRAYS, NA>& car, const sf::Color& color)
		: m_car(car), m_color(color)
	{}

	void draw(sf::RenderTarget& target, sf::RenderStates states) const override
	{
		FigureShape(m_car.path, m_color).draw(target, states);
	}

private:
	const Car<NRAYS, NA>& m_car;
	sf::Color m_color;
};

#endif
