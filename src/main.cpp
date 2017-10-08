#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <tuple>

#include <SFML/Graphics.hpp>

#include "geom.h"
#include "track.h"
#include "car.h"
#include "cacla.h"
#include "polygon.h"
#include "shape.h"

template <std::size_t NRAYS, std::size_t NA>
void runPolygon(Polygon<NRAYS, NA>& polygon)
{
	sf::Font font;
	font.loadFromFile("./sansation.ttf");

    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    sf::RenderWindow window(sf::VideoMode(1820, 1080, 32),
    	"Polygon (C++)",
    	sf::Style::Default, settings);

    const auto& world = polygon.get_world(0);

    FigureShape walls(*(polygon.walls));
    std::vector<CarShape<NRAYS, NA>> cars;
    for(auto i = 1; i < polygon.worlds.size(); i++) {
    	cars.emplace_back(
    		CarShape<NRAYS, NA>(polygon.get_world(i).car, sf::Color::Blue)
    	);
    }
    cars.emplace_back(CarShape<NRAYS, NA>(world.car, sf::Color::Red));

    sf::View gView(sf::Vector2f(0,0), sf::Vector2f(400, -400.0 * 1080 / 1820));
    sf::View tView = window.getDefaultView();

    auto n = 0;

    // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // clear the window with black color
        window.clear(sf::Color::White);        

        //view.zoom(1.5);
        //view.move(sf::Vector2f(0,0.5));

        window.setView(gView);

        // draw everything here...
        window.draw(walls);
        for(const auto& car: cars) {
        	window.draw(car);
    	}

        window.setView(tView);
        /*
            let text = format!("Cycles: {}\nSpeed:  {}\nWheels: {}\nAct[0]: {}\n\
                                Act[1]: {}\nReward: {}\nX: {}\nY: {}\n\
                                Offset: {}\nSigma: {}",
                        all_cycles, car.speed, car.wheels_angle,
                        world.last_action[0], world.last_action[1],
                        pg.last_reward, car.center.x, car.center.y,
                        10.0 * world.way.offset(&world.old_way_point, &world.way_point),
                        sigma.deref());

            let mut txt = Text::new().unwrap();
            txt.set_font(&font);
            txt.set_character_size(24);
            txt.set_string(&text);
            txt.set_position2f(1200.0, 30.0);
            txt.set_color(&Color::black());
            window.draw(&txt);
		*/
		//std::string text;
		std::stringstream sstr;

		sstr << "Cycles: " << n << "\n"
			 << "Speed:  " << world.car.speed << "\n"
			 << "Wheels: " << world.car.wheels_angle << "\n"
			 << "Act[0]: " << world.car.last_action[0] << "\n"
			 << "Act[1]: " << world.car.last_action[1] << "\n"
             << "MaxW(V): " << polygon.learner.V.max_q() << "\n"
             << "MaxW(Ac): " << polygon.learner.Ac.max_q() << "\n";
             ;
/*
                                Reward: {}\nX: {}\nY: {}\n\
                                Offset: {}\nSigma: {}",
                        all_cycles, car.speed, car.wheels_angle,
                        world.last_action[0], world.last_action[1],
                        pg.last_reward, car.center.x, car.center.y,
                        10.0 * world.way.offset(&world.old_way_point, &world.way_point),
                        sigma.deref());
*/

		auto text = sf::Text();
		text.setFont(font);
		text.setCharacterSize(24);
		text.setString(sstr.str());
		text.setPosition(1200, 30);
		text.setColor(sf::Color::Black);
		window.draw(text);

        // end the current frame
        window.display();

        auto nn = 100;
        std::cout << (n += nn) << ": " << polygon.run(nn) / nn << std::endl;
    }
}

int main()
{	
	std::cout << Pt(3, 4).norm() << "\n";
	std::cout << Pt(1, 2) - Pt(3, 4) << "\n";
	std::cout << 3 * Pt(3, 4) << "\n";

	std::cout << Mtx2(Pt(1, 2), Pt(3, 4)) << "\n";
	std::cout << Sect(Pt(1, 2), Pt(3, 4)) << "\n";

	std::cout << Isx(Pt(1, 2), 3) << "\n";
	std::cout << Path(std::vector<Sect>(5, Sect())) << "\n";

	Polygon<36, 2> polygon("123");
	runPolygon(polygon);
}
