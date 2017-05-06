polygon: geom.h track.h car.h cacla.h polygon.h shape.h main.cpp
	g++ -std=c++14 -O3 -L/usr/lib/x86_64-linux-gnu/ -o polygon main.cpp \
	-ldoublefann -lsfml-graphics -lsfml-window -lsfml-system

run: polygon
	./polygon
