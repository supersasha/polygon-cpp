.PHONY: polygon clean run

polygon:
	scons

clean:
	scons -c

run: polygon
	./polygon
