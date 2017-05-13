.PHONY: polygon clean run show

polygon:
	scons

clean:
	scons -c

run: polygon
	./polygon

show:
	dot -Tgif graph_net_example.txt -o graph.gif
	feh graph.gif 