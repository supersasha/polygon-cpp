import os
homedir = os.environ['HOME']

DefaultEnvironment(CC='g++', CCFLAGS='-std=c++14 -O3',
	CPPPATH=homedir + '/devel/lib/tiny-dnn')
VariantDir('build', 'src', duplicate=0)

sources = ['build/main.cpp']
libs = ['doublefann', 'sfml-graphics', 'sfml-window', 'sfml-system']
libpath = '/usr/lib/x86_64-linux-gnu'

Program('polygon', sources, LIBS=libs, LIBPATH=libpath)
