debug:
	cd ../../../debug/tests/math/function; make 

release:
	cd ../../../release/tests/math/function; make

%: %.cpp
	cd ../../../debug/tests/math/function; time make $@
