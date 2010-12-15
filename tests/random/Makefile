debug:
	cd ../../debug/tests/random; make 

release:
	cd ../../release/tests/random; make

%: %.cpp
	cd ../../debug/tests/random; time make $@
