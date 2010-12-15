debug:
	cd ../../debug/tests/math; make 

release:
	cd ../../release/tests/math; make

%: %.cpp
	cd ../../debug/tests/math; time make $@
