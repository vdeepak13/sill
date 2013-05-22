
debug:
	cd ../../debug/tests/factor; make 

release:
	cd ../../release/tests/factor; make

%: %.cpp
	cd ../../debug/tests/factor; time make $@
