debug:
	cd ../../debug/tests/model; make 

release:
	cd ../../release/tests/model; make

%: %.cpp
	cd ../../debug/tests/model; time make $@
