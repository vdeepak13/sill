
debug:
	cd ../../debug/tests/graph; make 

release:
	cd ../../release/tests/graph; make

%: %.cpp
	cd ../../debug/tests/graph; time make $@
