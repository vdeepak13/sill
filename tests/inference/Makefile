
debug:
	cd ../../debug/tests/inference; make 

release:
	cd ../../release/tests/inference; make

%: %.cpp
	cd ../../debug/tests/inference; time make $@
