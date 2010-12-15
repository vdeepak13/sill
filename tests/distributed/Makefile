
debug:
	cd ../../debug/tests/distributed; make 

release:
	cd ../../release/tests/distributed; make

%: %.cpp
	cd ../../debug/tests/distributed; time make $@
