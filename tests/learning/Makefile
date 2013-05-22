
debug:
	cd ../../debug/tests/learning; make 

release:
	cd ../../release/tests/learning; make

%: %.cpp
	cd ../../debug/tests/learning; time make $@
