debug:
	cd ../../../debug/tests/learning/discriminative; make 

release:
	cd ../../../release/tests/learning/discriminative; make

%: %.cpp
	cd ../../../debug/tests/learning/discriminative; time make $@
