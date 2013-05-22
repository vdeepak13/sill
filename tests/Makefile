debug:
	cd ../debug/tests; make 

release:
	cd ../release/tests; make

%: %.cpp
	cd ../debug/tests; time make $@
