debug:
	cd ../debug/timings; make 

release:
	cd ../release/timings; make

%: %.cpp
	cd ../release/timings; time make $@
