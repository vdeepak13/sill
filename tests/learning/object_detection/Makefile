
debug:
	cd ../../../debug/tests/learning/object_detection; make 

release:
	cd ../../../release/tests/learning/object_detection; make

%: %.cpp
	cd ../../../debug/tests/learning/object_detection; time make $@
