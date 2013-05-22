
debug:
	cd ../../../debug/tests/learning/order_based_search; make 

release:
	cd ../../../release/tests/learning/order_based_search; make

%: %.cpp
	cd ../../../debug/tests/learning/order_based_search; time make $@
