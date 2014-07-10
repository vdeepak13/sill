mkdir debug
mkdir release

cd release
cmake -D CMAKE_BUILD_TYPE=Release -G "MinGW Makefiles" ../.
cd ..

cd debug
cmake -D CMAKE_BUILD_TYPE=Debug -G "MinGW Makefiles" ../.
cd ..
