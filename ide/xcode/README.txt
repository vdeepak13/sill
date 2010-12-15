For those who develope on an apple this is really useful. To create an
xcode project simply run the create_xcode_project.sh script.  Then
open the PRL.project as follows:

%> ./create_xcode_project.sh
%> open PRL.xcodeproj

You will need to specify the base directory.  I recommend using
xcode_build. Once you have create a project you will not need to
recreate a project unless the CMake files are changed.  All
compilation can be done from within xcode.

