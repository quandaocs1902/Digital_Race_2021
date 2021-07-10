execute_process(COMMAND "/home/quandao/Digital_Race/python_ws/build/python_pkg/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/quandao/Digital_Race/python_ws/build/python_pkg/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
