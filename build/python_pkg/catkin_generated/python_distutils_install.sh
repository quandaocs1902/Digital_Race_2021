#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/quandao/Digital_Race/python_ws/src/python_pkg"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/quandao/Digital_Race/python_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/quandao/Digital_Race/python_ws/install/lib/python3/dist-packages:/home/quandao/Digital_Race/python_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/quandao/Digital_Race/python_ws/build" \
    "/usr/bin/python3" \
    "/home/quandao/Digital_Race/python_ws/src/python_pkg/setup.py" \
     \
    build --build-base "/home/quandao/Digital_Race/python_ws/build/python_pkg" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/quandao/Digital_Race/python_ws/install" --install-scripts="/home/quandao/Digital_Race/python_ws/install/bin"
