opencv_createsamples -info positive.info -bg negative.txt -vec new.vec -num 136 -w 30 -h 30
if [ $? -ne 0 ]; then
	exit $?
fi
