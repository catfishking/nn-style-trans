#!/bin/bash
echo "select gpu or unset(s/u)"

read action

if [ $action == 's' ]; then
	echo "which gpu?(0,1,2,3)"
	read n
	export CUDA_VISIBLE_DEVICES="$n"
else
	unset CUDA_VISIBLE_DEVICES
fi
