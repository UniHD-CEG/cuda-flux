#/usr/bin/env bash

NV_SMI=$(which nvidia-smi)
if [ ! -x $NV_SMI ]; then
	exit 77
fi

$@
