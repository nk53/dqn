#!/bin/bash

conda_path=$(which conda)

if [ -z "$conda_path" ];then
    echo "Need Anaconda to continue" 1>&2
    exit 1
fi

py_is_conda=$(which python | grep -i conda)
if [ -z "$py_is_conda" ]; then
    echo "Please make Anaconda's Python executable your default Python"
    exit 2
fi
pip_is_conda=$(which pip | grep -i conda)
if [ -z "$pip_is_conda" ]; then
    echo "Please make Anaconda's pip executable your default pip"
    exit 3
fi

target_env_exists=$(conda env list | grep -i kgpu)
if [ -n "$target_env_exists" ]; then
    echo "Similar target environment to 'kgpu' name exists:"
    echo $target_env_exists
    echo "To remove the environment use \`conda env remove --name env_name\`"
    echo "Aborting."
    exit 4
fi

conda env create kgpu
conda create --name kgpu python=3.6
conda install --name kgpu cudnn blas cudatoolkit

echo To finish up installation, run the following:
echo 'conda activate kgpu'
echo './finish_setup.sh'

