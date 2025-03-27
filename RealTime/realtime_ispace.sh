#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/perm/aswo/lelouedec/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/perm/aswo/lelouedec/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/perm/aswo/lelouedec/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/perm/aswo/lelouedec/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


source /export/home/aswo/jlelouedec/.bashrc 

conda activate CME_toolkit

cd /export/home/aswo/jlelouedec/beacon2science/RealTime/

python RT_b2s.py
