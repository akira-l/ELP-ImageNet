#!/bin/sh
#****************************************************************#
# ScriptName: test_knn.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2021-11-10 23:26
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-11-14 21:20
# Function: 
#***************************************************************#


python test_base.py --b 4096 --nw 32 --k 20 --tau 0.08  

python test_bem.py --b 4096 --nw 32 --k 20 --tau 0.08 


python test_base.py --b 4096 --nw 32 --k 200 --tau 0.08 

python test_bem.py --b 4096 --nw 32 --k 200 --tau 0.08 
