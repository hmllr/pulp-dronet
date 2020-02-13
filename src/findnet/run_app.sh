rm -rf BUILD
cd ~/gap_sdk

source sourceme.sh

cd -

make clean all run CORE=8