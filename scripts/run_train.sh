
BASEPATH=$(cd "$(dirname "$0")" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit

export CUDA_VISIBLE_DEVICES="0"

config_path="${BASEPATH}/../default_config.yaml"
echo "config path is : ${config_path}"

python -u "${BASEPATH}"/../train_m2sfe.py --config_path="$config_path" > train.log 2>&1 &
