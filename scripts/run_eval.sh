BASEPATH=$(cd "$(dirname "$0")" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
export DEVICE_ID=0

if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

config_path="${BASEPATH}/../default_config.yaml"
echo "config path is : ${config_path}"

python "${BASEPATH}"/../eval_m2sfe.py --config_path="$config_path"  > ./eval_m2sfe.log 2>&1 &
