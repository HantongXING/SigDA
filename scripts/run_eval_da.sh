
BASEPATH=$(cd "$(dirname "$0")" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

export CUDA_VISIBLE_DEVICES="0"

config_path="${BASEPATH}/../default_config.yaml"
echo "config path is : ${config_path}"

python -u "${BASEPATH}"/../eval_domain_adaptation.py --config_path="$config_path" > eval_domain_adaptation.log 2>&1 &
