###### !/usr/bin/env sh

CWD=`pwd` 
HDFS=hdfs://hobot-bigdata-ucloud/user/cheng03.wang
#set this to enable reading from hdfs
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms512m -Xmx10000m"

cd ${WORKING_PATH}
mkdir /job_data/oln
cp -r ${WORKING_PATH}/* /job_data/oln

CONDA_ENV_NAME=oln

####get conda env panoptic v1
echo "get conda env oln"
hdfs dfs -get ${HDFS}/envs/${CONDA_ENV_NAME}.tar.gz && tar zxf ${CONDA_ENV_NAME}.tar.gz
# sed -i '1c #!'${WORKING_PATH}/${CONDA_ENV_NAME}'/bin/python' ${WORKING_PATH}/${CONDA_ENV_NAME}/bin/pip
export PATH=${WORKING_PATH}/${CONDA_ENV_NAME}/bin:$PATH
# echo pip
# pip install numpy==1.19
# pip install /cluster_home/data/waymo_open_dataset_tf_2_0_0-1.2.0-cp37-cp37m-manylinux2010_x86_64.whl
# waymo-open-dataset-tf-2-0-0==1.2.0

# download pretrained weights
hdfs dfs -get ${HDFS}/pretrained/resnet50-19c8e357.pth resnet50.pth

CONFIG=${WORKING_PATH}/configs/oln_box/oln_box_cluster.py
GPUS=4
PORT=${PORT:-29500}

# 放到/job_data下的都会在输出网址下的output文件夹下找到
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ${WORKING_PATH}/tools/train.py $CONFIG --launcher pytorch ${@:3} --no-validate \
    --work-dir /job_data/work_dirs

python tools/test.py configs/oln_box/oln_box_local.py /job_data/work_dirs/latest.pth -eval bbox