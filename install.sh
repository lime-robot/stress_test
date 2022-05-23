
# Download the train file
TRAIN_FILE=input/train.zip
if test -f "$TRAIN_FILE"; then
    echo "$TRAIN_FILE exists. skip the download ..."
else
    echo "$TRAIN_FILE does not exists. start downloading..."
    mkdir -p input
    wget https://www.dropbox.com/s/mhcuifprogajqid/train.zip -P input/
    wget https://www.dropbox.com/s/wc8gj0dlnohpmp0/test.zip -P input/
fi


# Install Anaconda3
CONDA_DOWNLOAD_URL=https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
if test -d $(readlink -f ~/anaconda3); then
    echo "anaconda3 is already installed. skip the installation ..."
else
    echo "anaconda3 is not installed. start installation..."
    if ! [ -f "Anaconda3-2022.05-Linux-x86_64.sh" ]; then
        wget "$CONDA_DOWNLOAD_URL"
    fi
    bash ./Anaconda3-2022.05-Linux-x86_64.sh -b -p $(readlink -f ~/anaconda3)
    rm ./Anaconda3-2022.05-Linux-x86_64.sh
fi

eval "$(~/anaconda3/bin/conda shell.bash hook)"

conda create -n test -y python==3.8
conda activate test

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt




