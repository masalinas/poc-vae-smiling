# Description
VAE Smiling Model to be train in a Mac M1 or later using the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)

## Environment
Create environment space. Right now only exist tensorflow for M1, etc architecture GPU up to the version Python 3.11 included:

```
python3.10 -m venv .venv
source .venv/bin/activate
```

## Dependencies
```
pip install --upgrade tensorflow-macos tensorflow-metal
```

## Dataset
To train our VAE we will use the FFHQ Smiling Faces Dataset from this [link](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP). Download the **thumbnails128x128.zip** (1.97 GB) file from **ffhq-dataset** folder to train your model.

## Show Tensorboard
After training execute this command to open TensorBoard
```
tensorboard --logdir logs/vae
```

Open TensorBoard
```
http://localhost:6006
```