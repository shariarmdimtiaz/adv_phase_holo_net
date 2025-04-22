## [Advanced deep learning model for direct phase-only hologram generation using complex-valued neural networks](https://doi.org/10.1016/j.neucom.2025.129672)
- [Article Link](https://doi.org/10.1016/j.neucom.2025.129672)


<br>
<p align="center"> <img src="https://github.com/shariarmdimtiaz/adv_phase_holo_net/imgs/proposed-model.jpg" width="80%"> </p>

## Preparation:

### Requirement:
- pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
- pip install opencv-python
- pip install tqdm
- pip install scipy
- pip install scikit-image
- numpy 1.18.5, matplotlib 3.2.0 or higher.
- The code is tested with python=3.8, cuda=11.1.
- A single GPU with cuda memory larger than 16 GB is required.

### Datasets:

- We used the DIV2K dataset for training and evaluation. Please refer to the [website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for details.

### Path structure of Datasets:

```
├──./DIV2K/
│    ├── DIV2K_train_HR
│    │    ├──0000.png
│    │    ├──0001.png
│    │    ├──0002.png
│    │    ├── ...   
│    ├── DIV2K_valid_HR
│    │    ├──0801.png
│    │    ├──0802.png
│    │    ├──0803.png
│    │    ├── ... 
│    ├── Test_data
│    │    ├── ...

```

## Test LFs:

- Place the input LFs into `./DIV2K` (see the examples).
- Run `train.py` for the train the model using DIV2K dataset.
- Run `test.py` to test the model and generate phase only hologram (CGH).
- The result files (i.e., `scene_name.png`) will be saved to `./result/`.
