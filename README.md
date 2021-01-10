# EE5183 FinTech Final Project (Group 4)

## Prerequisites

The following python packages are used in this project:

* Pytorch
* Tqdm
* Numpy
* Pandas
* Matplotlib
* Prophet (for training Prophet)
* Scipy (for training RandomForest)
* XGBoost (for training XGB/XGBRandomForest)
* Torchvision (for tensorboard logging)
* PIL (for tensorboard logging)

## Train

To tune code, please change parameters in `script/train.sh`. Available algorithms included Recurrent Neural Networks (`GNN`, `RNN`, `LSTM`), [Random Forest Regression][Random Forest Regression] (`RandomForest`), [XGBoost Regression][XGBoost Regression] (`XGB`), [XGBoost Random Forest Regression][XGBoost Random Forest Regression] (`XGBRandomForest`), [Facebook Prophet][Facebook Prophet] (`Prophet`). Training code is followed by automatically prediction. For manual prediction, please set the `--plot_pred` flag in `script/train.sh`. Logging is supported by tensorboard.

```bash
cd EE5183-Final-project-G4/
bash script/train.sh
tensorboard --logdir=./log/ # For logging and comparing different settings
# python3 -m tensorboard.main --logdir=./log/
```

## Inference

Currently only Prophet (the best model) is available in inference stage. Pretrained weights are saved in `./checkpoint/`, if the folder is empty, please execute the training procedure with instructions from above section. Please manually change `pred_date` and `save_loc` for selecting which date to be predicted and which location to store inference result. Note that available `pred_date` is from `2020-03-10` to `2020-10-21`.

```bash
python ./inference.py 
```

### Untested

If training under Linux system, can try install [CuML][CuML] for speeding up Random Forest algorithm with GPU.

[Random Forest Regression]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
[XGBoost Regression]: https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbregressor#xgboost.XGBRegressor
[XGBoost Random Forest Regression]: https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbrfregressor#xgboost.XGBRFRegressor
[Facebook Prophet]: https://github.com/facebook/prophet
[CuML]: https://github.com/rapidsai/cuml