# IMU-to-segment assignment
The official implementation of [`Learning Sensor Interdependencies for IMU-to-Segment Assignment`](https://ieeexplore.ieee.org/document/9516022) (IEEE Access).

## Dependencies
- Python3
- TensorFlow >= 2.4

This code has been developed with Docker container created from `nvcr.io/nvidia/tensorflow:21.03-tf2-py3`.  
In addition, `mlxtend` is used to evaluate the trained model in our implementation.
```
pip install mlxtend
```

## Training
Download the dataset (e.g., [TotalCapture](https://cvssp.org/data/totalcapture/)).  
The acceleration, gyro, and orientation of the IMUs are extracted and reshaped into (#IMUs, #frames, dimensions).  
Afterwards, set the --dataset argument to the data path.

Train the I2S assignment model.
```
python train.py --log_dir ./logs --dataset ./data
```

## Evaluation
Set the --load_model argument to the corresponding trained model.
```
python test.py --load_model ./logs/model-data-sensors-root-0000/best.weights.hdf5
```

## Citation
```
@ARTICLE{kaichi2021learning,
  author={Kaichi, Tomoya and Maruyama, Tsubasa and Tada, Mitsunori and Saito, Hideo},
  journal={IEEE Access}, 
  title={Learning Sensor Interdependencies for IMU-to-Segment Assignment}, 
  year={2021},
  volume={9},
  number={},
  pages={116440-116452},
  doi={10.1109/ACCESS.2021.3105801}
}
```
