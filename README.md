A helper module to load/download mnist data

## Installation
```console
git clone https://github.com/bsounak/mnist.git
cd mnist
pip install .
```

### Usage
```python
$python
>>> import mnist
>>> mnist.download()
>>> X_train, y_train, X_test, y_test = mnist.load()
>>> X_train.shape
(60000, 28, 28)
>>> y_train.shape
(60000,)
>>> mnist.preview()
```
![image](https://imgur.com/dfe4Fsp.png)
