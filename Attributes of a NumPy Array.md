

NumPy array (ndarray class) is the most used construct of NumPy in Machine Learning and Deep Learning. Let us look into some important attributes of this NumPy array.

Let us create a Numpy array first, say, `array_A`.

Pass the above list to `array()` function of NumPy

```
array_A = np.array([ [3,4,6], [0,8,1] ])
```

Now, let us understand some important attributes of ndarray object using the above-created array `array_A`.

**(1) ndarray.ndim**

`ndim` represents the number of dimensions (axes) of the ndarray.

e.g. for this 2-dimensional array [ [3,4,6], [0,8,1]], value of `ndim` will be 2. This ndarray has two dimensions (axes) - rows (axis=0) and columns (axis=1)

**(2) ndarray.shape**

`shape` is a tuple of integers representing the size of the ndarray in each dimension.

e.g. for this 2-dimensional array [ [3,4,6], [0,8,1]], value of `shape` will be (2,3) because this ndarray has two dimensions - rows and columns - and the number of rows is 2 and the number of columns is 3

**(3) ndarray.size**

`size` is the total number of elements in the ndarray. It is equal to the product of elements of the shape. e.g. for this 2-dimensional array [ [3,4,6], [0,8,1]], `shape` is (2,3), `size` will be product (multiplication) of 2 and 3 i.e. (2*3) = 6. Hence, the size is 6.

**(4) ndarray.dtype**

`dtype` tells the data type of the elements of a NumPy array. In NumPy array, all the elements have the same data type.

e.g. for this NumPy array [ [3,4,6], [0,8,1]], `dtype` will be `int64`

**(5) ndarray.itemsize**

`itemsize` returns the size (in bytes) of each element of a NumPy array.

e.g. for this NumPy array [ [3,4,6], [0,8,1]], `itemsize` will be 8, because this array consists of integers and size of integer (in bytes) is 8 bytes.