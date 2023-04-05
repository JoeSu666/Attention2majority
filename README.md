# Attention2majority
This is the official implementation of the Attention2majority code on Camelyon16 dataset. 

Each WSI need to be patchified and saved in .h5 files. Each .h5 file has two datasets: "bag" and "label", where "bag" contains a matrix in Nx224x224x3, and "label" contains the label value of the WSI.

# intelligent sampling
```$ python discriminator.py```

# slide train
```$ python MIL.py```
