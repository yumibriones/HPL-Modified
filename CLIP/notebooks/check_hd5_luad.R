#importing relevant libraries
library(rhdf5)
library(uwot)
library(ggplot2)
library(umap)
library(readr)
library(RColorBrewer)

### TRAIN ###

h5_file <- "/gpfs/scratch/yb2612/dl4med_25/dl_project/scratch_data/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train-002.h5"
h5f <- H5Fopen(h5_file)
print(h5f)

labels <- h5read(h5f, "train_labels")

unique_labels <- unique(labels)
print(unique_labels)

label_counts <- table(labels)
print(label_counts)

### VALIDATION ###
h5_file <- "/gpfs/scratch/yb2612/dl4med_25/dl_project/scratch_data/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation-003.h5"
h5f <- H5Fopen(h5_file)
print(h5f)

labels <- h5read(h5f, "valid_labels")

unique_labels <- unique(labels)
print(unique_labels)

label_counts <- table(labels)
print(label_counts)

### TEST ###

h5_file <- "/gpfs/scratch/yb2612/dl4med_25/dl_project/scratch_data/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_test-001.h5"
h5f <- H5Fopen(h5_file)
print(h5f)

labels <- h5read(h5f, "test_labels")

unique_labels <- unique(labels)
print(unique_labels)

label_counts <- table(labels)
print(label_counts)


### TRAIN 250K ###