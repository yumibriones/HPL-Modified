# HPL-Modified VICReg 
Jennifer Motter

####Overview
We replicated the HPL pipeline using VICReg as an SSL method. TWIST was attempted for integration into the existing pipeline, but the model did not learn and was omitted. 

We replicated the original HPL pipeline using Barlow Twins to obtain pseudo-ground truth labels and embeddings. Ground truth HPL files sourced from the original HPL pipleine did not contain artifacts as denoted by the `_filtered`. 

### Repo Structure

 `files_to_add` - add or replace these files into the original HPL pipleline 

 `notebooks` - code to subsample the data, obtain losses, leiden clustering, UMAPs

 `scripts` - batch scripts to run the HPL-Modified pipeline

## Instructions
1. Clone the original HPL repository to your local machine ([text](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning.git))  

 `git clone https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning.git` 

 Installation to your local machine should take a few seconds. 

2. Once saved to your local machine, in the original HPL pipeline:

    - Replace `run_representationspathology.py` and `run_representationspathology_projection.py` and with the exact same named files in the same location from the `files_to_add` folder. 

    - In the folder `models`, replace the `loss.py` file with the exact same named file in the same location from the `files_to_add` folder. 

    - In the subfolder `selfsupervised` found in the `models` folder, add the `VICReg.py` file from the `files_to_add` folder.

    - In the subfolder `fold_creation` found in the `utilities` folder, replace the `class_folds.ipynb` notbook with the eact same named files in the same location from the `files_to_add`folder.
  
[**Important**]: Naming conventions were not changed because the original HPL pipeline assumes a strict naming convention to successuflly execute it.

3. In a folder of your choosing, download the Whole Slide Image (WSI) tiles found from the original HPL pipline for LUAD and LUSC (https://drive.google.com/drive/folders/18skVh8Vk6zoxG3Se5Vlb7a3EKP2xHXXd)). 
    - Ensure that the files are located in the appropriate directories and follow the exact same naming conventions as described in the Workspace Setup of the README.md file of the original HPL pipeline. 

4. Create a 250K sample from the original training data from the `00_get_h5_subsample.sh` found in the `scripts`. After successful implementation, you can verify the length of the H5 file from the `check_hd5_luad.R` or `check_hd5_luad.ipynb` file. 

5. Run the SSL model with VICReg using `01_run_ssl_VICReg.sh` found in the `scripts` folder. This file can be modified to run other methods, including Barlow Twins. The suffixes _0, _2, _5 were used to train the three variations of VICReg models. To plot the losses of after training, run the `loss.ipynb` file found in the subfolder `notebooks`. 
    - `VICReg.py` adopts the same hyperparameters as those listed in the original HPL pipeline. The VICReg loss function sets the invariance=25.0, variance=25.0, and covariance=1.0 as these performed as these values achieved optimal performance on imaging data in Bardes et al.'s work ([text](https://arxiv.org/pdf/2105.04906))
   
6. To get the tile vector representations for each tile image, run `02_representations.sh` found in the `scripts` folder. 

7. Combine all representation sets into one H5 file using `03_combine_files.sh` found in the `scripts` folder. 

8. Using LUAD and LUSC metadata found in `utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv`, run `04_combine_h5_metadata.sh` from the `scripts` folder. This combines the H5 files with the metadata. If performing cross-validation, please refer to the original HPL pipline README_HPL.md. 

9. To extract the embeddings, run `extract_embeddings_hpl.ipynb` found in the `notebook` folder.

10. Then to perform leiden clustering, run `leiden_clustering.ipynb` found in the `notebook` folder.

11. To obtain UMAPs with leiden clustering, run `umap.ipynb` found in the `notebook` folder.

## Further information 
Detailed information regarding steps 5-8 and original HPL files can be found in the original HPL pipline's README_HPL.md. 

Modify all scripts as necessary with appropriate directories to ensure files run successfully. 











