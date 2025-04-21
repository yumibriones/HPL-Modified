# HPL-CLIP tutorial

How to run HPL-CLIP. This is an integration of https://github.com/mlfoundations/open_clip and https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning.

## Training HPL-CLIP

1. Clone the open_clip repo to your directory of choice: https://github.com/mlfoundations/open_clip.

2. From this HPL-Modified repo, open `scripts/train_clip.sh`. Edit as needed (e.g., activate correct conda environment, specify correct paths to open_clip repo, `logs`, `train-data`, etc.).

3. Submit `scripts/train_clip.sh` as a batch job. A TensorBoard log should appear in the open_clip log directory (e.g., `/gpfs/home/yb2612/dl4med_25/dl_project/results/logs/`).

4. Check for the TensorBoard log in the log directory (e.g.,`events.out.tfevents.1745262456.gpu-0003.121832.0`).

5. Forward port (e.g., 9199) from your remote machine to your local machine.

```
ssh -L 9199:localhost:9199 bigpurple
```

6. Activate a conda environment which has TensorBoard installed (e.g., `dl4med_25`).

```
conda activate dl4med_25
```

7. Run TensorBoard and point it to the log directory that contains the event files:

```
tensorboard --logdir /gpfs/home/yb2612/dl4med_25/dl_project/results/logs/ --port 9199 --host 0.0.0.0
```

This should return something like:

```
TensorFlow installation not found - running with reduced feature set.
TensorBoard 2.19.0 at http://0.0.0.0:9199/ (Press CTRL+C to quit)
```

8. Open the link from the output above in a web browser (e.g., http://0.0.0.0:9199/). This opens the TensorBoard interface, which contains training metrics, graphs, etc.