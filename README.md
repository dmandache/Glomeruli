## Glomeruli classification using State of the Art models

## Train.py

arguments:
* `--dir` path to data directory
* `--split` 
  * validation split percentage (recommended value : 20)
  * randomly splits the data in train and test sets (a list of images chosen for each dataste is printed)
  * if argument is omitted, data dir should have the structure :
      ```
        +-- train
        |   +-- glomeruli
        |   +-- nonglomeruli
        +-- test
        |   +-- glomeruli
        |   +-- nonglomeruli
      ```
* `--out` 
    * path to output directory
    * if argument is omitted, a folder is created, eg `./output_inception_finetune`
* `--model` specify model name: **inception / vgg / resnet / tiny**
* `--finetune` mention if you want to **finetune** on model pre-trained on ImageNet
* `--transfer` mention if you want to do **transfer learning** from model pre-trained on ImageNet
    * if both `--finetune` and `--transfer` are omitted, then the model is trained from scratch


## Test_on_folder_blind.py

Test model on images folder. Ground truth unknown.
Images split according to prediction as follows:
```
        +-- glom_prediction
        |   +-- glomeruli
        |   +-- nonglomeruli
```

arguments:
* `--dir` path to data directory
* `--model` path to saved (trained) model

### Test_on_folders.py

Test model on images organized in glomeruli / nonglomeruli subfolders (ground truth known).
Images split according to prediction as follows:
```
        +-- glom_prediction
        |   +-- false_negatives
        |   +-- false_positives
        |   +-- true_negatives
        |   +-- true_positives
```

arguments:
* `--dir` path to data directory
* `--model` path to saved (trained) model

## Test_on_patch.py

Test model on single image (patch, not WSI).

arguments:
* `--dir` path to data directory
* `--model` path to saved (trained) model

## Visualize.py

* `--model` path to saved (trained) model
* `--weights` mention if you want to plot the **weights** of the model
* `--activations` mention if you want to plot the **filter activations** of the model
    * `--img` path to image to plot activations for
* `--maxinput` mention if you want to plot the **maximum filter activation** of the model
