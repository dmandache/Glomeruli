## Glomeruli clasiffication using State of the Art models

### Pull Repo

 `eval $(ssh-agent)` 
 
`ssh-add ~/.ssh/id_rsa` 

`if cd Glomeruli; then git pull; else git clone https://github.com/dmandache/Glomeruli.git Glomeruli; fi; cd ~`
 
### Train.py

arguments:
* `--dir` path to data directory
* `--split` 
  * validation split percentage (recommended value : 20), randomly splits the data in train and test (a list of images chosen for each dataste is printed)
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


### Test_on_folder.py

arguments:
* `--dir` path to data directory
* `--model` path to saved (trained) model
