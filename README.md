# Filter Pruning via Feature Discrimination in Deep Neural Networks

## Introduction
We only provide the pruning code and process of resnet-50 here

Pipeline:
1. Assign network weight to assistant network
2. Pruning with assistant network
3. Fine-tuning

##  Running

We test our code on Python 3.7; CUDA Version: 10.2; PyTorch 1.10.0 

Pipeline code:

get the weights of the auxiliary network
```bash
python change_model.py
```
execute DLRFC for pruning
```bash
python DLRFC_ResNet50.py --batch-size 256 --lr 0.001
```
Put the results of the previous pruning (weight and structure) into main_ Finereturn to fine tune
```bash
python main_finetune
```

## Note
1) The feature maps of each layer of the network in this version of the code are obtained using the auxiliary network, and the hook function is not used. Of course, the hook function can also be used to obtain the feature maps of each layer of the network.

2 ) change_model.py is to assign the network weight to the assistant network to obtain the activation response.
### Pruning strategy

We introduce a novel pruning method in our paper (Fig. 3). its pseudocode is shown in Algorithm 1. 
