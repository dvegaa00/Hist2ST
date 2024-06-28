# Hist2ST Adapted for the e Spatially Resolved Expression Database (SpaRED)

Hist2ST is a deep learning-based model developed by Zeng, which uses histology images to predict RNA-seq expression. At each sequenced spot, the corre-sponding histology image is cropped into an image patch, from which 2D vision features are learned through convolutional operations. Meanwhile, the spatial relations with the whole image and neighbored patches are captured through Transformer and graph neural network modules, respectively. These learned features are then used to predict the gene expression by following the zero-inflated negative binomial (ZINB) distribution. To alleviate the impact by the small spatial transcriptomics data, a self-distillation mechanism is employed for efficient learning of the model. Hist2ST was tested on the HER2-positive breast cancer and the cutaneous squamous cell carcinoma datasets, and shown to outperform existing methods in terms of both gene expression prediction and following spatial region identification.
       
![(Variational) gcn](Workflow.png)

To apply Hist2ST to the SpaRED database, we have adapted the original repository to fit the standardized format of SpaRED datasets. This adaptation allows researchers to seamlessly use Hist2ST for spatial transcriptomics prediction on both SpaRED and any custom datasets that follow the SpaRED format.

## Setup 
To setup the enviroment and install the required libraries run the following commands on your terminal:

```bash
conda create -n "myenv" python=3.8.0

conda activate myenv

pip install -r requirements.txt
```

# Run Hist2ST 

### Training and Evaluation on SpaRED Dataset

To train and evaluate Hist2ST on the SpaRED datasets you must run the following command:

```bash
python HIST2ST_main.py --dataset $dataset_name$ --prediction_layer $prediction_layer$
```
* Replace $dataset_name$ with the name of any dataset available in SpaRED (default = None).
* Replace $prediction_layer$ with the layer used to train and evaluate the model (default = c_t_log1p). 

This command loads the preprocessed dataset directly from the SpaRED repository and makes predictions on the specified layer. The results are logged in Weights and Biases.

### Training and Evaluation on Custom Datasets:

To train and evaluate Hist2ST on a custom dataset, you must first ensure that your AnnData object (adata) follows the SpaRED format. Then, run the following command:

```bash
python HIST2ST_main.py --path_adata $path_to_adata$ --prediction_layer $prediction_layer$
```
* Replace $path_to_adata$ with the path to your preprocessed adata file (in .h5ad format) (default = None).
* Replace $prediction_layer$ with the layer used to train and evaluate the model (default = c_t_log1p).

When `$path_to_adata$` is provided, the model automatically loads the adata from the specified path and ignores the `dataset` parameter, which is used to load an available SpaRED dataset.

This command loads the custom preprocessed dataset directly from the specified path and makes predictions on the defined layer. The results are logged in Weights and Biases. 

# Citation

Please cite Hist2ST paper:

```
@article{zengys,
  title={Spatial Transcriptomics Prediction from Histology jointly through Transformer and Graph Neural Networks},
  author={ Yuansong Zeng, Zhuoyi Wei, Weijiang Yu, Rui Yin,  Bingling Li, Zhonghui Tang, Yutong Lu, Yuedong Yang},
  journal={biorxiv},
  year={2021}
 publisher={Cold Spring Harbor Laboratory}
}

```
