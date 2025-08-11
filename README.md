# Wheat-Disease-Detection
It performs multiclass classification to identify different wheat diseases (here black point, fusarium foot rot, lead blight, wheat blast and healthy leaf).
# Dataset
The dataset used in this project is not my own. I am not the creator or owner of this dataset, all the rights and credits belong to the original authors:
>Radowan, Md Istiak Rahman; Ayon, Rokonozzaman Ayon (2025),
>“Disease Dataset of Wheat: Original, Augmented, and Balanced for Deep Learning”,
>Mendeley Data, V1, doi: 10.17632/5gc7hwydwg.1
This dataset is hosted on Mendeley Data and is the property of its respective authors.
# Authors
This project was jointly developed by:
- [Khushpreet Sindhu](https://github.com/Khushpreetsindhu) - Model develpoment, data preprocessing, evaluation
- [Eshani Patel](https://github.com/eshani-pa) - Model development, data preprocessing, evaluation
- Mahak raza Sheikh - Literature review and research
# Abstract
This study proposes an automated multiclass classification for different wheat diseases.With a combination of deep learning with transfer learning and attention mechanisms this study aims to identify an optimal model for real world deployment. The model is based on EfficientNetB0, pretrained on ImageNet which is both speed and light weight enhanced with a Convolutional Block Attention Module (CBAM) to improve feature representation through channel and spatial attention. A labeled wheat crop dataset was pre-processed and split into training (70%), validation (10%), and test (20%) sets. Extensive on the fly augmentation including geometric and color transformations, was applied to the training set to boost model robustness and reliability. Training involved frozen convolutional layers followed by selective fine-tuning for domain adaptation,
using the Adam optimizer and categorical cross-entropy loss, with early stopping to avoid overfitting. 
# Methodology
There are two approaches used with each approach consisting three models for optimal model selection.
## Approach 1
Approach 1 uses the 'split dataset' and  the following applies three models to it:
1. EfficientNet-B0 + CBAM
2. Custom CNN model
3. MobileNet-V2
## Approach 2
Approach 2 uses the 'original dataset' and splits it first into training (70%), testing (20%) and validation sets (10%) and then applies on-the-fly augmentation to the training set. This Approach also uses threee models.
1. EfficientNEt-B0 + CBAM
2. MobileNet-V2 + CBAM
3. Custom CNN model
# Results
The EfficientNEt-B0 + CBAM model belonging to the approach 2 is selected as the optimal model as it fixes the data leakage caused by approach 1 and is more reliable out of all. 
