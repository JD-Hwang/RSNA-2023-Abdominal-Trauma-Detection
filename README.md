# RSNA-2023-Abdominal-Trauma-Detection

> RSNA 2023 Kaggle competition (https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
- Solution of *Team MI2RL&CILAB* / *Team SANGMOON*  
  - Graphical description of the solution submitted.
![image](https://github.com/JD-Hwang/DNN_TL4fMRI/assets/65854964/6878d8eb-cd87-4a41-aada-a4c29fe99c0c)

1. We trained segmentation model first, to separate each condition (liver, kidney, bowel, spleen, extravasation) for better injury classification.
2. With input resize to 128x224x224, inference using trained segmentation model took 3~4 seconds per data.
3. Classifiers for each model were trained separately, then gathered at the final to get single log loss.
