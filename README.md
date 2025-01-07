# Grade my Writing

## Overview
The goal of this project is to assess the language proficiency of 8th-12th grade English Language Learners (ELLs). By using a dataset of essays written by ELLs, a model was trained to grade the writing on 6 metrics.
## Model Description
The project utilizes **DeBERTa-v3-large**, a state-of-the-art encoder-based language model. Key features and modifications include:

1. **Feedback Model Architecture**:
   - A custom model was built using the DeBERTa-v3-large transformer as the base. The architecture includes:
     - A transformer backbone loaded with pre-trained weights.
     - Multiple dropout layers with probabilities ranging from 0.1 to 0.5 for robust training.
     - A fully connected linear layer to output predictions for six metrics: 
       - *Cohesion*
       - *Syntax*
       - *Vocabulary*
       - *Phraseology*
       - *Grammar*
       - *Conventions*
     - The outputs are averaged across multiple dropout layers to enhance stability and reduce overfitting.

2. **Fine-tuning**:
   - The model was fine-tuned on the ELL essay dataset to score the six metrics on a 0-5 scale, achieving high accuracy.
3. **Example Result**
   ![Scoring image](https://github.com/Sushmita10062002/GradeMyWriting/blob/master/images/img1.png)
## Scoring Metric
**Mean Columnwise Root Mean Squared Error (MCRMSE)**, defined as:

![Scoring image](https://github.com/Sushmita10062002/GradeMyWriting/blob/master/images/img2.png)

MCRMSE measures the accuracy of predictions across multiple target columns.

## Dataset
You can access the ELL dataset [here](https://www.kaggle.com/competitions/feedback-prize-english-language-learning). After forking the code, please add the data in the `inputs` folder if you want to train your own model.

## Results
The fine-tuned model achieved MCRMSE loss of **0.43**. 

## Future Work
- Explore other approaches and models to reduce the loss on the task.
- Deploy the model and create a web app so that others can access it to score their writing.
