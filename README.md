# Welcome to the GitHub Repository of the UNIL_Zoom Team!

Formed by [Adam Monro](https://github.com/AdamMonroUnil) and [Cailen Hughes](https://github.com/cailenhughes), our team is participating in the ![Kaggle Competition](https://www.kaggle.com/static/images/site-logo.png) entitled [*Detecting the Difficulty Level of French Texts*](https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2023).

This competition is an evaluation tool for the *Data Mining & Machine Learning* course, given by Prof. Vlachos at the University of Lausanne.  
![UNIL Logo](https://hecnet.unil.ch/medias/plone/lg14/logo_unil.png)

Have a look at the: 
* [Syllabus of the course](https://www.unil.ch/hec/en/home/menuinst/masters/systemes-d-information/cours-et-horaires.html?url=/web/syllabus/2886)
* [GitHub of the course](https://github.com/michalis0/DataScience_and_MachineLearning)

You can find all of our files on this GitHub repository.

## Our Approach and Code Insights

In our project for the Kaggle competition "Detecting the Difficulty Level of French Texts", we employed advanced machine learning techniques to analyze and classify French texts based on their difficulty levels.

### Key Components of Our Code

1. **Use of Camembert for Text Classification**: 
   - We leveraged the power of the Camembert model, a variant of the RoBERTa model pre-trained on French text, for sequence classification.
   - The model was fine-tuned on our dataset to classify texts into different difficulty levels, enabling us to harness deep learning for natural language understanding.

2. **Data Preprocessing and Tokenization**:
   - Extensive data preprocessing was conducted using pandas and sklearn's LabelEncoder.
   - The CamembertTokenizer was utilized for tokenizing text data, preparing it for model training.

3. **Training and Validation Split**:
   - We split our dataset into training and validation sets, ensuring a robust model training and evaluation process.

4. **Model Training and Evaluation**:
   - The model was trained using the Trainer class from the transformers library, with specified training arguments for optimal performance.
   - Post-training, the model's performance was evaluated on unseen data, ensuring its effectiveness in real-world scenarios.

5. **Predictions and Results Analysis**:
   - For final predictions, our model was evaluated on a test dataset, and the results were analyzed to understand model performance.

### Logistic Regression with TF-IDF Vectorization:

In addition to the deep learning approach, we also implemented a machine learning pipeline using Logistic Regression coupled with TF-IDF Vectorization. This method provided a comparative analysis and an alternative approach to text classification.

- The pipeline involved transforming text data into TF-IDF vectors and then applying Logistic Regression for classification.
- We evaluated the model's performance using standard metrics like accuracy, precision, recall, and F1-score, ensuring a comprehensive understanding of its strengths and limitations.

### Insights and Learning:

Through this project, we gained valuable insights into the application of both deep learning and traditional machine learning techniques in natural language processing. Our experience with data preprocessing, model tuning, and performance evaluation has been enriching, contributing significantly to our understanding of text classification challenges and solutions.

### Conclusion:

Our journey in this Kaggle competition has been a blend of challenges and learning. We invite you to explore our repository for a deeper dive into our code, methodologies, and findings. Your feedback and contributions are highly welcome!
