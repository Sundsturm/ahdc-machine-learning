# ECG Holter: Heart Rate Monitoring and Arrhythmia Detection & Classification Application

## About the Project
This ECG Holter project is limited to the creation of the ECG Holter interface/GUI and the artificial intelligence for arrhythmia detection and classification. The interface/GUI is built with the Python Tkinter framework. The artificial intelligence utilizes a 1D-CNN model to detect and classify arrhythmias. Thus, the implementation of the ECG Holter is carried out on the software side, with the hardware side already available.

This project was carried out over approximately two months, from July 7, 2025, to around August 29, 2025.

## Contributor
| Full Name            | Email      | Field          | Responsibility |
| -------- | -------- | -------- | -------- |
| Kean Malik Aji Santoso    | keanmalik92@gmail.com     | Electrical Engineering     | Artificial Intelligence for Arrhythmia Detection & Classification |

## Repositories/Drives
Several repositories were created and used for this project. There are also repositories/folders used as references for the project's execution. The list of repositories/folders can be seen below.
1. [ECG Holter GUI](https://github.com/gastyaadhyatmika/-KP-EKG-Holter-Analysis-for-Arrhythmia-Detection)
2. [Artificial Intelligence for Arrhythmia Detection & Classification](https://github.com/Sundsturm/ahdc-machine-learning.git)
3. [Google Colab on AI Training for Arrhythmia Detection & Classification](https://colab.research.google.com/drive/1miIRfWbtPMGr7ze8EtRrRoscNEkoW6Wq?authuser=1#scrollTo=1SPEEx14l3u8)
4. [Figma for ECG Holter GUI](https://www.figma.com/design/T4ZLz3nBal5QjaI9wrwj3r/Mockup-ECG-Holter_PT-Xirka-Darma-Persada?node-id=3-2&t=fHxEvnm2tz4XCsaE-1)

## Specifications
### AI for Arrhythmia Detection & Classification Specifications
#### ML Training
Machine learning is implemented using **supervised learning**. Three machine learning models are used: 
1. **`MLP`**;
MLP or multi-layer perceptron is a fundamental form of artificial neural network consisting of several fully connected layers of neurons. Each neuron in one layer is connected to every neuron in the next. The neuron layers used are dense and dropout layers.
2. **`CNN`**; and
CNN or convolutional neural network is a type of machine learning model in the form of a neural network that excels at processing data with a grid-like structure, such as images or raw ECG signals. This machine learning model can automatically detect and learn hierarchical features from simple edges to complex patterns with convolutional layers.
3. **`Random Forest`**.
Random Forest is an ensemble learning model that works by building many decision trees during training. To make predictions, the model gathers the output from all trees and takes a majority vote for classification or the average for regression. This makes the training results more accurate and resistant to overfitting than a single tree.

The results of the artificial intelligence training can be determined based on specific metrics. The primary metrics for this training are:
1. **`Accuracy`**: Measures how often the model makes correct predictions out of the total data. It is calculated as the ratio of the number of correct predictions (both positive and negative) to the total number of predictions;
2. **`Precision`**: Focuses on the quality/consistency of positive predictions, or the percentage of truly positive predictions out of all positive predictions made by the model;
3. **`Recall/Sensitivity`**: Measures the model's ability to find all actual positive class samples, or the percentage of correctly predicted positive instances out of all actual positive instances;
4. **`F1-Score`**: The harmonic mean of Precision and Recall/Sensitivity, which balances both metrics; and
5. **`Specificity`**: Measures the model's ability to correctly identify negative class samples, or the percentage of truly negative predictions out of all actual negative instances.

#### Database
The database for the arrhythmia detection & classification AI is the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). In detail, this database can be described as follows.
* Utilized one-lead/two-lead cables
* Recorded at a frequency of 360 Hz, 11-bit resolution, and a magnitude range of 10 mV
* The database has its own labeling, which needs to be converted to follow the **AAMI EC57 standard**, as explained in the table below:

| Classification of AAMI EC57 | Classification of MIT-BIH |
| -------- | -------- |
| Normal Beat (N)     |  Normal Beat ( N atau . ), Left Bundle Branch Block Beat ( L ), Right Bundle Branch Block Beat ( R ), Atrial Escape Beat ( e ), Nodal/Junction Escape Beat ( j )    |
| Ventricular Ectopic Beat/VEB (V)     |  Premature Ventricular Contraction (V), Ventricular Escape Beat (E)    |
| Supraventricular Ectopic Beat/SVEB (S)     | Atrial Premature Beat (A), Aberrated Atrial Premature Beat (a), Nodal/Junctional Premature Beat (J), Supraventricular Premature Beat (S)     |
| Fusion Beat (F)     |   Fusion of Ventricular and Normal Beat (F)   |
| Unknown Beat (Q)     |  Paced Beat (P atau /), Fusion of Paced Beat and Normal Beat (f), Unclassifiable Beat (U)    |

## Project Implementation
### AI Implementation
#### Implementation Progress (August 31, 2025 & September 1, 2025)
So far, the implementation of the artificial intelligence for arrhythmia detection & classification is still in the training phase, focused on training the 1D-CNN model because it is better at detecting raw data, especially ECG signal data. The latest training results show that the model performs well in the normal class, fairly well in the ventricular class, but not yet well in the supraventricular and fusion classes. Further training is needed for this model or the other two models. If the model's performance improves, it can proceed to the model implementation stage, where it will be integrated into the ECG Holter GUI.
#### Google Colab Explanation
The Google Colab file contains training with `Windowed Features` data on the three models. Additionally, there is an automatic hyperparameter tuning feature for the MLP model. Unfortunately, this feature was not implemented for each model because tuning can be done manually through single-fold validation or ten-fold cross-validation.
#### Jupyter Notebook Explanation
There are four Jupyter Notebook files used during training:
1. **`training-A.ipynb`**: Training of the three models with Windowed Features data and single-lead training on the MLII lead;
2. **`training-B.ipynb`**: Training of the three models with WPT data for the MLP and Random Forest models and Raw Data for the 1D-CNN model, as well as dual-lead training based on the MIT-BIH Arrhythmia Database;
3. **`training-C.ipynb`**: Training of the three models with DWT data for the MLP and Random Forest models and Raw Data for the 1D-CNN model, as well as dual-lead training based on the MIT-BIH Arrhythmia Database; and
4. **`training-D.ipynb`**: Training of the three models with Raw Data and single-lead training on the V5 lead.

#### Computing Power on Each AI Training Method
Training with Google Colab often uses the CPU provided by Google Colab. Local training utilizes a **LOQ 15IAX9E** laptop with an **Intel Core i5-12450HX 2.4GHz CPU**, an **NVIDIA Laptop GPU RTX3050 6GB**, and **12GB RAM**. Local training often crashes in Jupyter Lab when running two or more Jupyter Notebook files simultaneously because the laptop has limited RAM and uses Windows 11, which is bloated or resource-heavy.

#### Data Preparation & Preprocessing
The artificial intelligence for arrhythmia detection and classification is trained with **additional ECG data from an arrhythmia simulator** and the **MIT-BIH Arrhythmia Database** dataset. The MIT-BIH Arrhythmia Database is split according to **De Chazal et al.**, and the split can be seen in the table below.
| Dataset | Patients IDs | Dataset Usage|
| -------- | -------- | -------- |
| `ds1`     | '101', '106', '108', '109', '112', '114', '115', '116', '118', '119','122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230' | Training & Validation
| `ds2`     | '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'     | Testing     |

The **additional ECG dataset from the arrhythmia simulator** is used as a **testing dataset** along with the `ds2` dataset.

Sequentially, the data preparation and preprocessing flow is as follows:
1. Load the MIT-BIH Arrhythmia Database and the ECG data from the arrhythmia simulator, and split the dataset as explained previously;
2. Perform data preprocessing for the ECG signals from each dataset;
3. Resample the base frequency of the MIT-BIH Arrhythmia Database from 360 Hz to 500 Hz to match the ECG data from the simulator, transform the data to the desired format, and label the MIT-BIH Arrhythmia Database dataset;
4. Read the arrhythmia simulator ECG data, which is in .bin format, transform the data to the desired format, and label the ECG signal dataset from the arrhythmia simulator;
5. Perform standard scaling and combine the processed data from step 3 and step 4;
6. Split the combined data into several sets for training (80%) and validation (20%), and apply the SMOTEEN/SMOTE algorithm to the training set; and
7. Assign the split data sets to each model to be trained.

The **data preprocessing** process utilizes **several digital filters** to remove baseline wander, electrical interference noise, and high-frequency noise in the ECG signal data. During model training, several forms of database transformations were performed. These transformations can be explained as follows:
1. **`Windowed Features (All models)`**: Time duration data from a window that holds every 10 RR intervals based on R-peak detection at a specific time;
2. **`Raw Data (1D-CNN)`**: The raw ECG signal is used as data for training and testing the model, especially for the used CNN model;
3. **`Discrete Wavelet Transform/DWT (MLP & Random Forest)`**: Transformation of the raw ECG signal into a discrete transform using the Daubechies 4 method;
4. **`Wavelet Packet Transform/WPT (MLP & Random Forest)`**: Transformation of the raw ECG signal into vector data containing detailed morphological analysis of the signal at frequency levels, resulting in vector data of energy, Shannon entropy, and basic morphological statistics;
5. **`Scalogram (2D-CNN)`**: Visualization of the ECG signal as a color intensity graph representing wavelet coefficients at a specific time. This visualization is created based on the time-domain and frequency-domain aspects of the ECG signal with a continuous wavelet transform (CWT); and
6. **`Tabular Features (MLP & Random Forest)`**: ECG signal data organized in a tabular format containing information on heart rate variability results and R-peak amplitude, thus containing no information about the shape/morphology of the ECG signal.

The *last multi-model training* used a combination of **raw data for the 1D-CNN model** and **WPT data for the MLP and Random Forest models**. The **very last training** utilized **raw data for the 1D-CNN model**.

#### Model Training
The training utilizes three models and the explanation for each modelcan be seen below.
1. **`MLP`**
    * Built with TensorFlow Keras
    * The layers are
        1. `Dense 1`: 512 neurons with input dimension hyperparameter and utilizes rectified linear unit (ReLU);
        2. `Dropout 1`: A regulator/overfitting preventer that reduces 10%/(0.1) neurons from the last layer;
        3. `Dense 2`: 512 neurons with ReLU;
        4. `Dropout 2`: A regulator/overfitting preventer that reduces 40%/(0.4) neurons from the last layer;
        5. `Dense 3`: This layer has number of neurons that are as same as the number of classes/labels (`output_dim`) and it also converts raw outputs into a probability distribution (`softmax`).
    * The recent training result can be seen as follows (August 31, 2025).
![Screenshot 2025-08-31 043524](https://hackmd.io/_uploads/r11jONXcll.png)

2. **`Random Forest`**
    * Built with cuML from RAPIDS AI so that the learning process utilizes GPU 
    * Consists of only setting the number of decision trees (`n_estimators`), which is 200, with a growth depth (`max_depth`) reaching level 30 and random state (`random_state`) of 42
    * The recent training result can be seen as follows (August 31, 2025).
![Screenshot 2025-08-31 043449](https://hackmd.io/_uploads/SkAadVXcxx.png)

3. **`1D-CNN`**
    * Built with TensorFlow Keras
    * Has `epoch=150` and `batch_size=100`
    * The utilized layers can be seen below.
![Screenshot 2025-08-26 153255](https://hackmd.io/_uploads/H1NvQR0Kxg.png)
    
    There are changes on **trainable layers hyperparameters**: **100, 256, dan 0.0001** for **Conv1D filters, Dense layer (Not Dense Output Layer), and Adam (Optimizer) learning rate**.
    * The recent training result can be seen as follows (September 1, 2025).
![image](https://hackmd.io/_uploads/HkhoUNm5le.png)

All training that has been conducted can be modified to achieve better performance.

#### Suggestions/Future Works
1. Tune the hyperparameters of the trainable layers in the 1D-CNN until the most optimal results are found
2. Tune the hyperparameters of the other models, namely MLP and Random Forest, until optimal results are found
3. Create separate Python files for each stage of the pipeline if the training results are good, so the repository can be easily reproduced
4. Review, understand, replicate, and modify references from GitHub related to machine learning model training for arrhythmia detection and classification
5. Explore other papers related to artificial intelligence for arrhythmia detection and classification
6. Implement the exported model into the ECG Holter GUI if the model's performance is excellent

## How to Use the Project
### Creating Virtual Environment and Using a Python Package Manager
1. Initialize the virtual environment;
```code
python -m venv <virtual environment folder name>
```
2. Acitvate the virtual environment; dan
```code
.\<virtual environment folder name>\Scripts\activate # Windows
source <virtual environment folder name>/bin/activate # Linux/MacOS 
```
3. Use `pip` as the package manager for installing required packages from a certain script with format of .txt
```code
pip install <package 1>==<package 1 ver.> <package 2>==<package 2 ver.> ....
pip install -r <script name>.txt # Installing packages using .txt script
```
### Using the Repository and Google Colab of AI Training
For Google Colab, you just need to open the file, select the available computing power in Google Colab, and start training. Training in Google Colab is easier than local training, but the computing power is limited to a certain duration. Therefore, local training is the more recommended method.

The steps for local training are
1. Clone the repository;
```
git clone https://github.com/Sundsturm/ahdc-machine-learning.git
```
2. Initialize a virtual environment in the project repository/folder and activate it;
3. Before installing the packages for this project, please install TensorFlow that uses a GPU **([TensorFlow Installation Tutorial](https://www.tensorflow.org/install))**, whether for AMD or NVIDIA, because the Random Forest model built with cuML, which utilizes the GPU for its training;
4. Install the packages with `pip` via the `requirement.txt` file in the repository;
5. Run Jupyter Lab/Jupyter Notebook; and
```
jupyter lab # Jupyter Lab
jupyter notebook #  Jupyter Notebook
```
6. Check and run the `.ipynb` files in the `notebooks` folder to perform model training.

## References
1. A. Raza, K. P. Tran, L. Koehl, and S. Li, "Designing ECG monitoring healthcare system with federated transfer learning and explainable AI," Knowledge-Based Systems, vol. 236, p. 107763, Jan. 2022, doi: 10.1016/j.knosys.2021.107763.
2. G. Silva, P. Silva, G. Moreira, V. Freitas, J. Gertrudes, and E. Luz, "A Systematic Review of ECG Arrhythmia Classification: Adherence to Standards, Fair Evaluation, and Embedded Feasibility," arXiv preprint arXiv:2503.07276, 2025. [Online]. Available: https://arxiv.org/abs/2503.07276
3. S. Aziz, S. Ahmed, and M.-S. Alouini, "ECG-based machine-learning algorithms for heartbeat classification," Sci. Rep., vol. 11, no. 1, Art. no. 18738, Sep. 2021, doi: 10.1038/s41598-021-97118-5.
4. Y. Ansari, O. Mourad, K. Qaraqe, and E. Serpedin, "Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017-2023," Front. Physiol., vol. 14, Art. no. 1246746, Sep. 2023, doi: 10.3389/fphys.2023.1246746.








