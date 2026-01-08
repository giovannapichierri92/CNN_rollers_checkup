
# 🛼 CNN_rollers_checkup – Analysis of Roller Skate Wheel Wear and Optimal Repositioning

This project uses deep learning models to automatically analyze roller skate wheel wear and suggest the optimal positioning on the skate based on the level of wear.
It also includes a Streamlit web app for performing analyses via a graphical interface and a chatbot for requesting technical specifications.

## 🚀 Main Features

- Wear classification using neural networks.

- Fine-tuning of three models:

  - MobileNetV3

  - ResNet18

  - Vision Transformer (ViT)

- Performance comparison on a dedicated test set.

- Automatic wheel repositioning algorithm based on wear.

- Streamlit web app with:

  - Graphical interface for image upload and analysis

  - Display of uploaded images

  - Automatic repositioning

  - Integrated chatbot

- Sample dataset, preprocessed images, and 3D-printed support for photography.

- CSV files with model performance metrics.

## 🧠 Models Used

Three models were trained and compared:

| Modello         | Architettura       | Note                                          |
| --------------- | ------------------ | --------------------------------------------- |
| **MobileNetV3** | Lightweight CNN    | Great trade-off between speed and accuracy    |
| **ResNet18**    | Classic CNN        | Stable and high-performing pipeline           |
| **ViT**         | Vision Transformer | Excellent performance on complex images       |


The results are available in the CSV files in the /CSVS/model_performance folder.

## ▶️ Web App Info

The UI allows you to:

- Upload wheel photos

- View processed images

- Estimate wear level

- Automatically reorder wheels

- Interact with a chatbot for technical questions

## 📷 Dataset and 3D Support

The **IMAGES/** folder contains:

- Wheel images used for testing

- Preprocessing outputs

- Photos of the 3D-printed support for controlled wheel acquisition

## 📌 Possible Future Developments

- Dataset expansion

- Improvement of the chatbot with a model trained on skates/wheels domain

- Exporting the algorithm to a mobile app

## 📄 Presentation PDF

The project presentation PDF is CNN_rollers_checkup.pdf

## 📁 Repository Structure

```text
├── APP_STREAMLIT/                        # Folder for Streamlit web app
|
├── CODES/
│   ├── MobileNetV3-Large.ipynb            # MobileNet training
│   ├── ResNet18_regression.ipynb          # ResNet training
│   ├── vit_regression.ipynb               # ViT Regression training
│   ├── calcolo_mse.ipynb                  # Metrics and model comparison
│   ├── find_edges.ipynb                   # Testing optimal Canny parameters and preprocessing study
│   └── riordinamento_ruote.ipynb          # Optimal wheel repositioning algorithm
|
├── CSVS/
│   ├── test_predictions_mobilenetv3.csv   # Test set results for MobileNet
│   ├── test_predictions_resnet.csv        # Test set results for ResNet
│   └── test_predictions_vit.csv           # Test set results for ViT
|
├── IMAGES/
│   ├── edge_tests/                        # Examples of edge detection with Canny
│   ├── ruote_catalogate_def/              # Examples of organized input wheels
│   └── IMG-sostegno3D.jpg                 # Image of 3D-printed support
|
├── MODELS/                                # MobileNet model after fine-tuning
│   └── regression_mobilenetv3_finetuned.pth
|
└── CNN_rollers_checkup.pdf                # Project presentation PDF
