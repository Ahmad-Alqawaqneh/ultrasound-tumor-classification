# Ultrasound Tumor Classification (Computer Vision)

Image classification project from the **Tahaluf Al Emarat Machine Learning program** Kaggle competition
([competition page](https://www.kaggle.com/competitions/tumor-classification/overview)).
The goal is to classify ultrasound images into **Benign**, **Malignant** or **Normal**.

## Approach
- **Image preprocessing** – resizing (100×100 / 200×200), grayscale conversion, cropping and reflection.
- **Filter-based feature engineering** – compared Gabor kernels, Sobel, Scharr, Prewitt, Roberts, Laplacian, bilateral / Gaussian / median blur, morphology, entropy and thresholding (OpenCV, scikit-image).
- **Class imbalance** – augmentation of minority classes with **Augmentor** (rotation, flips, zoom).
- **Dimensionality reduction** – PCA on pixel / filter features.
- **Models** – Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, KNN, SVC, XGBoost, **LightGBM**; tuned with GridSearchCV and Stratified K-Fold, evaluated with learning curves and ROC / PR curves.

## Results
- Baseline on raw pixels: about 0.64 accuracy.
- With filters + PCA + gradient boosting: about 0.69–0.72.
- With **augmentation + PCA + LightGBM / Random Forest**: **about 0.81 accuracy**.

## Notebooks
- `notebooks/image_filters_feature_extraction.ipynb` – comparison of image filters as features
- `notebooks/tumor_classification_augmentation.ipynb` – augmentation, PCA and final models
- `notebooks/gradient_boosting_pca.ipynb` – filter datasets + PCA + gradient boosting / KNN (best ≈ 0.72)
- `notebooks/reflection_cropping_experiments.ipynb` – reflection and cropping augmentation experiments
- `app/ultrasound_gui.py` – Tkinter desktop app: enter an image number, see the image and the model prediction (needs `trained_model.pkl`)
- `presentation/ultrasound_presentation.pdf` – team presentation of the solution

**Tech:** Python · Tkinter · OpenCV · scikit-image · Augmentor · scikit-learn · LightGBM · XGBoost

> Team project during the Tahaluf Al Emarat ML internship. Images are not included; download them from the competition page into `./data/`.
