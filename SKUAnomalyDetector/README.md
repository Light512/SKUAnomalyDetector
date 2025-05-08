Methodology
This project uses an unsupervised learning approach to identify visually distinct (mixing/anomalous) items among a main class of SKU images. Here's a step-by-step breakdown of the pipeline:

1. Feature Extraction using DINOv2
The model used is facebook/dinov2-base, a state-of-the-art Vision Transformer.

For each image:

It's preprocessed using the corresponding HuggingFace AutoImageProcessor.

Passed through the model to extract deep visual features.

We use the mean of the last hidden state as the image's embedding, which captures global semantic information.

2. Feature Normalization and Dimensionality Reduction
Extracted feature vectors are standardized using StandardScaler to ensure consistent scaling.

Then PCA (Principal Component Analysis) is applied to reduce dimensionality while retaining 95% of variance (or up to 50 components).

This step makes clustering and anomaly detection more effective and faster.

3. Anomaly Detection using Isolation Forest
We apply IsolationForest, an unsupervised algorithm specifically designed for anomaly detection.

It works by isolating outliers via recursive random partitioning of the feature space.

Each image receives an anomaly score:

Images with score below a threshold (e.g., score < 0) are labeled as anomalies (likely mixing items).

Remaining images are considered normal (main class).

4. Results Compilation
Each image is associated with:

Its original folder/class,

A binary anomaly flag: 1 for anomaly (mixing), 0 for normal.

Final results are saved to a CSV file clustered_output.csv for easy review and downstream usage.

This approach is robust even with a small number of images and works without labeled data, making it highly suitable for real-world SKU validation and anomaly screening tasks.