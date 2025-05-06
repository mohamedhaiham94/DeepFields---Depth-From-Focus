# DeepFields - Depth From Focus

TBD

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── dof   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes dof a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Main Idea
The primary objective of this project is to distinguish between in-focus and out-of-focus regions within focal stacks using a classification model—specifically, a Multi-Layer Perceptron (MLP). This task aims to support focus-aware image analysis in synthetic-aperture imaging.


The project addresses the following research questions:


- What is the most effective combination of input features (e.g., standard deviation (STD), entropy, spatial STD)?
- How does the sampling strategy (Top-Down vs. Circular) influence model performance?
- Can a model trained on simulated data generalize effectively to real-world data?

# Pipeline flow

### Feature Stack Computation:
- Compute focal stacks (integral images) and corresponding feature maps (STD, entropy, spatial STD) in the angular domain.

### Ground-Truth Depth Generation:
- Generate a depth map using Blender, with rendering parameters identical to those used for focal stack generation (e.g., field of view (FOV), virtual FOV, sampling spacing, start and end positions).

### Data Assembly:

- For each focal stack layer, combine the feature stack values and corresponding depth map data into a per-pixel CSV file.

### Training Data Preparation:
- Filter depth data using a predefined threshold equal to the layer spacing in the focal stack.
- Label each pixel as follows:
    - 1: Pixel is in focus (depth matches focal layer).

    - 0: Pixel is out of focus (depth lies outside the focal layer).
- Balancing is done by downsampling the out of focus to match the number of focus pixels.
- Use the MLP model to train on the labeled data.

### Evaluation and Visualization:
- Generate binary classification images from test results to enable 3D visual comparison.
- Report performance using accuracy metrics and a full classification report.


# ToDO

- Compare model performance and reconstruction quality between Top-Down and Circular sampling strategies across different feature combinations.



# Findings

- Angular STD consistently outperforms entropy as an input feature.


- Combining STD and entropy yields only marginal improvement in classification accuracy.


- The depth filtering threshold should match the spacing between focal stack layers for optimal label accuracy


