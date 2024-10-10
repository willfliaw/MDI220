# Statistics (MDI220) - 2023/2024

## Course Overview

This repository contains materials and resources for the course **MDI220: Statistics**, part of the **Mathematics** curriculum. The course provides essential tools for practicing statistics and data analysis. It covers fundamental concepts such as statistical models, point estimation, hypothesis testing, Bayesian methods, and confidence intervals, with practical examples and Python programming exercises.

### Key Topics:

- Statistical Models: Understanding and building models for data analysis.
- Point Estimation: Maximum likelihood estimation, method of moments, and properties of estimators.
- Quadratic Risk: Concepts such as the Cramer-Rao lower bound.
- Decision Procedures: Hypothesis testing and Bayesian approaches.
- Confidence Intervals: Methods for constructing intervals with given confidence levels.
- Bayesian Methods: Introduction to Bayesian inference and applications.

## Prerequisites

Students are expected to have knowledge of:
- Probability Theory: Conditional expectations, Gaussian vectors.
- Programming: Python programming is required for practical exercises.

## Course Structure

- Total Hours: 24 hours of in-person sessions (16 sessions), including:
  - 9 hours of directed study (TD)
  - 10.5 hours of lessons
  - 1.5 hours of practical exercises
  - 3 hours of knowledge assessment (final exam)
- Estimated Self-Study: 38.5 hours
- Credits: 2.5 ECTS
- Evaluation: Final written exam (3 hours).

## Instructor

- Professor Pavlo Mozharovskyi

## Installation and Setup

Some exercises and projects require Python and statistical libraries. You can follow the instructions below to set up your environment using `conda`:

1. Anaconda/Miniconda: Download and install Python with Anaconda or Miniconda from [Conda Official Site](https://docs.conda.io/en/latest/).
2. Image Processing Libraries: Create a new conda environment with the necessary packages:
   ```bash
   conda create -n mdi220 python numpy pandas matplotlib scipy scikit-learn jupyter ipykernel seaborn
   ```
3. Activate the environment:
   ```bash
   conda activate mdi220
   ```
4. Launch Jupyter Notebook (if required for exercises):
   ```bash
   jupyter notebook
   ```

This setup will allow you to complete practical exercises related to statistical analysis.

## How to Contribute

Feel free to contribute to the repository by:
- Submitting pull requests for corrections or improvements.
- Providing additional examples or extending the projects.
