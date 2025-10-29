yt-chrome-plugin
==============================

An ML project that scrapes the comments from the current video using the YouTube API, analyses them and gives them sentiment {positive, negative, neutral} using an ML model from flask_app, which operates as a  backend.

This is the final, complete, and fully formatted **README.md** file, incorporating all the project details, the MLOps components (MLflow, CI/CD, DVC), the deployment status, and the local reproduction guide you requested.

You can **copy and paste this entire block** directly into the `README.md` file of your main GitHub repository (`yt-chrome-plugin`).

-----

# 🌟 YouTube Comment Sentiment Analyser 🤖💬

### Project Overview

The **YouTube Comment Sentiment Analyser** is a two-part application that provides real-time sentiment analysis of comments on any YouTube video. It consists of a **Chrome Extension (Frontend)** communicating with a containerized **Python/Machine Learning (ML) API (Backend)** to classify comments as **Positive, Negative, or Neutral**.

This project demonstrates a complete MLOps workflow, focusing on model reproducibility and deployment readiness.

-----

### 🛠️ Technology Stack & Architecture

| Component | Repository | Primary Technologies | MLOps Tools |
| :--- | :--- | :--- | :--- |
| **Backend (ML API)** | [srvmann/yt-chrome-plugin](https://github.com/srvmann/yt-chrome-plugin) | Python, Flask, Custom ML Model, Docker | **MLflow**, **GitHub Actions** (CI), DVC |
| **Frontend (Chrome Extension)** | [srvmann/chrome-plugin-frontend](https://github.com/srvmann/chrome-plugin-frontend) | JavaScript, HTML, Chrome Manifest | N/A |

-----

### 🏆 Project Excellence & MLOps Highlights

This project utilizes industry tools for governance and automation, ensuring reproducibility and quality.

#### 1\. MLflow Tracking Server (Experiment Reproducibility)

All model training runs, parameters, and performance metrics are logged and tracked using **MLflow**, providing full auditability and the ability to reproduce any model version.

  * **🌐 MLflow Tracking Server:** `http://13.203.227.156:5000/`

#### 2\. Continuous Integration (CI) with GitHub Actions

The code base is protected by an automated CI pipeline. Every commit triggers a workflow that runs tests, linting, and environment checks, ensuring code quality before deployment.

  * **Proof of CI Flow:** View the successful workflow runs on the GitHub Actions page:
      * [srvmann/yt-chrome-plugin GitHub Actions](https://www.google.com/search?q=https://github.com/srvmann/yt-chrome-plugin/actions)

-----

### 💻 Live Demonstration

See the YouTube Comment Analyser working live on a local system, demonstrating real-time comment fetching and sentiment classification.


https://github.com/srvmann/yt-chrome-plugin/blob/master/Recording%202025-10-29%20210828.mp4
-----

### ⚙️ Local Reproduction Guide: ML Pipeline & Containerised API

This guide details how to reproduce the project by first pulling the DVC-tracked model artefacts and then running the final application using the original container structure.

#### Prerequisites

1.  **Python 3.x**, **Git**, and **Docker** installed.
2.  **DVC** (Data Version Control) installed globally (`pip install dvc`).
3.  **YouTube Data API Key** (required for the backend to fetch comments).
4.  **AWS CLI** configured (required for DVC to pull remote model/data from S3).

#### Step 1: Clone the Backend & Setup Environment

1.  **Clone and Navigate:**
    ```bash
    git clone https://github.com/srvmann/yt-chrome-plugin.git
    cd yt-chrome-plugin
    ```
2.  **Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate # Use .\venv\Scripts\activate on Windows
    pip install -e .
    ```

#### Step 2: Reproduce Model Artefacts (DVC Pipeline)

The trained model and data dependencies are versioned using DVC. This step retrieves them, ensuring the API has the necessary files to run.

1.  **Retrieve Artefacts:** This pulls the trained model (the artefact) from the DVC remote storage.
    ```bash
    # Requires AWS credentials to be configured in your environment
    dvc pull
    ```
2.  **Run ML Pipeline:** (Optional - to re-train the model)
    ```bash
    dvc repro
    ```

#### Step 3: Run Containerised API (`app.py`)

1.  **Set Environment Variable:** Pass your required YouTube Data API Key.

    ```bash
    export YOUTUBE_API_KEY="YOUR_API_KEY_HERE" # Use 'set' or '$env:' on Windows
    ```

2.  **Run the Container:** The container executes the `flask_app/app.py` logic.

    ```bash
    # Note: Use your specific ECR URI
    docker run -d -p 5000:5000 \
    -e YOUTUBE_API_KEY="YOUR_API_KEY_HERE" \
    233803224854.dkr.ecr.ap-south-1.amazonaws.com/yt-chrome-plugin:latest
    ```

    The API is now running locally at **`http://localhost:5000`**.

-----

### 🌐 Frontend Setup & Usage

1.  **Clone the Frontend Repo:** [https://github.com/srvmann/chrome-plugin-frontend](https://github.com/srvmann/chrome-plugin-frontend)
2.  **Install the Extension:** Go to `chrome://extensions`, enable **Developer mode**, click **Load unpacked**, and select the cloned folder.
3.  **Test:** Open any YouTube video and click the extension icon to trigger the full analysis.

-----

### ☁️ AWS Deployment Status and Free Tier Limit

All development and MLOps steps were completed, successfully creating a production-ready container, but the final deployment was halted due to cost constraints.

  * **Registry Image:** The final container image is available on ECR: `233803224854.dkr.ecr.ap-south-1.amazonaws.com/yt-chrome-plugin:latest`.
  * **The Limitation:** Final **code deployment** to a persistent compute service (like an **AWS EC2 instance**) was stopped because running the server 24/7 immediately consumes the **AWS Free Tier 750 hours/month limit**, risking unbudgeted charges.

**Recommendation:** The most cost-effective path for cloud hosting is to deploy the container using a serverless model (e.g., **AWS Lambda with a Container Image**) to utilize pay-per-use billing.
