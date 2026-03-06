# Predictive Maintenance in Manufacturing using Machine Learning

This project implements a predictive maintenance system that estimates the probability of machine failure from industrial sensor measurements and generates a maintenance recommendation before a breakdown occurs.

Instead of repairing machines after failure or servicing them at fixed intervals, the system monitors machine condition and schedules maintenance only when operational risk becomes significant. The objective is to reduce unexpected downtime, avoid unnecessary servicing, and support data-driven maintenance planning.

The project simulates an Industry 4.0 environment where industrial equipment is monitored through sensor measurements and a machine learning model continuously evaluates the health state of the machine.

---

# Project Motivation

Traditional maintenance strategies operate with limited information.

Reactive maintenance repairs machines only after they fail, which can halt production and cause expensive emergency repairs. Preventive maintenance improves reliability by servicing machines on a fixed schedule, but it often replaces components that are still functioning normally.

Predictive maintenance takes a different approach. Instead of relying on time-based schedules, it analyzes the actual operating condition of machines and predicts when failure is likely to occur. Maintenance can then be scheduled only when risk becomes significant.

This project demonstrates how machine learning can learn patterns of mechanical stress from sensor data and convert those patterns into actionable maintenance decisions.

---

# Dataset

The system uses the **AI4I 2020 Predictive Maintenance dataset**, a simulated industrial dataset designed to represent realistic machine behavior.

Each record represents the operating state of a machine and includes sensor measurements such as air temperature, process temperature, rotational speed, torque, and accumulated tool wear. The dataset also includes labels indicating whether a machine failure occurred.

Failures are rare events in the dataset, reflecting real industrial environments where machines operate normally most of the time but occasionally fail under high mechanical or thermal stress. Because of this imbalance, evaluating the model requires more than simple accuracy metrics.

---

# Machine Learning Approach

The project follows a full machine learning workflow beginning with exploratory analysis and ending with a deployable prediction system.

Initial data exploration was used to understand how temperature, mechanical load, and wear interact during failure events. Feature engineering then introduced an additional variable representing the difference between process temperature and air temperature. This feature acts as an indicator of overheating conditions that can accelerate machine degradation.

Multiple models were trained and compared during experimentation, including a linear baseline model, a tree-based ensemble model, and a gradient boosting model. The final system uses **XGBoost**, which provided the best balance between detecting true failures and minimizing unnecessary maintenance alerts.

Rather than returning a simple binary prediction, the model outputs a **failure probability**. This probability is converted into an operational decision using a threshold policy. When predicted failure risk exceeds 30%, the system recommends maintenance.

---

# System Architecture

The project evolved from a machine-learning experiment into a small, deployable analytics system.

The trained model processes sensor measurements to estimate the probability of failure. The prediction engine loads the saved model and generates maintenance recommendations based on the risk threshold.

A **Streamlit dashboard** provides an interactive interface for exploring the system. Users can enter machine operating conditions and instantly see the predicted failure probability along with maintenance guidance.

The system architecture can be viewed as a simple pipeline:

Machine sensor data → trained ML model → failure probability → maintenance decision → dashboard interface

---

# Streamlit Dashboard

To make the model easier to interact with, a web dashboard was built using Streamlit.

The dashboard presents the predictive maintenance system in four sections. The overview page introduces the concept of predictive maintenance and explains how the system works. The live prediction page allows users to input sensor values and obtain real-time failure predictions from the trained model. The risk analysis page visualizes the factors contributing to machine stress and highlights conditions that increase failure probability. The maintenance planner provides simple recommendations based on predicted risk levels.

This interface transforms the machine learning model from an offline experiment into a decision-support tool that can be explored interactively.

---

# Continuous Integration with Jenkins

The project also includes a Jenkins pipeline that demonstrates how the system could be integrated into a continuous integration workflow.

The pipeline automatically retrieves the latest code from the repository, installs project dependencies, and prepares the environment required to run the system. While the pipeline is simple, it reflects how machine learning systems are often integrated into DevOps pipelines to automate testing, updates, and deployment steps.

The pipeline configuration is defined in the **Jenkinsfile** located in the repository.

---

# Repository Structure

The repository is organized to separate experimentation, trained artifacts, and application components.

The `notebooks` directory contains the full data science workflow, including data understanding, feature engineering, model training, and visualization. The `models` directory stores the trained model used for prediction. The `src` directory contains utility modules that load the model and perform inference.

The Streamlit dashboard is implemented through the main `app.py` file and the pages stored in the `pages` directory. Configuration settings for the dashboard are located in the `.streamlit` folder.

Supporting datasets are stored in the `data` directory, while generated visualizations and analysis outputs are stored in `reports`.

---

# Running the Project

Clone the repository and navigate to the project directory.

Install the required Python dependencies:

pip install -r requirements.txt

Once the environment is ready, launch the dashboard:

streamlit run app.py

The application will open in a browser window where you can explore machine failure predictions interactively.

---

# Results

The trained model demonstrates a strong ability to distinguish between healthy machines and machines approaching failure conditions. Instead of relying on simple threshold rules, the model captures nonlinear relationships between torque, rotational speed, temperature, and tool wear.

This allows the system to detect risky operating conditions earlier and generate maintenance recommendations before failure occurs.

---

# Applications

Predictive maintenance systems like this can be applied to manufacturing equipment such as CNC machines, industrial motors, turbines, and other machinery monitored through sensor data. In smart factories where machines generate continuous operational data, systems like this can significantly reduce downtime and maintenance costs.

---

# Limitations and Future Work

The dataset used in this project is simulated and does not represent real-time sensor streams. Future improvements could include integrating live IoT data, predicting the remaining useful life of components, and deploying the system on a cloud platform for continuous monitoring.

---

# Authors

Aanya Singh
Vikas Kumar Singh
Nikita Shelke

Bachelor of Computer Applications
