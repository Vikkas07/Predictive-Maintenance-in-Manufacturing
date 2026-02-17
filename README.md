Predictive Maintenance in Manufacturing using Machine Learning
This project implements a predictive maintenance system that estimates the probability of machine failure from industrial sensor measurements and generates a maintenance recommendation before breakdown occurs.

Instead of repairing machines after failure or servicing them at fixed intervals, the system monitors machine condition and schedules maintenance only when operational risk becomes significant. The goal is to reduce downtime, avoid unnecessary servicing, and support data-driven maintenance planning.

The project simulates an Industry 4.0 scenario where IoT sensors continuously record machine behavior and a trained model evaluates the health state of the equipment.

Project Motivation

Traditional maintenance strategies operate blindly:

Reactive maintenance repairs machines after failure, causing production downtime and high cost.

Preventive maintenance services machines on a fixed schedule, often replacing healthy components.

Predictive maintenance observes actual machine behavior and decides maintenance timing based on condition.
This project demonstrates how machine learning can learn complex stress patterns from sensor data and convert them into actionable maintenance decisions.

Dataset

The system uses the AI4I 2020 Predictive Maintenance dataset, a simulated industrial dataset designed to represent real equipment behavior.

Each record describes a machine’s operating state using measurements such as temperature, rotational speed, torque, and tool wear, along with a label indicating whether a failure occurred.

The dataset is imbalanced, meaning failures are rare events. This reflects real industrial conditions and requires careful model evaluation beyond simple accuracy.

Methodology

The project follows a complete machine learning pipeline.

First, exploratory analysis was performed to understand how mechanical load and temperature relate to failure.
Then feature engineering introduced an overheating indicator derived from the difference between process temperature and air temperature.

Multiple models were trained and compared:

A linear baseline model to observe general trends
A tree ensemble model to capture nonlinear patterns
A gradient boosting model for final optimization

The final system uses an XGBoost classifier because it achieved the best balance between detecting failures and avoiding excessive false alarms.

The model outputs failure probability rather than a simple yes/no prediction.
A decision threshold converts probability into an operational instruction.

When predicted risk exceeds 30%, maintenance is recommended.

System Behavior

The model discovered that machine failure is driven primarily by combined mechanical stress rather than a single threshold variable.

High rotational speed combined with torque and accumulated tool wear significantly increases failure probability.
Temperature alone is less informative than the interaction between thermal and mechanical stress.

This demonstrates why predictive analytics outperforms rule-based monitoring systems.

Example output from the prediction module:

Failure probability: 0.799
Decision: Maintenance Required

Repository Structure

The repository is organized to separate experimentation, trained artifacts, and executable system components.

The notebooks directory contains the full analysis pipeline including data understanding, feature engineering, model training, and visualization.
The src directory contains the prediction module that loads the trained model and generates maintenance recommendations.
The models directory stores the trained model object used during inference.
The reports directory contains generated visualizations and results used for evaluation.

Running the Project

Clone the repository and navigate into the project directory.

Install dependencies:

pip install -r requirements.txt

Run the prediction system:

python src/predict.py

The script will load the trained model and print the predicted failure probability along with a maintenance recommendation.

Results

The final model demonstrates strong ability to distinguish failing machines from healthy machines.
It detects most upcoming failures while keeping unnecessary maintenance alerts low.

The threshold-based decision system converts machine learning output into an operational policy rather than a theoretical metric.

Applications

This approach can be applied to production lines, CNC machining, industrial motors, turbines, and other equipment monitored through sensor data.
It is particularly useful in smart factories where continuous monitoring data is available.

Limitations and Future Work

The dataset is simulated and does not include real-time streaming data.
Future work may include integrating live sensor feeds, predicting remaining useful life, and deploying the system in a dashboard environment.

Author

Aanya Singh
Bachelor of Computer Applications
