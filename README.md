Modern manufacturing systems depend on machines operating continuously under mechanical load and thermal stress. Unexpected machine failure leads to production downtime, financial loss, and safety risks. Traditional maintenance strategies do not solve this efficiently. Reactive maintenance repairs equipment only after breakdown, which causes costly interruptions, while preventive maintenance services machines on a fixed schedule regardless of their actual condition, often replacing components that still function correctly.

This project implements a predictive maintenance system that analyzes industrial sensor data and forecasts machine failure before it occurs. Instead of reacting to failure or following a rigid schedule, maintenance decisions are based on the real health state of the machine inferred from data.

The system is built using machine learning and simulates an Industry 4.0 scenario in which IoT sensors continuously monitor machine behavior. The trained model estimates the probability of failure and triggers a maintenance recommendation when the risk becomes significant.

Dataset
The project uses the AI4I 2020 Predictive Maintenance dataset, a synthetic dataset designed to represent real industrial equipment behavior. It contains ten thousand observations of machine operating conditions. Each record describes the state of a machine using measurements such as temperature, rotational speed, torque, and tool wear, along with a label indicating whether a failure occurred.

Although synthetic, the dataset reflects realistic industrial relationships between thermal conditions, mechanical stress, and wear accumulation.

Approach
The project follows a full machine learning workflow rather than a single model experiment.

The first stage focuses on understanding the operational behavior of machines through exploratory analysis. During this phase, relationships between temperature, speed, and wear were studied to identify early indicators of malfunction.

In the second stage, feature engineering was performed. A derived overheating indicator was created by measuring the difference between internal process temperature and surrounding air temperature. This feature improved the model’s ability to recognize stress conditions inside the machine.

The third stage involved training multiple classification algorithms to predict machine failure. Logistic Regression was used as a baseline interpretable model, Random Forest provided a nonlinear ensemble reference, and XGBoost was selected as the final model due to its superior balance between detection reliability and false alarms.

The final stage converts prediction into a real maintenance decision. Instead of relying only on classification accuracy, a probability threshold was designed so that maintenance is recommended only when risk becomes operationally meaningful.

Results
The final XGBoost model achieved a high discrimination capability with strong ability to detect failures while maintaining very low false alarm rates. The model successfully identified most failing machines in advance and demonstrated that mechanical stress indicators such as rotational speed, torque, and tool wear were significantly more important than environmental temperature alone.

The system outputs a probability value representing machine risk. When the predicted risk exceeds thirty percent, the system recommends maintenance. This converts machine learning predictions into an actionable industrial policy rather than a theoretical metric.

An example prediction produced by the system:
Failure probability: 0.799
Decision: Maintenance Required

System Behavior
The project demonstrates that machine failure is rarely caused by a single variable exceeding a fixed threshold. Instead, breakdowns emerge from combined effects of overheating, mechanical load, and accumulated wear. The model learns these interactions automatically and detects failure patterns that traditional monitoring rules cannot capture.

By shifting maintenance from schedule-based to condition-based operation, downtime can be reduced and inspection resources can be used more efficiently.

Project Structure
The repository contains development notebooks used for data analysis and model training, a prediction module that produces maintenance decisions, stored trained models, and generated visualizations explaining model behavior.

Running the Project
Clone the repository and install the required dependencies listed in the requirements file. After installation, running the prediction script will load the trained model and output the failure probability along with a maintenance recommendation.

Applications
This approach applies to automated production lines, CNC machining systems, turbines, and any industrial environment where continuous monitoring data is available. The same framework can be integrated into smart factories to enable real-time maintenance planning.

Limitations and Future Work
The dataset is simulated and does not include true streaming sensor data. Future work may incorporate real-time IoT data ingestion, temporal deep learning models for remaining useful life prediction, and integration into industrial monitoring dashboards.

Author
Aanya Singh
Bachelor of Computer Applications
