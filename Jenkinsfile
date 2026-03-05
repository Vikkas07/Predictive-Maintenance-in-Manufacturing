pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/Vikkas07/Predictive-Maintenance-in-Manufacturing.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run Prediction Script') {
            steps {
                bat 'python src/predict.py'
            }
        }

    }
}
