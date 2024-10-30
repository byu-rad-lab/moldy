from moldy.validation.prediction.prediction_validation import PredictionValidation

from moldy.case_studies.inverted_pendulum.model_ip import InvertedPendulum
from moldy.case_studies.inverted_pendulum.learnedModel_ip import LearnedModel_InvertedPendulum

tester = PredictionValidation(LearnedModel_InvertedPendulum, 500, InvertedPendulum())
result = tester.run_trials("/home/daniel/catkin_ws/src/moldy/case_studies/inverted_pendulum/results/best_models/",
                            "/home/daniel/catkin_ws/src/moldy/case_studies/inverted_pendulum/results/test_results/")
