from rital_nlp_project.common.metrics import DEFAULT_SCORING_NAMES, make_binary_cv_scorers
from rital_nlp_project.movies.models_config import task


scoring = make_binary_cv_scorers(task)
scoring_names = DEFAULT_SCORING_NAMES
