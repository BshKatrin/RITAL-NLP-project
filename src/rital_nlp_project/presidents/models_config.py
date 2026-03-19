from rital_nlp_project.common.metrics import BinaryTaskSpec


task = BinaryTaskSpec(
    name="presidents",
    negative_label=-1,
    positive_label=1,
    cv_n_splits=5,
    random_state=42,
    test_size=0.2,
    sequential=True,
)

cv_n_splits = task.cv_n_splits
random_state = task.random_state
test_size = task.test_size
