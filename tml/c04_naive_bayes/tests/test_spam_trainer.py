import io
import math

from tml.c04_naive_bayes.email_object import EmailObject
from tml.c04_naive_bayes.spam_trainer import SpamTrainer


def test_counts_all_at_zero():
    training = [['spam', 'fixtures/plain.eml'], ['ham', 'fixtures/small.eml'], ['scram', 'fixtures/plain.eml']]
    trainer = SpamTrainer(training)
    for cat in ['_all', 'spam', 'ham', 'scram']:
        assert trainer.total_for(cat) == 0


def test_multiple_categories():
    training = [['spam', 'fixtures/plain.eml'], ['ham', 'fixtures/small.eml'], ['scram', 'fixtures/plain.eml']]
    trainer = SpamTrainer(training)
    categories = trainer.categories
    expected = set([k for k, v in training])
    assert categories == expected


def test_preference_category():
    training = [['spam', 'fixtures/plain.eml'], ['ham', 'fixtures/small.eml'], ['scram', 'fixtures/plain.eml']]
    trainer = SpamTrainer(training)
    expected = sorted(trainer.categories, key=lambda cat: trainer.total_for(cat))
    assert trainer.preference() == expected


def test_probability_being_1_over_n():
    training = [['spam', 'fixtures/plain.eml'], ['ham', 'fixtures/small.eml'], ['spam', 'fixtures/plain.eml']]
    trainer = SpamTrainer(training)
    with io.open('fixtures/plain.eml', 'rb') as eml_file:
        email = EmailObject(eml_file)
    scores = list(trainer.score(email).values())
    for score in scores:
        assert math.isclose(score, 0.33, abs_tol=0.4)


def test_adds_up_to_one():
    training = [['spam', 'fixtures/plain.eml'], ['ham', 'fixtures/small.eml'], ['spam', 'fixtures/plain.eml']]
    trainer = SpamTrainer(training)
    with io.open('fixtures/plain.eml', 'rb') as eml_file:
        email = EmailObject(eml_file)
    scores = list(trainer.normalized_score(email).values())
    assert sum(scores) == 1.0


def test_give_preference_to_whatever_has_the_most():
    training = [['spam', 'fixtures/plain.eml'], ['ham', 'fixtures/small.eml'], ['spam', 'fixtures/plain.eml']]
    trainer = SpamTrainer(training)
    with io.open('fixtures/plain.eml', 'rb') as eml_file:
        email = EmailObject(eml_file)
    score = trainer.score(email)
    preference = trainer.preference()[-1]
    preference_score = score[preference]
    expected = SpamTrainer.Classification(preference, preference_score)
    assert trainer.classify(email) == expected
