import io
import os
from fractions import Fraction

from tml.c07_support_vectors.support_vector_machines.sentiment_classifier import SentimentClassifier

file_abs_path = os.path.abspath(__file__)
parent_abs_path = os.path.abspath(os.path.join(file_abs_path, os.pardir))
parent_parent_abs_path = os.path.abspath(os.path.join(parent_abs_path, os.pardir))


def split_file(filepath):
    """Splits a file into training and validation sets."""
    ext = os.path.splitext(filepath)[1]
    counter = 0
    training_filename = parent_parent_abs_path + f'/tests/fixtures/training{ext}'
    validation_filename = parent_parent_abs_path + f'/tests/fixtures/validation{ext}'
    with io.open(filepath, errors='ignore') as input_file:
        with io.open(validation_filename, 'w') as val_file:
            with io.open(training_filename, 'w') as train_file:
                for line in input_file:
                    if counter % 2 == 0:
                        val_file.write(line)
                    else:
                        train_file.write(line)
                    counter += 1
    return {'training': training_filename, 'validation': validation_filename}


def validate(classifier, file, sentiment):
    """Validates the classifier."""
    total = 0
    misses = 0

    with io.open(file, errors='ignore') as f:
        for line in f:
            if classifier.classify(line) != sentiment:
                misses += 1
            total += 1
    return Fraction(misses, total)


def test_validate():
    """Cross validates with an error of 35% or less."""
    neg = split_file(parent_parent_abs_path + '/data/rt-polaritydata/rt-polarity.neg')
    pos = split_file(parent_parent_abs_path + '/data/rt-polaritydata/rt-polarity.pos')

    classifier = SentimentClassifier.build([neg['training'], pos['training']])
    classifier.c = 2 ** 7
    classifier.reset_model()

    n_er = validate(classifier, neg['validation'], 'negative')
    p_er = validate(classifier, pos['validation'], 'positive')
    total = Fraction(n_er.numerator + p_er.numerator, n_er.denominator + p_er.denominator)
    print(total)

    assert total < 0.35


def test_validate_itself():
    """Yields a zero error when it uses itself."""
    classifier = SentimentClassifier.build([
        parent_parent_abs_path + '/data/rt-polaritydata/rt-polarity.neg',
        parent_parent_abs_path + '/data/rt-polaritydata/rt-polarity.pos'
    ])
    classifier.c = 2 ** 7
    classifier.reset_model()

    n_er = validate(classifier, parent_parent_abs_path + '/data/rt-polaritydata/rt-polarity.neg', 'negative')
    p_er = validate(classifier, parent_parent_abs_path + '/data/rt-polaritydata/rt-polarity.pos', 'positive')
    total = Fraction(n_er.numerator + p_er.numerator, n_er.denominator + p_er.denominator)
    print(total)

    assert round(float(total), 3) == 0
