"""
tests/deep_learning/test_text_cnn.py

Tests unitaires pour src/deep_learning/text_cnn.py.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


from deep_learning.CNN import TextCNN
from deep_learning.text_embedding import (
    build_vocabulary,
    encode_texts,
    evaluate_classifier,
    train_classifier,
)


def test_forward_pass_produces_correct_output_shape():
    model = TextCNN(vocab_size=100, embed_dim=16, num_filters=8, kernel_sizes=(2, 3))
    x = torch.randint(1, 100, (5, 20))  # batch de 5 phrases, 20 tokens chacune
    output = model(x)
    assert output.shape == (5,)


def test_padding_vector_stays_zero_after_training():
    vocab = build_vocabulary(["good great excellent", "bad terrible awful"] * 20)
    X = encode_texts(
        ["good great excellent", "bad terrible awful"] * 20, vocab, max_len=10
    )
    y = torch.tensor(([1.0, 0.0] * 20))

    model = TextCNN(
        vocab_size=len(vocab), embed_dim=16, num_filters=8, kernel_sizes=(2,)
    )
    train_classifier(model, X, y, epochs=5)

    assert torch.allclose(model.embedding.weight[0], torch.zeros(16))


def test_order_sensitivity_different_outputs_for_reversed_sequence():
    # LA propriete qui distingue le CNN du classifieur moyenne : deux
    # sequences avec les memes tokens mais un ordre different doivent
    # produire des sorties DIFFERENTES (contrairement a la moyenne,
    # verifiee identique dans la conversation associee)
    torch.manual_seed(42)
    model = TextCNN(vocab_size=10, embed_dim=8, num_filters=4, kernel_sizes=(2,))

    sequence_a = torch.tensor([[1, 2, 3, 0, 0]])
    sequence_b = torch.tensor([[3, 2, 1, 0, 0]])

    with torch.no_grad():
        sortie_a = model(sequence_a)
        sortie_b = model(sequence_b)

    assert not torch.allclose(sortie_a, sortie_b)


def test_multiple_kernel_sizes_produce_correct_concatenated_dimension():
    # 3 tailles de noyau x 8 filtres chacune = 24 valeurs concatenees
    # avant la couche finale
    model = TextCNN(vocab_size=50, embed_dim=16, num_filters=8, kernel_sizes=(3, 4, 5))
    assert model.fc.in_features == 8 * 3


def test_classifier_learns_something_on_toy_data():
    vocab = build_vocabulary(["good great excellent", "bad terrible awful"] * 20)
    X = encode_texts(
        ["good great excellent", "bad terrible awful"] * 20, vocab, max_len=5
    )
    y = torch.tensor(([1.0, 0.0] * 20))

    model = TextCNN(
        vocab_size=len(vocab), embed_dim=16, num_filters=8, kernel_sizes=(2,)
    )
    train_classifier(model, X, y, epochs=30, lr=0.01)
    results = evaluate_classifier(model, X, y)

    assert results["accuracy"] > 0.8


def test_pretrained_weights_are_used_when_provided():
    vocab_size = 5
    matrice = torch.zeros(vocab_size, 4)
    matrice[2] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=4,
        num_filters=2,
        kernel_sizes=(2,),
        pretrained_weights=matrice,
    )
    assert torch.allclose(model.embedding.weight[2], torch.tensor([1.0, 2.0, 3.0, 4.0]))
