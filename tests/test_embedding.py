"""The gene vocabulary as a module: `embedding(g) = W . prior(g) + delta(g)`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vccp.models.embedding import FEATURE, TARGET, GeneVocabulary, build_vocabulary


@pytest.fixture
def toy_priors():
    rng = np.random.default_rng(0)
    return {
        FEATURE: rng.standard_normal((11, 6)).astype(np.float32),
        TARGET: rng.standard_normal((11, 4)).astype(np.float32),
    }


def test_embedding_is_W_prior_plus_delta_and_delta_starts_zero(toy_priors):
    vocabulary = GeneVocabulary(toy_priors, embedding_dim=5)

    for role in (FEATURE, TARGET):
        # delta is zero at initialization, so the embedding is exactly the
        # projected prior: a gene never seen in training still has one.
        assert torch.equal(vocabulary.delta(role), torch.zeros(11, 5))

        embedded = vocabulary(role=role)
        expected = vocabulary.projections[role](vocabulary.prior(role))
        assert torch.allclose(embedded, expected, atol=1e-6)

    # Move delta and the embedding moves by exactly that much.
    with torch.no_grad():
        vocabulary.deltas[FEATURE][3] += 2.0
    before = vocabulary.projections[FEATURE](vocabulary.prior(FEATURE))[3]
    assert torch.allclose(vocabulary(role=FEATURE)[3], before + 2.0, atol=1e-6)


def test_a_gene_subset_is_embedded_consistently(toy_priors):
    vocabulary = GeneVocabulary(toy_priors, embedding_dim=5)
    with torch.no_grad():
        vocabulary.deltas[FEATURE].normal_()

    everything = vocabulary(role=FEATURE)
    subset = vocabulary(torch.tensor([7, 1, 1, 4]), role=FEATURE)
    assert torch.allclose(subset, everything[[7, 1, 1, 4]], atol=1e-6)


def test_the_two_roles_are_separate_vocabularies(toy_priors):
    vocabulary = GeneVocabulary(toy_priors, embedding_dim=5)
    assert vocabulary.prior_widths[FEATURE] != vocabulary.prior_widths[TARGET]

    with torch.no_grad():
        vocabulary.deltas[FEATURE].fill_(1.0)
    assert torch.equal(vocabulary.delta(TARGET), torch.zeros(11, 5))


def test_delta_penalty_is_zero_at_initialization_and_grows(toy_priors):
    vocabulary = GeneVocabulary(toy_priors, embedding_dim=5, delta_l2=0.5)
    assert vocabulary.delta_penalty().detach().item() == 0.0

    with torch.no_grad():
        vocabulary.deltas[FEATURE].fill_(1.0)
    # 11 genes x 5 dims x 1.0^2 x 0.5
    assert vocabulary.delta_penalty().detach().item() == pytest.approx(0.5 * 11 * 5)


def test_delta_penalty_is_off_when_not_configured(toy_priors):
    vocabulary = GeneVocabulary(toy_priors, embedding_dim=5, delta_l2=0.0)
    with torch.no_grad():
        vocabulary.deltas[FEATURE].fill_(3.0)
    assert vocabulary.delta_penalty().detach().item() == 0.0


def test_gradients_reach_both_W_and_delta(toy_priors):
    vocabulary = GeneVocabulary(toy_priors, embedding_dim=5, delta_l2=1e-3)
    loss = vocabulary(torch.tensor([0, 2, 5]), role=TARGET).pow(2).sum()
    loss = loss + vocabulary.delta_penalty()
    loss.backward()

    assert vocabulary.projections[TARGET].weight.grad is not None
    assert vocabulary.projections[TARGET].weight.grad.abs().sum() > 0
    assert vocabulary.deltas[TARGET].grad is not None
    assert vocabulary.deltas[TARGET].grad.abs().sum() > 0


def test_unknown_role_is_rejected(toy_priors):
    vocabulary = GeneVocabulary(toy_priors, embedding_dim=5)
    with pytest.raises(KeyError, match="unknown role"):
        vocabulary(role="perturbation")


def test_disagreeing_gene_counts_are_rejected():
    with pytest.raises(ValueError, match="gene count"):
        GeneVocabulary(
            {FEATURE: np.zeros((4, 3), np.float32), TARGET: np.zeros((5, 3), np.float32)},
            embedding_dim=2,
        )


def test_built_from_the_real_priors(session_cfg, built_priors):
    """The widths come from the data, not from a constant."""
    vocabulary = build_vocabulary(
        built_priors,
        session_cfg.model.embedding_dim,
        delta_l2=session_cfg.model.delta_l2,
    )

    assert vocabulary.n_genes == built_priors.n_genes
    for role in (FEATURE, TARGET):
        assert vocabulary.prior_widths[role] == built_priors.width(role)
        embedded = vocabulary(role=role)
        assert embedded.shape == (built_priors.n_genes, session_cfg.model.embedding_dim)
        assert torch.isfinite(embedded).all()
