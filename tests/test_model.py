"""The shared gene-token core and its adapters (checklist items 5 and 6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vccp.models.adapters import LoRALinear, adapter_names, iter_lora
from vccp.models.checkpoint import load_core, save_core
from vccp.models.core import as_index, build_model, zscore_across_genes
from vccp.priors.build import FEATURE, TARGET


@pytest.fixture(scope="module")
def model_and_cfg(session_cfg, built_priors):
    return build_model(session_cfg, built_priors), session_cfg


def profile(batch, n_genes, channels=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_genes, channels, generator=generator)


def wake_heads(model, scale=0.1):
    """Give the output heads non-zero weights.

    Heads are zero-initialized on purpose, so an untrained model predicts
    exactly no change (§4.7). Anything testing that an *input* changes the
    *output* has to move them off zero first, or it is only measuring that
    zero equals zero.
    """
    with torch.no_grad():
        for head in model.heads.values():
            head[-1].base.weight.normal_(0, scale)


# --------------------------------------------------------------------------- #
# item 5 — one core, any gene subset
# --------------------------------------------------------------------------- #
def test_same_core_accepts_any_input_and_output_gene_subset(model_and_cfg):
    """The three phases feed different gene sets through the same network:
    LINCS the panel, a Replogle screen a few thousand, the challenge all."""
    model, _ = model_and_cfg
    n_genes = model.vocabulary.n_genes
    rng = np.random.default_rng(0)

    for n_in, n_out in [(50, 10), (200, 400), (n_genes, 32)]:
        gene_in = as_index(np.sort(rng.choice(n_genes, n_in, replace=False)))
        gene_out = as_index(np.sort(rng.choice(n_genes, n_out, replace=False)))
        out = model(profile(3, n_in), gene_in, gene_out)
        assert out.shape == (3, n_out)
        assert torch.isfinite(out).all()


def test_output_genes_need_not_be_input_genes(model_and_cfg):
    """Phase 2's whole job is panel -> rest: the output genes are ones the
    model was never given a value for."""
    model, _ = model_and_cfg
    gene_in = as_index(np.arange(0, 100))
    gene_out = as_index(np.arange(500, 560))
    out = model(profile(2, 100), gene_in, gene_out)
    assert out.shape == (2, 60)
    assert torch.isfinite(out).all()


def test_decoding_is_chunked_but_chunk_independent(session_cfg, built_priors):
    """Chunking is what keeps decoder memory flat in the number of output
    genes; it must not change the answer."""
    import dataclasses

    gene_out = as_index(np.arange(300))
    gene_in = as_index(np.arange(100))
    values = profile(2, 100)

    outputs = []
    for chunk in (4096, 37):
        cfg = dataclasses.replace(
            session_cfg, model=dataclasses.replace(session_cfg.model, gene_chunk=chunk)
        )
        model = build_model(cfg, built_priors)
        torch.manual_seed(0)
        model.eval()
        with torch.no_grad():
            outputs.append(model(values, gene_in, gene_out))
    # Different models, so compare shapes; the same model is compared below.
    assert outputs[0].shape == outputs[1].shape


def test_chunking_does_not_change_one_models_answer(session_cfg, built_priors):
    model = build_model(session_cfg, built_priors)
    wake_heads(model)
    gene_in, gene_out = as_index(np.arange(80)), as_index(np.arange(250))
    values = profile(2, 80)

    model.eval()
    with torch.no_grad():
        model.gene_chunk = 4096
        whole = model(values, gene_in, gene_out)
        model.gene_chunk = 17
        chunked = model(values, gene_in, gene_out)
        model.gene_chunk = 4096
    assert torch.allclose(whole, chunked, atol=1e-5)


def test_an_untrained_model_predicts_exactly_no_change(model_and_cfg):
    """Every head predicts a change, so zero is the honest starting point —
    and it is what the challenge's metrics reward not knowing (§4.7)."""
    model, _ = model_and_cfg
    gene_in, gene_out = as_index(np.arange(30)), as_index(np.arange(30))
    model.eval()
    with torch.no_grad():
        out = model(profile(2, 30), gene_in, gene_out)
    assert torch.equal(out, torch.zeros_like(out))


def test_the_perturbation_token_changes_the_answer(session_cfg, built_priors):
    """Conditioning on the perturbation must actually condition."""
    model = build_model(session_cfg, built_priors)
    wake_heads(model)
    gene_in, gene_out = as_index(np.arange(60)), as_index(np.arange(60))
    values = profile(2, 60)
    assay = session_cfg.phase2.pert_type

    model.eval()
    with torch.no_grad():
        unperturbed = model(values, gene_in, gene_out)
        a = model(values, gene_in, gene_out, target_idx=as_index([3, 3]), pert_type=assay)
        b = model(values, gene_in, gene_out, target_idx=as_index([900, 900]), pert_type=assay)

    assert not torch.allclose(unperturbed, a, atol=1e-6)
    assert not torch.allclose(a, b, atol=1e-6), "two targets must not look identical"


def test_the_context_vector_changes_the_answer(session_cfg, built_priors):
    model = build_model(session_cfg, built_priors)
    wake_heads(model)
    gene_in, gene_out = as_index(np.arange(60)), as_index(np.arange(60))
    values = profile(2, 60)

    # FiLM also starts as the identity, and its linear is adapter-wrapped,
    # so the base weight is what has to move.
    with torch.no_grad():
        model.film.to_params.base.weight.normal_(0, 0.1)

    model.eval()
    with torch.no_grad():
        a = model(values, gene_in, gene_out, context_profile=torch.zeros(2, 60, 1))
        b = model(values, gene_in, gene_out, context_profile=torch.ones(2, 60, 1))
    assert not torch.allclose(a, b, atol=1e-6)


def test_heads_are_separate_outputs(session_cfg, built_priors):
    model = build_model(session_cfg, built_priors)
    wake_heads(model)
    gene_in, gene_out = as_index(np.arange(40)), as_index(np.arange(40))
    values = profile(2, 40)

    model.eval()
    with torch.no_grad():
        sig = model(values, gene_in, gene_out, head="sig")
        value = model(values, gene_in, gene_out, head="value")
    assert not torch.allclose(sig, value, atol=1e-6)

    with pytest.raises(KeyError, match="unknown head"):
        model(values, gene_in, gene_out, head="nope")


def test_both_gene_roles_are_used(model_and_cfg):
    """Input tokens use the feature role, the perturbation token the target
    role (CLAUDE.md §4.1)."""
    model, cfg = model_and_cfg
    gene_in = as_index(np.arange(20))
    tokens = model.tokenize(
        profile(1, 20), gene_in, target_idx=as_index([7]), pert_type=cfg.phase2.pert_type
    )

    assert tokens.shape == (1, 21, model.dim)
    expected = model.vocabulary(as_index([7]), role=TARGET) + model.perturbation_type(
        cfg.phase2.pert_type
    )
    assert torch.allclose(tokens[:, -1], expected, atol=1e-5)
    feature = model.vocabulary(gene_in, role=FEATURE)
    assert torch.allclose(tokens[0, :20] - model.value_encoder(profile(1, 20))[0], feature, atol=1e-5)


# --------------------------------------------------------------------------- #
# item 6 — adapters
# --------------------------------------------------------------------------- #
def test_every_core_linear_is_adapter_capable(model_and_cfg):
    model, _ = model_and_cfg
    assert model.n_adapted_linears > 0
    assert len(list(iter_lora(model))) == model.n_adapted_linears


def test_an_adapter_is_the_identity_until_it_is_trained(session_cfg, built_priors):
    """Adding an adapter must not change what the model does."""
    model = build_model(session_cfg, built_priors)
    gene_in, gene_out = as_index(np.arange(40)), as_index(np.arange(40))
    values = profile(2, 40)

    model.eval()
    with torch.no_grad():
        before = model(values, gene_in, gene_out)
        model.add_adapter("phase2")
        model.use_adapters(["phase2"])
        after = model(values, gene_in, gene_out)
    assert torch.allclose(before, after, atol=1e-6)


def test_a_trained_adapter_does_change_the_model(session_cfg, built_priors):
    model = build_model(session_cfg, built_priors)
    wake_heads(model)
    model.add_adapter("phase2")
    model.use_adapters(["phase2"])
    gene_in, gene_out = as_index(np.arange(40)), as_index(np.arange(40))
    values = profile(2, 40)

    model.eval()
    with torch.no_grad():
        before = model(values, gene_in, gene_out)
        for parameter in model.adapter_parameters(["phase2"]):
            parameter.normal_(0, 0.1)
        after = model(values, gene_in, gene_out)
    assert not torch.allclose(before, after, atol=1e-6)


def test_adapters_are_switchable_and_independent(session_cfg, built_priors):
    """A later phase's adapter must not leak into an earlier phase's."""
    model = build_model(session_cfg, built_priors)
    wake_heads(model)
    for name in ("phase1", "phase2"):
        model.add_adapter(name)
    assert adapter_names(model._core_modules()) == ["phase1", "phase2"]

    with torch.no_grad():
        for parameter in model.adapter_parameters(["phase2"]):
            parameter.normal_(0, 0.1)

    gene_in, gene_out = as_index(np.arange(40)), as_index(np.arange(40))
    values = profile(2, 40)
    model.eval()
    with torch.no_grad():
        model.use_adapters(["phase1"])
        with_phase1 = model(values, gene_in, gene_out)
        model.use_adapters([])
        with_none = model(values, gene_in, gene_out)
        model.use_adapters(["phase2"])
        with_phase2 = model(values, gene_in, gene_out)

    assert torch.allclose(with_phase1, with_none, atol=1e-6), "phase1 is untrained"
    assert not torch.allclose(with_phase2, with_none, atol=1e-6)


def test_adapter_state_is_per_model_not_global(session_cfg, built_priors):
    a = build_model(session_cfg, built_priors)
    b = build_model(session_cfg, built_priors)
    a.add_adapter("x")
    b.add_adapter("x")
    a.use_adapters(["x"])
    b.use_adapters([])

    assert all(layer._active == ("x",) for layer in iter_lora(a))
    assert all(layer._active == () for layer in iter_lora(b))


def test_lora_linear_starts_as_its_base():
    base = torch.nn.Linear(6, 4)
    layer = LoRALinear(base, rank=2)
    layer.add_adapter("a")
    layer.set_active(["a"])
    x = torch.randn(3, 6)
    assert torch.allclose(layer(x), base(x), atol=1e-7)


# --------------------------------------------------------------------------- #
# checkpoints carry the prior layout they were trained against
# --------------------------------------------------------------------------- #
def test_three_checkpoints_share_core_parameter_names(session_cfg, built_priors, tmp_path):
    """The phases save the same core, not three different networks."""
    model = build_model(session_cfg, built_priors)
    for adapter in ("phase1", "phase2", "phase3"):
        model.add_adapter(adapter)

    names = []
    for phase in ("phase1", "phase2", "phase3"):
        path = tmp_path / f"core_{phase}.pt"
        save_core(path, model, phase=phase, layout_hash=built_priors.layout_hash())
        payload = torch.load(path, map_location="cpu", weights_only=False)
        names.append(sorted(payload["state_dict"]))
    assert names[0] == names[1] == names[2]


def test_a_checkpoint_refuses_a_different_prior_layout(session_cfg, built_priors, tmp_path):
    """The block set is not fixed — the server's third screen adds blocks and
    shifts every offset, so `W` would read the wrong columns."""
    model = build_model(session_cfg, built_priors)
    path = tmp_path / "core.pt"
    save_core(path, model, phase="phase1", layout_hash=built_priors.layout_hash())

    load_core(path, model, layout_hash=built_priors.layout_hash())
    with pytest.raises(ValueError, match="layout"):
        load_core(path, model, layout_hash="0" * 16)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_zscore_across_genes_describes_a_profiles_shape():
    """In control-SD units a control mean is zero by construction, so the
    context vector has to come from the profile's shape."""
    profile_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scaled = zscore_across_genes(profile_values * 10.0)
    assert np.allclose(zscore_across_genes(profile_values), scaled, atol=1e-5)
    assert abs(float(scaled.mean())) < 1e-6
    assert np.isfinite(zscore_across_genes(np.zeros(5, dtype=np.float32))).all()


# --------------------------------------------------------------------------- #
# adapters are created where the model already is
# --------------------------------------------------------------------------- #
# Every stage moves the model to the device and *then* adds its adapters —
# that is what an adapter is for. A parameter created on the default device
# instead of the base layer's would sit on the CPU while the rest of the model
# is on the GPU, and the first forward pass would die with "mat2 is on cpu".
# `meta` is a real device that exists without a GPU, so these run anywhere.
META = "meta"


def test_an_adapter_lands_on_the_device_its_base_layer_is_on():
    import torch.nn as nn

    from vccp.models.adapters import LoRALinear

    layer = LoRALinear(nn.Linear(4, 3), rank=2).to(META)
    layer.add_adapter("phase1")

    for parameter in layer.parameters():
        assert parameter.device.type == META, "an adapter was created off-device"


def test_an_adapter_inherits_its_base_layer_s_dtype():
    import torch.nn as nn

    from vccp.models.adapters import LoRALinear

    layer = LoRALinear(nn.Linear(4, 3), rank=2).to(torch.float64)
    layer.add_adapter("phase1")

    assert all(p.dtype == torch.float64 for p in layer.parameters())


def test_the_whole_model_stays_on_one_device_when_adapters_are_added(
    mini_cfg, built_priors
):
    """The guard that covers any parameter created after the move, not only
    the adapters'."""
    model = build_model(mini_cfg, built_priors).to(META)
    for name in ("phase1", "phase2", "phase3", "context:A"):
        model.add_adapter(name)

    devices = {p.device.type for p in model.parameters()}
    devices |= {b.device.type for b in model.buffers()}
    assert devices == {META}, f"the model is split across devices: {devices}"
