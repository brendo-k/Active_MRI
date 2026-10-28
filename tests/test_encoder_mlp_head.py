import torch

from activemri.samplers.conv import SampProb2D
from activemri.training.trainer import HEAD_MLP_PARAMETERS

SHAPE = (32, 32)
DEPTH = 2


def build(head, seed=0):
    torch.manual_seed(seed)
    return SampProb2D(shape=SHAPE, chans=8, depth=DEPTH, head=head)


def observation(batch, seed=1):
    generator = torch.Generator().manual_seed(seed)
    kspace = torch.randn(batch, *SHAPE, 4, generator=generator)
    mask = torch.zeros(batch, 1, *SHAPE)
    mask[..., 14:18, 14:18] = 1
    return kspace, mask


def unet_input(kspace, mask):
    return torch.cat([kspace.permute(0, 3, 1, 2), mask], dim=1)


def randomise_output_layer(net):
    with torch.no_grad():
        net.encoder_head.mlp[-1].weight.normal_(std=0.1)


def test_shares_every_other_weight_and_the_rng_with_the_plain_net():
    torch.manual_seed(0)
    plain = SampProb2D(shape=SHAPE, chans=8, depth=DEPTH, head='none')
    after_plain = torch.rand(3)
    torch.manual_seed(0)
    headed = SampProb2D(shape=SHAPE, chans=8, depth=DEPTH, head='encoder_mlp')
    after_headed = torch.rand(3)
    assert torch.equal(after_plain, after_headed)
    headed_state = headed.state_dict()
    for name, value in plain.state_dict().items():
        assert torch.equal(value, headed_state[name]), name


def test_starts_as_the_plain_net():
    plain, headed = build('none'), build('encoder_mlp')
    kspace, mask = observation(4)
    for mode in (True, False):
        plain.train(mode), headed.train(mode)
        assert torch.allclose(plain(kspace, mask)[0], headed(kspace, mask)[0], atol=1e-6)


def test_term_has_zero_batch_mean_in_training():
    headed = build('encoder_mlp').train()
    randomise_output_layer(headed)
    kspace, mask = observation(6)
    _, _, bottleneck = headed._forward_encoder(unet_input(kspace, mask))
    raw = headed.encoder_head.mlp(bottleneck)
    assert raw.std(dim=0).mean() > 1e-3
    assert raw.mean(dim=0).abs().max() < 1e-4


def test_term_differs_between_inputs_and_is_bounded():
    headed = build('encoder_mlp').eval()
    randomise_output_layer(headed)
    with torch.no_grad():
        headed.encoder_head.mlp[-1].weight.mul_(1e4)
    kspace, mask = observation(3)
    _, _, bottleneck = headed._forward_encoder(unet_input(kspace, mask))
    term = headed.encoder_head(bottleneck)
    assert term.abs().max() <= headed.logit_limit
    assert (term[0] - term[1]).abs().max() > 1e-3


def test_gradient_reaches_the_head_from_zero_init():
    headed = build('encoder_mlp').train()
    kspace, mask = observation(4)
    prob, _, _ = headed(kspace, mask)
    (prob * torch.randn_like(prob)).sum().backward()
    assert headed.encoder_head.mlp[-1].weight.grad.abs().sum() > 0


def test_head_parameters_get_the_head_lr_group():
    headed = build('encoder_mlp')
    names = [name for name, _ in headed.named_parameters() if any(key in f'.{name}' for key in HEAD_MLP_PARAMETERS)]
    assert names and all(name.startswith('encoder_head.') for name in names)
    assert not any(key in f'.{name}' for name, _ in build('none').named_parameters() for key in HEAD_MLP_PARAMETERS)
