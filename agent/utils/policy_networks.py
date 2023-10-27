import torch

def PolicyNetworkVanillaReLU():
    """
        returns a default policy network from SB3 but with ReLU activation functions

        :return: policy_kwargs required for RL models like PPO
    """
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        )

    return policy_kwargs

##################
# Other examples #
##################
# Note: when using images as state representation, the default PPO architecture of 64 neurons per layer may be too small to solve the task

# net_arch=[128, 128, dict(pi=[64,64], vf=[64,64])],

# policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])],
#                     features_extractor_class=your_pytorch_network_class_here,
#                     features_extractor_kwargs=give_your_kwargs_when_your_pytorch_network_class_is_instantiated)
