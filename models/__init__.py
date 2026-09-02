from .physics_loss import physics_consistency_loss, mse_only, physics_penalty_only
from .micro_resnet import build_micro_resnet
from .mlp_branch import build_mlp_branch
from .fusion_model import build_fusion_model

CUSTOM_OBJECTS = {
    'physics_consistency_loss': physics_consistency_loss,
    'mse_only': mse_only,
    'physics_penalty_only': physics_penalty_only,
}
