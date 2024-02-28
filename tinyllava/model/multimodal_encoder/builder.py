import os
from tinyllava.model.multimodal_encoder.clip.clip_encoder import CLIPVisionTower
from tinyllava.model.multimodal_encoder.siglip.siglip_encoder import SigLipVisionTower
from tinyllava.model.multimodal_encoder.eva_clip.eva_clip_encoder import EvaClipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
        
    if 'sig' in vision_tower.lower():
        if is_absolute_path_exists or vision_tower.startswith("google") or vision_tower.startswith('bczhou'):
            return SigLipVisionTower(vision_tower, vision_tower_cfg, **kwargs)
        
    elif 'eva' in vision_tower.lower():
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    elif 'clip' in vision_tower.lower():
         if is_absolute_path_exists or vision_tower.startswith('openai') or vision_tower.startswith('laion'):
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        
    raise ValueError(f'Unknown vision tower: {vision_tower}')
