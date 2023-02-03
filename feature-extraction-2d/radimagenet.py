import torch
import torchvision
from fire import Fire
from extract_rad import extract_features_
from getcuda import get_free_gpu_indices

__all__ = ['extract_resnet50_radimagenet_features_']


def extract_resnet50_radimagenet_features_(*slide_tile_paths, **kwargs):
    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(torch.load('marugoto/extract/RadImageNet-ResNet50_notop.pth'))
    model.fc = torch.nn.Identity()
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_ids = get_free_gpu_indices()
    if len(device_ids)==0:
        device = 'cpu'
    else:
        device='cuda:'+str(device_ids[0])

    print(device)
    model = model.eval().to(device)

    return extract_features_(slide_tile_paths=slide_tile_paths, **kwargs, model=model, model_name='resnet50-RADimagenet')


if __name__ == '__main__':
    Fire(extract_resnet50_radimagenet_features_)
