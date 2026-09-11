import torch

from classification.model import SkinDiseaseClassifier

model = SkinDiseaseClassifier()

x = torch.randn(4,3,224,224)

y = model(x)

print(y.shape)