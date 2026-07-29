"""
Disease Risk Prediction Core: Deep Learning Model and Prediction

@author: DanWu and Agreewithu (Ruixin Dai)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve

# --- 1. GAF Transfer ---
class GAFTransformer:
    """Convert Time Series Features into Two-dimensional GAF image features"""
    def __init__(self, min_size: int = 3):
        self.min_size = min_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.gaf = GramianAngularField(image_size=self.min_size, sample_range=(-1, 1))

    def transform(self, feature_list: list) -> list:
        gaf_feature = []
        for patient_records in feature_list:
            records_arr = np.array(patient_records).reshape(-1, 1).T
            # upper triangular matrix
            gaf_img = np.triu(self.gaf.transform(records_arr)[0]) 
            gaf_feature.append(gaf_img)
        return gaf_feature

# --- 2. Deep Learning Model for Disease Risk Prediction---
class TimeEmbedding(nn.Module):
    """Time Embedding"""
    def __init__(self, channels: int):
        super(TimeEmbedding, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = t
        out = self.conv(x)
        out = self.bn(out)
        out = torch.sigmoid(out)
        out = t * out
        out += residual
        return out


class Flatten(nn.Module):
    """Flatten for ChannelGateCBAM"""
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
    
class BasicConv(nn.Module):
    "Convolution for SpatialGateCBAM"
    def __init__(self, in_planes, out_planes, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, relu = True, bn = True, bias = False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)
        self.bn = nn.BatchNorm2d(out_planes, eps = 1e-5, momentum = 0.01, affine = True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

class ChannelGateCBAM(nn.Module):
    """Channel Process"""
    def __init__(self, gate_channels, reduction_ratio = 16, pool_types = ['avg', 'max']):
        super(ChannelGateCBAM, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels//reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_types in self.pool_types:
            if pool_types == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride = (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_types == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride = (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
            
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        scale = scale.expand_as(x)
        
        return x * scale


class ChannelPool(nn.Module):
    """Channel Pooling for CBAM"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGateCBAM(nn.Module):
    """Spatial Process"""
    def __init__(self):
        super(SpatialGateCBAM, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride = 1, padding = (kernel_size-1)//2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, gate_channels: int, reduction_ratio: int = 16, pool_types: list = ['avg', 'max'], no_spatial = False):
        super(CBAM, self).__init__()
        # Channel Attention and Spatial Attention
        self.ChannelGateCBAM = ChannelGateCBAM(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGateCBAM = SpatialGateCBAM()
    
    def forward(self, x):
        x_out = self.ChannelGateCBAM(x)
        if not self.no_spatial:
            x_out = self.SpatialGateCBAM(x_out)
        return x_out


class CBAMBottleneck(nn.Module):
    """
    Integrating TimeEmbedding and CBAM into the Residual Bottleneck Network Architecture
    Multi Feature Map Attention Model (MFMAM)
    """
    expansion = 4 # channels expansion
    
    def __init__(self, inplanes: int, planes: int, num_classes: int, kernel_size: int):
        super(CBAMBottleneck, self).__init__()
        self.te = TimeEmbedding(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(planes * self.expansion)
        self.conv4 = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(planes * self.expansion, num_classes, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.te(t, x) # time embedding
        # convolutions
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out) # attention
        
        # residual connection
        bo, co, ho, wo = out.size()
        _, cr, hr, _ = residual.size()
        if co != cr or ho != hr:
            residual = self.conv4(residual).view(bo, co, ho, wo)
        out += residual
        
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return torch.sigmoid(out)

# --- 3. Inference Interface ---
class DeepLearningRiskModel:
    """
    Disease Prediction
    """
    def __init__(self, target_disease_model_config: dict, model_weights_path: str, valid_y_true: np.ndarray, vaild_y_scores: np.ndarray, risk_label: str):
        # target disease
        self.target_disease = target_disease_model_config.get("Disease", "T2D")

        # model config depend on target disease
        model_in_channels = target_disease_model_config.get("In_Channels", 192)
        model_out_channels = target_disease_model_config.get("Out_Channels", 64)
        model_conv_kernel_size = target_disease_model_config.get("Kernel_Size", 3)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = CBAMBottleneck(inplanes=model_in_channels, planes=model_out_channels, num_classes=1, kernel_size=model_conv_kernel_size).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

        # threshold
        self.threshold = self._sel_threshold(valid_y_true, vaild_y_scores)

        self.risk_label = risk_label

    @staticmethod
    def _sel_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Youden Index for Threshold"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return thresholds[np.argmax(tpr - fpr)]

    def predict_risk(self, hers_processed_data: dict) -> tuple:
        """
        Model Inference
        """
        # patient HERs processed data to model input tensor
        delta_t_gaf_tensor = torch.tensor(GAFTransformer.transform(hers_processed_data.get("delta_t")))
        delta_t_gaf_tensor = delta_t_gaf_tensor.unsqueeze(dim=0) # add batch dimension

        feature_gaf_lists = []
        features_imputed, missing_indicators = hers_processed_data.get("features_imputed"), hers_processed_data.get("missing_indicators")
        for i in range(len(features_imputed)):
            feature_gaf_lists.append(GAFTransformer.transform(features_imputed[i])[0])
            feature_gaf_lists.append(GAFTransformer.transform(missing_indicators[i])[0])

        patient_gaf_tensor = torch.tensor(feature_gaf_lists)
        patient_gaf_tensor = patient_gaf_tensor.unsqueeze(dim=0) # add batch dimension for model

        with torch.inference_mode():
            output = self.model(patient_gaf_tensor.to(self.device), delta_t_gaf_tensor.to(self.device)).to(torch.float32)
            risk_score = output.item()
            
            # score v.s. threshold
            risk_level = self.risk_label if risk_score > self.threshold else "Low Risk" 
            
        return risk_score, risk_level