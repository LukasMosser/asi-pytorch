import torch
import torch.nn as nn
import torch.nn.functional as F

class MalenovNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 50, kernel_size=5, stride=4, padding=2)
        self.bn1 = nn.BatchNorm3d(50)
        
        self.conv2 = nn.Conv3d(50, 50, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(50)        
        self.drop2 = nn.Dropout3d(0.2)
        
        self.conv3 = nn.Conv3d(50, 50, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(50)  
        self.drop3 = nn.Dropout3d(0.2)
            
        self.conv4 = nn.Conv3d(50, 50, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(50)          
        self.drop4 = nn.Dropout3d(0.2)
        
        self.conv5 = nn.Conv3d(50, 50, kernel_size=3, stride=2, padding=1)  
        
        self.linear1 = nn.Linear(400, 50)
        self.bn_ln1 = nn.BatchNorm1d(50)
        self.linear2 = nn.Linear(50, 10)
        self.bn_ln2 = nn.BatchNorm1d(10)
        
        self.logits = nn.Linear(10, 9)
        self.bn_logits = nn.BatchNorm1d(9)
        
        """
        Keras model definition from MalenoV
        model = Sequential()
        model.add(Conv3D(50, (5, 5, 5), padding='same', input_shape=(cube_size,cube_size,cube_size,num_channels), strides=(4, 4, 4), \
                         data_format="channels_last",name = 'conv_layer1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding = 'same',name = 'conv_layer2'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer3'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer4'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer5'))
        model.add(Flatten())
        model.add(Dense(50,name = 'dense_layer1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(10,name = 'attribute_layer'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(num_classes, name = 'pre-softmax_layer'))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))"""
        
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.drop2(self.conv2(x))))
        x = F.relu(self.bn3(self.drop3(self.conv3(x))))   
        x = F.relu(self.bn4(self.drop4(self.conv4(x)))) 
        x = self.conv5(x)
        x = x.view(-1, 400)
        x = F.relu(self.bn_ln1(self.linear1(x)))
        x = F.relu(self.bn_ln2(self.linear2(x)))
        logits = self.bn_logits(self.logits(x))
        return logits