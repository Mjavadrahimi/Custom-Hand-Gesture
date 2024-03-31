import torch
import torch.nn as nn
import mediapipe as mp


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.L1 = nn.Linear(input_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.L1(input)
        output = self.relu(output)
        output = self.L2(output)
        output = self.sigmoid(output)
        return output


def landmark_bbox(hand_landmarks, height, width) -> [float, float, float, float]:
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    for landmark in hand_landmarks.landmark:
        x, y = landmark.x * width, landmark.y * height
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    return x_min, y_min, x_max, y_max


def normalized_landmarks(hand_landmarks, bbox, height, width) -> list[(float, float)]:
    x_min, y_min, x_max, y_max = bbox
    bbox_W, bbox_H = (x_max - x_min), (y_max - y_min)
    normalized_landmarks = []
    for landmark in hand_landmarks.landmark:
        x = (landmark.x * width - x_min) / bbox_W
        y = (landmark.y * height - y_min) / bbox_H
        normalized_landmarks.append((x, y))
    return normalized_landmarks


class HandGesture:
    def __init__(self, PATH):
        self.model = Model(42, 20, 3)
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def predict(self, input) -> int:
        pred = self.model(input)

        return pred

    def landmark_to_tensor(self, hand_landmarks):
        hand_landmarks
        return


MODEL_PATH = 'hand_gesture_model.pth'
if __name__ == '__main__':
    # load model
    obj = HandGesture(MODEL_PATH)
