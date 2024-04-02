import torch
import torch.nn as nn


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


def normalized_landmarks(hand_landmarks, height, width) -> list[(float, float)]:  # mp.landmark => [(int, int)]
    bbox = landmark_bbox(hand_landmarks, height, width)
    x_min, y_min, x_max, y_max = bbox
    bbox_W, bbox_H = (x_max - x_min), (y_max - y_min)
    normalized_landmarks = []
    for landmark in hand_landmarks.landmark:
        x = (landmark.x * width - x_min) / bbox_W
        y = (landmark.y * height - y_min) / bbox_H
        normalized_landmarks.append((x, y))
    return normalized_landmarks


def to_tensor(positions, device):  # [(float, float)] => tensor[42]
    positions = torch.Tensor(positions).type(torch.float).flatten().to(device)
    return positions


CODE_TO_STRING = {
    0: 'clenched_fist',
    1: 'index_finger_up',
    2: 'open_palm'
}

STRING_TO_CODE = {
    'clenched_fist': 0,
    'index_finger_up': 1,
    'open_palm': 2
}


class HandGesture:
    def __init__(self, PATH='hand_gesture_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(42, 20, 3).to(self.device)
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def predict(self, landmark: list[(float, float)]) -> (str, float):
        input = to_tensor(landmark, self.device)
        pred = self.model(input)
        pred_string, pred_confidence = CODE_TO_STRING[pred.argmax().item()], pred.max().item()
        return pred_string, pred_confidence


if __name__ == '__main__':
    # load model
    MODEL_PATH = 'hand_gesture_model.pth'
    obj = HandGesture(MODEL_PATH)
