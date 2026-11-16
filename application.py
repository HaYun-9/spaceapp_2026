import torch
import torch.nn as nn

import pandas as pd

# Define necessary globals
evaluated_metrics = ['RH2M','WS2M','T2M','T2M_MAX','T2M_MIN','WS50M','WD2M','WD50M']
crop_names = pd.read_csv("possible_crops.csv")
raw_to_contiguous = {raw_id: idx for idx, raw_id in enumerate(crop_names.index)}


class PredictorModel(nn.Module):
    def __init__(self, num_outputs: int, num_inputs: int):

        super(PredictorModel, self).__init__()

        hidden_1 = 64
        hidden_2 = 128

        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, num_outputs)
        )
        

    def forward(self, x):

        return self.net(x)

device = torch.device('cuda' if torch.backends.cuda.is_built() else 'cpu')

model = PredictorModel(num_outputs=256, num_inputs=len(evaluated_metrics))
model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))

model.eval()

desired_date = 20261114

# Get the location of the user.
import geocoder
import requests

g = geocoder.ip('me')
lat = g.latlng[0]
lng = g.latlng[1]

# Request json data for the current requested URL
nasa_api_url = f"https://power.larc.nasa.gov/api/temporal/daily/point?start={desired_date}&end={desired_date}&latitude={lat}&longitude={lng}&community=ag&parameters=RH2M%2CWS2M%2CT2M%2CT2M_MAX%2CT2M_MIN%2CWS50M%2CALLSKY_SFC_LW_DWN%2CWD2M%2CWD50M%2CALLSKY_SFC_SW_DWN&format=json&units=metric&header=false&time-standard=utc"
response = requests.get(nasa_api_url)
response.raise_for_status()

# Pull the desired metrics from the weather locational data.
my_info = response.json()['properties']['parameter']
data = {
    metric: my_info[metric][str(desired_date)] for metric in evaluated_metrics
}

# Load the obtained data to the same device as the neural network
data = pd.DataFrame(data, index=[0])
data = torch.tensor(data.to_numpy(), dtype=torch.float32)

# Try and predict which croplands/biomes would be most likely to occur here.
tensor_x = data.to(device)
model.to(device)
tensor_y = model(tensor_x)
tensor_y.to('cpu')

# Extract the output into a usable format, ordered from most likely to be usable to least likely to be usable in this climate.
outputs = tensor_y.tolist()

output_dict = {
    crop_names[crop_names['crop_int'] == i]['Class_Names'].item(): output for i, output in enumerate(outputs[0])
}
output_dict = dict(sorted(output_dict.items(), key=lambda item: item[1], reverse=True))

print(f"=== TOP SUGGESTED CROP/ENVIRONMENT TYPES FOR {lat}, {lng}, RANKED BEST-WORST ===")
rank = 1
for crop_type in output_dict.keys():

    if crop_type == "UNKNOWN" or crop_type == "":
        continue
    print(f'{rank}: {crop_type}')
    rank += 1