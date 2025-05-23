import requests
import pandas as pd
import numpy as np


class SimulationConnection:
    def __init__(self, base_url: str = "http://localhost:5256/device"):
        self.base_url = base_url
        self.session = requests.Session()

    def get_profile(self):
        response = self.session.get(f"{self.base_url}/GetProfile")
        response.raise_for_status()
        result = np.array(response.json(), dtype=np.float32)
        return result

    def get_profile_ts(self):
        response = self.session.get(f"{self.base_url}/GetProfileTS")
        response.raise_for_status()
        result = np.array(response.json(), dtype=np.float32)
        return result

    def get_actuator_position(self):
        response = self.session.get(f"{self.base_url}/GetActuatorPosition")
        response.raise_for_status()
        result = np.array(response.json(), dtype=np.float32)
        return result

    def get_actuator_position_ts(self):
        response = self.session.get(f"{self.base_url}/GetActuatorPositionTS")
        response.raise_for_status()
        result = np.array(response.json(), dtype=np.float32)
        return result

    def get_actuator_deviation(self):
        response = self.session.get(f"{self.base_url}/GetActuatorDeviation")
        response.raise_for_status()
        result = np.array(response.json(), dtype=np.float32)
        return result

    def get_actuator_deviation_ts(self):
        response = self.session.get(f"{self.base_url}/GetActuatorDeviationTS")
        response.raise_for_status()
        result = np.array(response.json(), dtype=np.float32)
        return result

    def get_sigma_2(self):
        response = self.session.get(f"{self.base_url}/GetSigma2")
        response.raise_for_status()
        result = np.array(response.json(), dtype=np.float32)
        return result

    def set_actuator_position(self, positions):
        response = self.session.post(
            f"{self.base_url}/SetActuatorPosition", json=positions)
        response.raise_for_status()

    def set_actuator_action(self, actions):
        response = self.session.post(
            f"{self.base_url}/SetActuatorAction", json=actions)
        response.raise_for_status()
