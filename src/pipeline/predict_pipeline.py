# import os, sys
# import pandas as pd
# from flask import request
# from dataclasses import dataclass

# from src.logger import logging
# from src.exception import CustomException
# from src.constant import TARGET_COLUMN
# from src.utils.main_utils import MainUtils


# @dataclass
# class PredictionFileDetail:
#     prediction_output_dirname: str = "predictions"
#     prediction_file_name: str = "predicted_file.csv"
#     prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)


# class PredictionPipeline:
#     def __init__(self, request: request):
#         self.request = request
#         self.utils = MainUtils()
#         self.prediction_file_detail = PredictionFileDetail()

#     def save_input_file(self) -> str:
#         try:
#             input_dir = "prediction_artifacts"
#             os.makedirs(input_dir, exist_ok=True)

#             input_file = self.request.files['file']
#             input_path = os.path.join(input_dir, input_file.filename)
#             input_file.save(input_path)

#             return input_path
#         except Exception as e:
#             raise CustomException(e, sys)

#     def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
#         try:
#             preprocessor = self.utils.load_object("artifacts/preprocessor.pkl")

#             # ✅ Drop any columns not seen during training
#             expected_features = preprocessor.feature_names_in_
#             df = df[expected_features]

#             transformed = preprocessor.transform(df)
#             return transformed
#         except Exception as e:
#             raise CustomException(e, sys)


#     def predict(self, features):
#         try:
#             model = self.utils.load_object("artifacts/model/model.pkl")
#             return model.predict(features)
#         except Exception as e:
#             raise CustomException(e, sys)

#     def get_predicted_dataframe(self, input_path: str):
#         try:
#             df = pd.read_csv(input_path)
#             X = self.preprocess(df)
#             preds = self.predict(X)

#             df[TARGET_COLUMN] = [0 if p == 0 else 1 for p in preds]
#             df[TARGET_COLUMN] = df[TARGET_COLUMN].map({0: "phising", 1: "safe"})

#             os.makedirs(self.prediction_file_detail.prediction_output_dirname, exist_ok=True)
#             df.to_csv(self.prediction_file_detail.prediction_file_path, index=False)

#             logging.info("✅ Prediction completed.")
#         except Exception as e:
#             raise CustomException(e, sys)

#     def run_pipeline(self):
#         try:
#             input_path = self.save_input_file()
#             self.get_predicted_dataframe(input_path)
#             return self.prediction_file_detail
#         except Exception as e:
#             raise CustomException(e, sys)

import os, sys
import pandas as pd
from flask import request
from dataclasses import dataclass
from urllib.parse import urlparse
import re
import socket
import requests
from bs4 import BeautifulSoup

from src.logger import logging
from src.exception import CustomException
from src.constant import TARGET_COLUMN
from src.utils.main_utils import MainUtils


@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "predicted_file.csv"
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)


class PredictionPipeline:
    def __init__(self, request: request = None):
        self.request = request
        self.utils = MainUtils()
        self.prediction_file_detail = PredictionFileDetail()

    def save_input_file(self) -> str:
        input_dir = "prediction_artifacts"
        os.makedirs(input_dir, exist_ok=True)
        input_file = self.request.files['file']
        input_path = os.path.join(input_dir, input_file.filename)
        input_file.save(input_path)
        return input_path

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessor = self.utils.load_object("artifacts/preprocessor.pkl")
        expected_features = preprocessor.feature_names_in_
        df = df[expected_features]
        return preprocessor.transform(df)

    def predict(self, features):
        model = self.utils.load_object("artifacts/model/model.pkl")
        return model.predict(features)

    def get_predicted_dataframe(self, df: pd.DataFrame):
        X = self.preprocess(df)
        preds = self.predict(X)
        df[TARGET_COLUMN] = [0 if p == 0 else 1 for p in preds]
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({0: "phishing", 1: "safe"})
        os.makedirs(self.prediction_file_detail.prediction_output_dirname, exist_ok=True)
        df.to_csv(self.prediction_file_detail.prediction_file_path, index=False)

    def run_pipeline(self):
        input_path = self.save_input_file()
        df = pd.read_csv(input_path)
        self.get_predicted_dataframe(df)
        return self.prediction_file_detail

    def extract_full_features(self, url: str) -> pd.DataFrame:
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            html = ""
            try:
                html = requests.get(url, timeout=5).text
            except:
                pass
            soup = BeautifulSoup(html, "html.parser")

            features = {
                "having_IP_Address": int(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) is not None),
                "URL_Length": len(url),
                "Shortining_Service": int(any(s in url for s in ["bit.ly", "tinyurl", "goo.gl"])),
                "having_At_Symbol": int("@" in url),
                "double_slash_redirecting": int(url.find("//") > 7),
                "Prefix_Suffix": int("-" in hostname),
                "having_Sub_Domain": int(len(hostname.split(".")) > 3),
                "SSLfinal_State": int(parsed.scheme == "https"),
                "Domain_registeration_length": 1,  # placeholder
                "Favicon": int("favicon" in html),
                "port": 0,  # placeholder
                "HTTPS_token": int("https" in url.lower()),
                "Request_URL": int("request" in html),
                "URL_of_Anchor": int("href" in html),
                "Links_in_tags": int("link" in html),
                "SFH": 0,  # placeholder
                "Submitting_to_email": int("mailto:" in html),
                "Abnormal_URL": int("about:blank" in url),
                "Redirect": int("window.location" in html),
                "on_mouseover": int("onmouseover" in html),
                "RightClick": int("event.button==2" in html),
                "popUpWidnow": int("popup" in html),
                "Iframe": int("<iframe" in html),
                "age_of_domain": 1,  # placeholder
                "DNSRecord": 1,  # placeholder
                "web_traffic": 1,  # placeholder
                "Page_Rank": 1,  # placeholder
                "Google_Index": int("index" in html),
                "Links_pointing_to_page": html.count("href"),
                "Statistical_report": int("phishtank" in html or "stopbadware" in html),
            }

            return pd.DataFrame([features])
        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_url(self, url: str) -> str:
        df = self.extract_full_features(url)
        X = self.preprocess(df)
        pred = self.predict(X)[0]
        return "phishing" if pred == 0 else "safe"
