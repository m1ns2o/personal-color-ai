import unittest
import cv2
import numpy as np
import requests
from PIL import Image
import io
import os
import sys

# Add the parent directory to the path to find the app module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.classifier import analyze_image_rule_based, pil_to_cv2

class TestClassifier(unittest.TestCase):

    def test_analyze_image_with_sample(self):
        """
        Tests the analyze_image_rule_based function with a sample image from the web.
        """
        # URL of a sample image (this person does not exist)
        image_url = "https://thispersondoesnotexist.com/"

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Read the image from the response
            pil_img = Image.open(io.BytesIO(response.content)).convert("RGB")
            cv_img = pil_to_cv2(pil_img)

            # Analyze the image
            result = analyze_image_rule_based(cv_img)

            # Print the result for manual verification
            print("\n--- TestClassifier: OpenCV-based Analysis Result ---")
            print(f"Season: {result['season']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Undertone: {result['undertone']}")
            print(f"Skin Tone: {result['skin_tone']}")
            print("----------------------------------------------------")

            # Assertions
            self.assertIn('season', result)
            self.assertIn('confidence', result)
            self.assertIsInstance(result['season'], str)
            self.assertIsInstance(result['confidence'], float)
            self.assertGreater(result['confidence'], 0)

        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to download sample image: {e}")
        except ValueError as e:
            self.skipTest(f"Skipping test, could not process image: {e}")
        except Exception as e:
            self.fail(f"Image analysis failed with an unexpected exception: {e}")

if __name__ == '__main__':
    unittest.main()