import unittest
import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import pandas as pd

# Add the parent directory to the path to find the app module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.classifier import analyze_image, pil_to_cv2

class TestAccuracy(unittest.TestCase):

    DATASET_PATH = Path(__file__).parent.parent / "personal color.v1i.multiclass"
    
    # Map English folder names (from CSV columns) to Korean season names
    SEASON_MAP = {
        "spring": "봄",
        "summer": "여름",
        "fall": "가을", # Roboflow uses 'fall' for autumn
        "winter": "겨울",
    }

    def test_dataset_accuracy(self):
        """
        Tests the accuracy of the personal color classifier against the provided dataset.
        The dataset is expected to have _classes.csv files for labeling.
        """
        if not self.DATASET_PATH.exists():
            self.fail(f"Dataset not found at {self.DATASET_PATH}. Please ensure the dataset is in the correct location.")

        total_images = 0
        correct_predictions = 0
        results_by_season = {season: {"correct": 0, "total": 0} for season in self.SEASON_MAP.values()}

        print(f"\n--- Running Accuracy Test on Dataset: {self.DATASET_PATH} ---")

        # Iterate through train, valid, test directories
        for subset_dir_name in ["test"]:
            subset_path = self.DATASET_PATH / subset_dir_name
            if not subset_path.is_dir():
                print(f"Warning: Subset directory '{subset_dir_name}' not found. Skipping.")
                continue

            classes_csv_path = subset_path / "_classes.csv"
            if not classes_csv_path.exists():
                print(f"Warning: _classes.csv not found in '{subset_path}'. Skipping subset.")
                continue

            print(f"\nProcessing subset: {subset_dir_name}")
            df_classes = pd.read_csv(classes_csv_path)
            df_classes.columns = df_classes.columns.str.strip() # Strip whitespace from column names
            # print(f"DataFrame columns: {df_classes.columns.tolist()}") # Debug print



            for index, row in df_classes.iterrows():
                filename = row["filename"]
                image_path = subset_path / filename

                if not image_path.exists():
                    print(f"Warning: Image file '{image_path}' not found. Skipping.")
                    continue

                # Determine ground truth season from CSV row
                ground_truth_season_en = None
                one_hot_columns = ['fall', 'spring', 'summer', 'winter']
                
                # Filter the row to only include the one-hot encoded columns
                season_values = row[one_hot_columns]
                
                # Find the column name where the value is 1
                if (season_values == 1).any(): # Check if any season is marked 1
                    ground_truth_season_en = season_values.idxmax()
                
                if ground_truth_season_en is None:
                    print(f"Warning: No ground truth season found for {filename}. Skipping.")
                    continue
                
                expected_season_ko = self.SEASON_MAP[ground_truth_season_en]
                
                total_images += 1
                results_by_season[expected_season_ko]["total"] += 1

                try:
                    pil_img = Image.open(image_path).convert("RGB")
                    cv_img = pil_to_cv2(pil_img)
                    result = analyze_image(cv_img)
                    predicted_season_ko = result['season']

                    is_correct = (predicted_season_ko == expected_season_ko)
                    if is_correct:
                        correct_predictions += 1
                        results_by_season[expected_season_ko]["correct"] += 1
                    
                    # Optional: print detailed results for each image
                    # print(f"  {filename}: Expected={expected_season_ko}, Predicted={predicted_season_ko}, Correct={is_correct}")

                except ValueError as e:
                    print(f"  Skipping {filename} due to error: {e}")
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
        
        print("\n--- Accuracy Report ---")
        print(f"Total images processed: {total_images}")
        print(f"Correct predictions: {correct_predictions}")
        
        overall_accuracy = (correct_predictions / total_images * 100) if total_images > 0 else 0
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")

        print("\nAccuracy by Season:")
        for season, data in results_by_season.items():
            season_accuracy = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
            print(f"  {season}: {season_accuracy:.2f}% ({data['correct']}/{data['total']})")

        self.assertGreater(overall_accuracy, 0, "Overall accuracy is 0%. Check dataset or classifier logic.")
        print("-----------------------")

if __name__ == '__main__':
    unittest.main()