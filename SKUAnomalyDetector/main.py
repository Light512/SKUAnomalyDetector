import pandas as pd
from src.config import IMAGE_ROOT, OUTPUT_CSV
from src.anomaly_detector import load_images_and_extract_features, detect_anomalies

def main():
    print("Loading and processing images...")
    features, image_paths = load_images_and_extract_features(IMAGE_ROOT)

    if len(features) < 3:
        print("Too few images to cluster. Exiting.")
        return

    results = detect_anomalies(features, image_paths, contamination=0.05)

    print(f"Saving results to {OUTPUT_CSV}...")
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
